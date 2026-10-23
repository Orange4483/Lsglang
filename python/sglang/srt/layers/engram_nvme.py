"""Opt-in exact NVMe Engram lookup on the lk_moe N1 async filler.

The Engram table lives in its immutable local safetensors shard; rows are
gathered with the same capture-clean doorbell protocol as the PLE NVMe cache:
a lookup kernel publishes missed rows into a pinned ring and bumps the
doorbell, the lk_moe filler thread drains, sort/run-merges, and preads rows
into pinned stage slots, and a merge kernel reads the staged bytes over PCIe
and dequantizes (fp8 e4m3 payload, e8m0 block scales) to bf16. Nothing here
touches CUDA from the host side: no host callbacks, so CUDA-graph capture and
replay are exact, and no table copy exists anywhere (RAM peak at load is
zero). Cache and staging knobs from the pre-N1 implementation are accepted
and ignored: the N1 path stores nothing but the pinned stage.

Backend precedence for the Engram table (LK_ENGRAM_NVME
(host-table flag SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE, upstream):
NVMe set means host-table mode is rejected; unset falls through to host
table when requested.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_emulate import (
    e4m3fn_u8_to_f32,
    fp8_native_supported,
)
from sglang.srt.utils import logger
from sglang.srt.utils.common import (
    is_lk_engram_nvme_enabled,
    lk_nvme_cache_bytes,
    lk_nvme_direct,
)

# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127
# Per Engram layer: the ring bounds the in-flight depth (a wrap-over-unserved
# id trips the gate), so 64K slots = 16MB pinned stage per layer across every
# plausible prefill/DP-replicate batch. Larger gathers walk in chunks.
_RING = 1 << 16
_DTK_MASK = _RING - 1
_STAGE = _RING
_CHUNK = _RING // 2
_GATE_LIMIT_US = 300000


def checkpoint_table(model_path, layer_id, rows, dim):
    """Return validated local shard and offsets without materializing tensors."""
    if dim != 256 or rows < 1:
        raise ValueError("NVMe Engram requires nonempty 256-wide FP8 tables")
    root = Path(model_path).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("NVMe Engram requires a local indexed safetensors checkpoint")
    index = json.loads(index_path.read_text())["weight_map"]
    prefix = f"layers.{layer_id}.engram.embed."
    names = [prefix + "weight", prefix + "scale"]
    if any(name not in index for name in names):
        raise ValueError(f"Missing Engram checkpoint entries for layer {layer_id}")
    if index[names[0]] != index[names[1]]:
        raise ValueError("NVMe Engram weight and scale must share a checkpoint shard")
    shard = (root / index[names[0]]).resolve()
    if not shard.is_relative_to(root):
        raise ValueError("Engram shard must be inside the checkpoint directory")
    with shard.open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(source.read(length))
    offsets = []
    extents = []
    for name, dtype, width in zip(names, ("F8_E4M3", "F8_E8M0"), (256, 8)):
        tensor = header.get(name, {})
        span = tensor.get("data_offsets", [])
        if (
            tensor.get("dtype") != dtype
            or tensor.get("shape") != [rows, width]
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] <= span[1] <= size - 8 - length
            or span[1] - span[0] != rows * width
        ):
            raise ValueError(f"Invalid Engram tensor metadata: {name}")
        extents.append(span)
        offsets.append(8 + length + span[0])
    if max(x[0] for x in extents) < min(x[1] for x in extents):
        raise ValueError("Overlapping Engram weight and scale extents")
    return shard, offsets


def validate_configuration(parallel, schedule, spec, graph, host_table):
    if sys.platform != "linux" or torch.version.cuda is None:
        raise ValueError("NVMe Engram requires Linux and CUDA")
    if parallel.pp_size != 1:
        raise ValueError("NVMe Engram requires pipeline parallelism = 1")
    if graph.enable_torch_compile:
        raise ValueError("NVMe Engram does not support torch.compile")
    if spec.speculative_algorithm not in (None, "DSPARK"):
        raise ValueError(
            "NVMe Engram supports only ordinary decode or fixed-width DSpark"
        )
    if spec.speculative_algorithm == "DSPARK":
        from sglang.srt.environ import envs

        width = getattr(spec, "speculative_num_draft_tokens", None)
        gamma = getattr(spec, "speculative_dspark_block_size", None)
        if width not in (4, 6) or gamma != width - 1:
            raise ValueError(
                "NVMe Engram DSpark requires block size 3 or 5 and "
                "speculative_num_draft_tokens = block size + 1"
            )
        if envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
            raise ValueError(
                "NVMe Engram DSpark requires SGLANG_RAGGED_VERIFY_MODE=static"
            )
    if host_table:
        raise ValueError("NVMe Engram and host-table mode are mutually exclusive")


@triton.jit
def _nvme_lookup(ids_ptr, d_tk_ptr, h_ring_ptr, h_cnt_ptr, stat_ptr, row_hi, ring_mask):
    r = tl.program_id(0)
    id = tl.load(ids_ptr + r)
    h_cnt = h_cnt_ptr.to(tl.pointer_type(tl.int64))
    h_ring = h_ring_ptr.to(tl.pointer_type(tl.int64))
    tl.atomic_add(stat_ptr, 1)
    if (id >= 0) & (id < row_hi):
        pos = tl.atomic_add(h_cnt, 1)
        tl.store(h_ring + (pos & ring_mask), id)
        tl.store(d_tk_ptr + (r & 65535), pos)
    else:
        tl.store(d_tk_ptr + (r & 65535), 0)
        tl.atomic_add(stat_ptr + 2, 1)


@triton.jit
def _nvme_merge(
    ids_ptr,
    out_ptr,
    d_tk_ptr,
    h_stamp_ptr,
    h_stage_ptr,
    stat_ptr,
    row_hi,
    stage_mask,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    FP8_PTR: tl.constexpr,
):
    r = tl.program_id(0)
    id = tl.load(ids_ptr + r)
    owned = (id >= 0) & (id < row_hi)
    tk = tl.load(d_tk_ptr + (r & 65535))
    slot = tk & stage_mask
    # Exact staging-fault detection: the filler stamps slot with ticket+1
    # after the row lands; any other value means the row never arrived.
    if tl.load(h_stamp_ptr.to(tl.pointer_type(tl.int64)) + slot) != tk + 1:
        if owned:
            tl.atomic_add(stat_ptr + 1, 1)
    base = h_stage_ptr.to(tl.int64) + slot.to(tl.int64) * (DIM + DIM // BLK)
    offs = tl.arange(0, DIM)
    if FP8_PTR:
        vals = tl.load(
            base.to(tl.pointer_type(tl.float8e4nv)) + offs, mask=owned, other=0.0
        ).to(tl.float32)
    else:
        b = tl.load(base.to(tl.pointer_type(tl.uint8)) + offs, mask=owned, other=0)
        vals = e4m3fn_u8_to_f32(b)
    exps = tl.load(
        (base + DIM).to(tl.pointer_type(tl.uint8)) + offs // BLK, mask=owned, other=0
    ).to(tl.int32)
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = tl.where(owned, vals * scale, 0.0)
    tl.store(out_ptr + r * DIM + offs, out.to(tl.bfloat16))


class NvmeEngramStore:
    """One Engram layer's rows served by the lk_moe N1 filler (one slot each).

    ``row_start``/``shard_rows`` are the caller's TP shard in global table rows
    (``shard_range``): they are the only split rule, so the filler and the
    lookup kernel agree on which ids this rank owns. Ids stay shard-local end
    to end (``EngramEmbedding`` subtracts ``row_start``), while the checkpoint
    offsets point at global row 0, so both file offsets are rebased here once:
    local id ``i`` preads global row ``row_start + i``.
    """

    def __init__(
        self,
        model_path,
        layer_id,
        rows,
        dim,
        cache_bytes,
        staging_bytes,
        *,
        row_start: int,
        shard_rows: int,
    ):
        import lk_moe

        if not 0 <= row_start <= row_start + shard_rows <= rows:
            raise ValueError(
                f"Engram shard [{row_start}, {row_start + shard_rows}) is not "
                f"inside the table of {rows} rows"
            )
        self.rows = rows
        self.row_hi = shard_rows
        self._loaded = set()
        shard, (w_off, s_off) = checkpoint_table(model_path, layer_id, rows, dim)
        w_off += row_start * dim
        s_off += row_start * (dim // 32)  # scale blocks per row, as validated above
        dev = torch.device("cuda")
        self._h_ring = torch.full(
            (_RING,), -1, dtype=torch.int64, device="cpu"
        ).pin_memory()
        self._h_cnt = torch.zeros(2, dtype=torch.int64, device="cpu").pin_memory()
        self._h_stamp = torch.zeros(
            _STAGE, dtype=torch.int64, device="cpu"
        ).pin_memory()
        self._h_stage = torch.zeros(
            (_STAGE, 264), dtype=torch.uint8, device="cpu"
        ).pin_memory()
        self._d_tk = torch.zeros(_RING, dtype=torch.int64, device=dev)
        self._err = torch.zeros(1, dtype=torch.int64, device=dev)
        self._stat = torch.zeros(3, dtype=torch.int64, device=dev)
        self._pin = torch.zeros(3, dtype=torch.int64, device="cpu").pin_memory()
        self._evt = torch.cuda.Event()
        self._n = 0
        fd = os.open(str(shard), os.O_RDONLY)
        direct_fd = 0
        if lk_nvme_direct(is_lk_engram_nvme_enabled()):
            direct_fd = os.open(str(shard), os.O_RDONLY | os.O_DIRECT)
        self._slot = lk_moe.nvme_filler_attach_gather(
            fd,
            w_off,
            s_off,
            256,
            8,
            self.row_hi,
            _DTK_MASK,
            self._h_ring.data_ptr(),
            self._h_cnt.data_ptr(),
            [self._h_stage.data_ptr()],
            _STAGE,
            self._h_stamp.data_ptr(),
            direct_fd,
            lk_nvme_cache_bytes(is_lk_engram_nvme_enabled()),
        )
        if self._slot < 0:
            os.close(fd)
            raise RuntimeError("NVMe Engram N1 filler attach failed")
        # The worker preads through this descriptor for the process lifetime;
        # captured graphs reference the pinned buffers, so neither is released.
        self._fd = fd
        _register(self)

    def validate_weight(self, kind, source):
        if kind in self._loaded:
            raise RuntimeError(
                "NVMe Engram checkpoints are immutable; restart to reload"
            )
        width, dtype = (
            (256, torch.float8_e4m3fn)
            if kind == "weight"
            else (8, torch.float8_e8m0fnu)
        )
        if list(source.shape) != [self.rows, width] or source.dtype != dtype:
            raise ValueError(f"Engram {kind} checkpoint shape/dtype mismatch")
        self._loaded.add(kind)

    def close(self):
        raise RuntimeError(
            "Captured NVMe Engram storage is owned by the worker until exit"
        )

    def finish_load(self, label: str = ""):
        # ``checkpoint_table`` already parsed the safetensors header for the
        # weight/scale pair, so an empty tally means the loader skipped the
        # tensors (storage-backed, RAM-zero). Anything else must have loaded
        # and been shape-checked, but must not be partial.
        if self._loaded == set():
            return
        if self._loaded != {"weight", "scale"}:
            raise ValueError(
                "NVMe Engram checkpoint did not load both weight and scale"
            )

    def stats(self):
        import lk_moe

        out = dict(lk_moe.nvme_filler_stats(self._slot))
        self._pin.copy_(self._stat, non_blocking=True)
        torch.cuda.synchronize()
        look, bad, bad_id = (int(x) for x in self._pin.tolist())
        out["lookups"] = look
        out["bad"] = bad
        out["bad_ids"] = bad_id
        return out

    def lookup(self, indices):
        import lk_moe

        count = indices.numel()
        out = torch.empty(
            (*indices.shape, 256), dtype=torch.bfloat16, device=indices.device
        )
        if not count:
            return out
        if indices.dtype != torch.int64 or not indices.is_cuda:
            raise ValueError("Engram ids must be int64 CUDA ids")
        stream = torch.cuda.current_stream().cuda_stream
        flat = indices.reshape(-1)
        rows = out.view(-1, 256)
        for off in range(0, count, _CHUNK):
            n = min(_CHUNK, count - off)
            _nvme_lookup[(n,)](
                flat[off:],
                self._d_tk,
                self._h_ring.data_ptr(),
                self._h_cnt.data_ptr(),
                self._stat,
                self.row_hi,
                _DTK_MASK,
                num_warps=1,
            )
            lk_moe.nvme_gate(
                self._h_cnt.data_ptr(), self._err.data_ptr(), _GATE_LIMIT_US, stream
            )
            _nvme_merge[(n,)](
                flat[off:],
                rows[off:],
                self._d_tk,
                self._h_stamp.data_ptr(),
                self._h_stage.data_ptr(),
                self._stat,
                self.row_hi,
                _DTK_MASK,
                DIM=256,
                BLK=32,
                E8M0_ZERO=_E8M0_ZERO,
                FP8_PTR=fp8_native_supported(),
            )
        self._tick()
        return out

    def _tick(self):
        import lk_moe

        self._n += 1
        if self._n % 64 == 0 and not torch.cuda.is_current_stream_capturing():
            self._pin.copy_(self._stat, non_blocking=True)
            self._evt.record()
            if self._evt.query():
                look, bad, bad_id = (int(x) for x in self._pin.tolist())
                if self._n % 1024 == 0:
                    fs = lk_moe.nvme_filler_stats(self._slot)
                    logger.info(
                        "[engram-nvme] layer-lookups=%d bad=%d bad_ids=%d"
                        " filler(cycles=%d rows=%d preads=%d dedup=%d MB=%d)",
                        look,
                        bad,
                        bad_id,
                        fs["cycles"],
                        fs["rows"],
                        fs["preads"],
                        fs["dedup"],
                        fs["bytes"] >> 20,
                    )
                    if bad or bad_id:
                        raise RuntimeError(
                            "NVMe Engram gather fault: bad=%d bad_ids=%d"
                            " gate_err=%d" % (bad, bad_id, int(self._err.sum().item()))
                        )
                self._stat.zero_()


_INSTANCES: list[NvmeEngramStore] = []


def _register(store):
    _INSTANCES.append(store)


def shutdown():
    """Tear down every filler worker (process exit; captured graphs gone)."""
    import lk_moe

    lk_moe.nvme_filler_detach_all()
    _INSTANCES.clear()
