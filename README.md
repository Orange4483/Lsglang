# Lsglang — lk_moe Hybrid Inference for sglang

Lsglang is a special extension of [sglang](https://github.com/sgl-project/sglang) that adds
**CPU-GPU hybrid (MOE) inference** on top of the latest sglang release version, fully compatible
with stock sglang.

The actual hybrid inference engine is **[lk_moe](https://pypi.org/project/lk-moe/)**, sglang/vllm
only provide the "GPU path", lk_moe provides the "hybrid path". Lsglang is the concrete integration
case of lk_moe into sglang.

> **Release policy:** Lsglang version updates are released **in sync with sglang releases** — on top of
> a fresh sglang tag we keep the code "as-is + lk_moe". We do **not** pile on extra features; unless a
> necessary bug-fix patch is required, the diff against upstream stays minimal (just the lk_moe layer).

---

Note 1: x86 CPUs with AVX2+ instruction sets and Nvidia GPUs with sm80+ architectures.

---

## Layer compute modes

The modes below are **per-MoE-layer runtime roles**: within one model, different layers can run
in different modes at the same time (attention is always on the GPU). Speculative decoding is the
exception — it is a global switch; GPU Pool is global today as well (per-layer pool control is on
the roadmap).

### Layer compute modes (per layer)

| # | Mode | Decode path | Prefill path | Control | When to use |
|---|---|---|---|---|---|
| 1 | **GPU** (stock sglang path) | weights resident in VRAM, original GPU kernels | same | `LK_GPU_RESIDENT_MOE_LAYERS` | Large VRAM; saves host memory, lifts decode and prefill together |
| 2 | **CPU** (hybrid) | MoE on CPU (NUMA host weights), attention on GPU | MoE on CPU, chunk by chunk | default layer role when `LK_MOE_HYBRID_ENABLED=1` | **Recommended for high-performance CPU subsystems** (many cores + high memory bandwidth); zero extra VRAM |
| 3 | **CPU + Speculative decoding** | speculative verification × CPU MoE | same | `LK_MOE_HYBRID_ENABLED=1` + `--speculative-algorithm NEXTN` / `DSPARK` | **Recommended for high-performance CPU subsystems** |
| 4 | **CPU + GPU Pool** (expert cache) | pool-resident experts on GPU, the rest on CPU | same split | `LK_POOL_SLOTS` (e.g. 64) | **Recommended for low-power CPU subsystems + high-performance GPUs** (spare GPU compute + high-bandwidth PCIe) |
| 5 | **CPU + GPU Pool + Speculative decoding** | speculative verification × pool's GPU/CPU halves | same split | `LK_POOL_SLOTS` (e.g. 64) + `--speculative-algorithm NEXTN` / `DSPARK` | Optional depending on task type — gains do not stack fully (best for latency-bound, high single-stream workloads) |

**Overlay — Super Mix Prefill**: GPU prefill with super-chunk sizing (decoupled from the
token budget), runs in parallel with CPU-side decode at near-100% GPU utilization;
coordinates automatically with modes 2-5.
`LK_GPU_PREFILL_MIN_TOKENS` (e.g. 1024) + `LK_GPU_PREFILL_SUB_M` (e.g. 16384).

> **Selection principle**: whenever VRAM has headroom, keep prefill on the GPU
> (Super Mix Prefill and/or pool/GPU-resident modes).

---

## Getting started

### Installation

Prerequisite: **CUDA 13.2.1** (driver + toolkit) — matches the `torch==2.13.0` CUDA build below.

```bash
conda create -n Lsglang python==3.12.11 && conda activate Lsglang
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
sudo apt-get install libnuma-dev      # Ubuntu  /  sudo dnf install numactl-devel  # Rocky

pip install lsglang                   # or build from source below
```

From source:

```bash
git clone https://github.com/guqiong96/Lsglang.git
cd Lsglang
pip install -U setuptools wheel scikit-build-core cmake
pip install torchaudio triton torchvision torch==2.13.0
pip install grpcio-tools wheel-stub
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release \
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
pip install -e "python" --no-build-isolation -vvv
```

(`MAX_JOBS=32 NVCC_THREADS=1`: reduce compile memory; `CMAKE_BUILD_TYPE=Release`: perf option.)

### Quick start (Qwen3.8-Flash-Next [RTX 3090 *2])

Full launch script: [`launch/sglang_serve_qwen38_mtp_tp2_3090.sh`](launch/sglang_serve_qwen38_mtp_tp2_3090.sh) · [ple on nvme](launch/sglang_serve_qwen38_mtp_tp2_3090_nvme.sh).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,3 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_SUB_M=32768 \
LK_POOL_SLOTS=32 \
python -m sglang.launch_server \
    --model /data/models/Qwen3.8-Flash-Next-NVFP4-W4A16-4o6-FP8 \
    --served-model-name Qwen3.8-Flash-Next-NVFP4 \
    --host 0.0.0.0 --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --disable-shared-experts-fusion \
    --disable-custom-all-reduce \
    --ple-offload-embedding --ple-offload-backend pinned \
    --chunked-prefill-size 16384 --max-total-tokens 262144 \
    --mem-fraction-static 0.90 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size 11 \
    --reasoning-parser auto --tool-call-parser auto \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

---

## Configuration parameters

All lk_moe switches use the unified `LK_*` prefix. Legacy `LVLLM_*` names are still accepted: they are auto-renamed at startup with a one-line warning (if both spellings are set, `LK_*` wins). Our knobs never use the `SGLANG_*` prefix.

### Switch linkage (one master gate)

`LK_MOE_HYBRID_ENABLED` is the single master switch: with it off (the default) `lk_moe` is never imported and **every `LK_*` below is inert** — the server runs the stock sglang MoE path bit-for-bit. Every downstream tier is ANDed with it (switches only — perf knobs never appear in a linkage):

```
LK_MOE_HYBRID_ENABLED=1                     master
 ├─ pool on            LK_POOL_SLOTS (e.g. 64)      (dual knob: value itself is the switch, 0 = off)
 │   └─ pool decode    LK_POOL_DECODE=1     (unset = follows pool on; explicit `1`/`0` wins)
 ├─ GPU prefill on     LK_GPU_PREFILL_MIN_TOKENS (e.g. 1024)   (dual knob, same rule)
 │   └─ super mix      LK_GPU_PREFILL_SUB_M>0        (super-chunk staging on; 0 = per-chunk staging)
 ├─ nvme filler on    LK_NVME_DIRECT=1          (needs the feature's own backend flag: --ple-offload-backend nvme / engram host table;
 └─ resident layers   LK_GPU_RESIDENT_MOE_LAYERS*   (complementary: a resident layer never pools)
LK_MOE_HYBRID_ENABLED=0  →  every branch above unreachable = original sglang
( LK_NVME_DIRECT with the engram/PLE nvme backends is the one knob that also works without the LK master )
```

### Core

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_MOE_HYBRID_ENABLED` | core | `0` | **master switch**: `1`-on, `0`-off (= stock sglang, all `LK_*` inert) |
| `LK_MOE_LAYERWISE_LOAD` | load | 0 | `1`: load MoE layers one by one to cut peak load-time VRAM |

### CPU / NUMA / power

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_THREADS` | perf | cores/GPU | host decode thread count (read directly by lk_moe at bootstrap) |
| `LK_NUMA_ILP` | perf | `0` | `1`: interleaved ILP mode in the NUMA allocator (pairs with `LK_NUMA_HUGE`; read by lk_moe) |
| `LK_THREAD_BINDING` | perf | `CPU_CORE` | `CPU_CORE` bind by core, `NUMA_NODE` bind by node |
| `LK_NUMA_INTERLEAVE` | perf | 1 | `1`: interleave host weights across NUMA nodes (avoids node OOM; required for the pump's per-node copy streams); `0` off |
| `LK_NUMA_HUGE` | perf | `0` | `1`: opt-in 2M THP-aligned host slabs (single-node / THP-friendly boxes); NUMA-interleave, not page size, is the usual win |
| `LK_POWER_SAVING` | power | 0 | `1`: CPU power-saving mode on the decode path |
| `LK_HF_MBOX` | perf | `1` | mailbox early-start of host-side decode (non-`0` on, `0` off) |
| `OMP_NUM_THREADS` | perf | - | set to 1 to avoid slow model loading |

### GPU-resident layers

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_GPU_RESIDENT_MOE_LAYERS` | GPU | none | expert layers resident in VRAM, e.g. `0`, `0-1`, `0,9` |
| `LK_GPU_RESIDENT_MOE_LAYERS_SPEC` | GPU | none | draft model layers resident in GPU |
| `LK_GPU_RESIDENT_MOE_LAYERS_MTP` / `LK_GPU_RESIDENT_MOE_LAYERS_DSPARK` | GPU | follow `_SPEC` | per-draft-role override (MTP / DSpark workers) |

### Super Mix Prefill

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_GPU_PREFILL_MIN_TOKENS` | prefill | 0 = off | **tier switch AND batch gate**: the value itself switches (e.g. 1024 turns the tier on; `0` = pure CPU prefill) = minimum token batch for the GPU path; not enabled by default (0 on 16 GB cards) |
| `LK_GPU_PREFILL_SUB_M` | prefill | 0 | super-mix staging: value itself switches (e.g. 16384 enables super-chunk staging; `0` = per-chunk staging, inert when the tier is off) |
| `LK_GPU_PREFETCH_WINDOW` | prefill | `1` | staging-ring blocks in flight; `1` = serialized (default); `2` = ping-pong (copy overlaps gemm, needs fast PCIe to pay off) |
| `LK_GPU_RESERVE_GB` | prefill | 0 | GB of KV budget reserved for lk_moe device-side staging |

### GPU Pool (expert cache)

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_POOL_SLOTS` | pool | 0 = off | **GPU Pool switch AND capacity**: the value itself switches (e.g. 64 turns the pool on; `0` = off, all `LK_POOL_*` inert) = resident expert slots per MOE layer (extra VRAM = slots × layers × expert size); not enabled by default |
| `LK_POOL_DECODE` | pool | follow pool | decode GPU half: pool-resident experts go to the GPU resident-gemv half. `1` on / `0` off explicitly; unset follows `LK_POOL_SLOTS` > 0 |
| `LK_POOL_WINDOW` | pool | 32 | Expected-reuse window in steps: an evicted expert is pumped back only if it is seen again within the window; illegal values clamp to min `8`, never disables |
| `LK_POOL_INFLIGHT` | pool | `16` | in-flight DMA jobs / event-ring depth (backpressure cap = 8x; lower values false-positive under small batches) |
| `LK_POOL_LAYERS` | pool | -- | RETIRED: pool membership is global (`LK_POOL_SLOTS` > 0); per-layer mixing with GPU prefill is not supported |
| `LK_POOL_STAGE` | pool | 1 | coalesced H2D staging ring (read by lk_moe) |
| `LK_POOL_LOG` | pool | 0 | windowed `[pool]` stats line every N decode steps (0 = silent) |

### Speculative decode

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_VERIFY_RESIDENT` | spec | 0 | verify batches cover draft rows' pool-resident experts; requires pool + NEXTN/DSPARK draft rows |
| `LK_VERIFY_SKIP_MISS` | spec | 0 | skip verification for rows whose experts missed residency (`1`: trust CPU path) |

### PLE / NVMe offload

| Knob | Type | Default | Description |
|------|--------|--------|------|
| `--ple-offload-backend nvme` | ple (CLI) | pinned | PLE table stays on storage; loader writes it once, RAM never holds the table; gathers via VRAM tag cache + pread |
| `LK_PLE_BLOCK_MB` | ple | unset | PLE prefill conv temporaries budget in MB (bf16, ~6 B/token/channel); unset = block derived from the chunked-prefill size |
| `LK_NVME_DIRECT` | nvme | `0` | `1`: row filler switches from page-cache pread to O_DIRECT + pinned direct-mapped cache; **only effective under the feature's master switch** |
| `LK_NVME_CACHE_MB` | nvme | `1024` | MB of the O_DIRECT pinned tag cache (inert when `LK_NVME_DIRECT` or master is off) |
| `LK_ENGRAM_NVME` | nvme | `0` | DeepSeek-V4.1 Engram: exact NVMe lookup from the safetensors shard itself (no host table) |
| `LK_ENGRAM_NVME_CACHE_MB` / `LK_ENGRAM_NVME_STAGING_MB` | nvme | `1024` / `128` | Engram NVMe store cache / staging budgets |

### Diagnostics (all default off; no effect on the compute path)

| Env var | Area | Description |
|--------|------|------|
| `LK_ROUTE_PROBE` | routing | per-step `[RP]` routing traces for offline calibration (requires the pool on) |
| `LK_SPEC_PROBE` | spec | CUDA-event timed draft/verify stages |
| `LK_PLE_PROBE` | ple | PLE ngram-row temporal locality + traffic probe (`LK_PLE_PROBE_DUMP=<path>` dumps rows) |
| `LK_PP_DEBUG` | prefill | extra tracing inside the prefill staging tier |
| `LK_POOL_VERIFY` | pool | pump byte-check 1-in-N (read by lk_moe) |
| `LK_DECODE_TS` / `LK_TS_SKIPGEMM` | host decode | decode-path timestamp trace / skip-the-gemm A/B (read by lk_moe) |
| `LK_MBOX_STOPFILE` | mbox | stop-file for orderly teardown (read by lk_moe) |
| `LK_STAGING_M_TILE` | staging | internal | prefill staging M-tile (lk_moe python, default 32) |
| `LK_DISABLE_QSA_SHARE` | draft | draft worker: disable QSA share (A/B isolation only) |
| `SGLANG_RS_DEBUG` | upstream | rejection-sampling traces; upstream sglang knob, kept as-is |



## Tuning recipes

All recipes assume `LK_MOE_HYBRID_ENABLED=1` (off = stock sglang).

| Mode | Parameter recipe |
|---|---|
| **1 GPU-resident layers** | `LK_GPU_RESIDENT_MOE_LAYERS=0-<last>` (format `0,1,8-9`; some models start at a non-zero layer, e.g. Step-3.5-Flash at layer 3); draft model: `LK_GPU_RESIDENT_MOE_LAYERS_SPEC=0-2` |
| **2 CPU hybrid** | default layer role, no extra parameters; tune the CPU side: `LK_THREADS=(physical cores / #GPUs)`, `LK_THREAD_BINDING=CPU_CORE`; give the saved VRAM to the KV cache (`--mem-fraction-static` as high as it fits) |
| **4 CPU + GPU Pool** | `LK_POOL_SLOTS=<from spare VRAM>` (e.g. 64; `>0` is itself the pool switch — there is no separate `LK_POOL` flag; membership covers every non-resident layer); check `LK_POOL_LOG=100` and watch `[pool] win hit=` — raise `LK_POOL_SLOTS` or `LK_POOL_WINDOW` if the hit rate is low |
| **3/5 Speculative decoding** | `--speculative-algorithm NEXTN` (or DSPARK flags) + resident draft: `LK_GPU_RESIDENT_MOE_LAYERS_SPEC`; when combined with GPU Pool, check VRAM and measure first |
| **Super Mix Prefill** | `LK_GPU_PREFILL_MIN_TOKENS=1024~4096 LK_GPU_PREFILL_SUB_M=16384` + `--chunked-prefill-size 32000`; on cards with fast PCIe try `LK_GPU_PREFETCH_WINDOW=2` (ping-pong, copy overlaps gemm; default `1` = serialized); on 16 GB-class cards: `LK_GPU_PREFILL_MIN_TOKENS=0` + `--chunked-prefill-size 4096` |

What is actually active:

- hybrid: the lk_moe load line in the startup log;
- GPU Pool: `[pool] ... hit=` window line with `LK_POOL_LOG=100`.

Machine-level notes:

- **BIOS NUMA**: AMD EPYC NPS4 / Intel XEON SNC4; use 2,4,8 nodes (multiple of GPU count is best), up to 32.
- **Thread count**: HT on -> physical cores / GPUs; HT off -> (physical cores-2) / GPUs.
- **CPU power saving**: `LK_POWER_SAVING=1`.

---

## Benchmarks

### Performance

| Model | Version | CPU | Memory | GPU | Prefill | Decode | Spec. Decoding |
|-------|---------|-----|--------|-----|---------|--------|---------|
| DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | 780 t/s [in 32768] | 29 t/s [in 32768] | 35~50 t/s |
| DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0[ branch: 0.5.19-lkmoe-deepseekv4-sm80plus] | EPYC 7642 *2 | 16ch ddr4 3200 | 3090 * 2 | 1060 t/s [in 32768] | 31 t/s [in 32768] | 35~50 t/s |
| DeepSeek-V4-Flash-0731 | Lsglang-v1.4.7 | EPYC 9684x *2 | 24ch ddr5 4800 | pro 6000 * 1 | 4600 t/s [in 131072] | 75 t/s [in 131072] | 100~132 t/s |
| DeepSeek-V4.1-Flash | Lsglang-v1.5.3 | EPYC 9V74 *2 | ddr5 576g | 4080S * 2 | — | — | 60 t/s |
| DeepSeek-V4.1-Flash | Lsglang-v1.5.6 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | — | 21 t/s [in 65536, plain] | — |

All numbers collected with CPU Hybrid (default layer role) + legacy GPU prefill staging — before Super Mix Prefill and GPU Pool. Speculative decoding via NEXTN / DSPARK flags.

Experimental DeepSeek-V4.1 storage option: exact NVMe Engram lookup (`LK_ENGRAM_NVME=1`; host-table fallback `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`; budgets `LK_ENGRAM_NVME_CACHE_MB` (default 1024) / `LK_ENGRAM_NVME_STAGING_MB` (default 128)).

### Supported models & quant formats

Models:

Legend: ✅ tested · `ok` supported, not yet tested · blank = to be filled.

| Model | sm80 | sm86 | sm89 | sm90 | sm100 | sm120+ | Launch |
|-------|:----:|:----:|:----:|:----:|:-----:|:------:|--------|
| [Qwen3.8-Flash-Next](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-4-Over-6-FP8) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_qwen38_mtp_tp2_3090.sh) · [ple on nvme](launch/sglang_serve_qwen38_mtp_tp2_3090_nvme.sh) |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_dsv41_dspark_tp2_3090.sh) · [engram on nvme](launch/sglang_serve_dsv41_tp2_3090_nvme.sh) |
| [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_0731_dspark_tp2_3090.sh) |
| [GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_glm53_mtp_tp2_3090.sh) |
| [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_qwen36_mtp_tp2_3090.sh) |
| Qwen3-Coder-Next / 30B-A3B | | | | | | | |
| gemma-4-26B-A4B-it | | | | | | | |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | | | | | | | |
| Qwen3.5-35B-A3B | | | | | | | |
| Qwen3.5-122B-A10B | | | | | | | |
| Qwen3.5-397B-A17B | | | | | | | |
| Qwen3-VL-30B | | | | | | | |
| MiniMax-M2.7 / 2.5 / 2.1 | | | | | | | |
| GLM-5.2-NVFP4 | | | | | | | |
| GLM-5.1 / 5.0-FP8 | | | | | | | |
| GLM-4.7(-Flash) / 4.6V | | | | | | | |
| Kimi k2.6 / k2.5 | | | | | | | |

Quant formats:

| Format | Status |
|--------|--------|
| BF16 | tested |
| FP8 | tested |
| NVFP4 — compressed-tensors / modelopt | tested |
| MXFP4 | tested |
| AWQ INT4 (symmetric, per-group) | tested |

The MoE computation is delegated to lk_moe's quantized kernels; attention, shared experts and
all other layers run on the original sglang GPU path.

---

## Integration & development

lk_moe integrates into any inference framework as a drop-in MOE backend — the integration guide
(layer roles, feature gates, config fields, weight placement) lives in
**[docs/integration.md](./docs/integration.md)**.

---

## License

lk_moe is provided under a proprietary software license agreement. See [LICENSE](./LICENSE) for
terms and conditions.
