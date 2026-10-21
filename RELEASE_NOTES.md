# lsglang v1.6.1 — Upstream Main Sync, CUTLASS GPU-Prefill, SM8x Determinism

**Base:** upstream sglang main @ `06342acad2` (2026-10-21, 1,387 commits since
the v1.6.0 base `583fada954`); **lk_moe v2.7.1**.

Everything on top of upstream is the lk_moe integration and its upgrades.

## Highlights

- **Upstream main sync** — unified memory (asymmetric KV rows, fused draft /
  sliding-window / recurrent-state layers), diffusion pipeline work, Rust
  gRPC/router extensions, scheduler and MemCache refactors, plus the
  DeepSeek-V4.1 late-layer pinned-row perf path.
- **CUTLASS GPU-prefill engine is now the default**
  (`LK_GPU_PREFILL_ENGINE=cutlass`): restored C++ grouped-GEMM path with
  unbounded pool-row borrowing (the `LK_GPU_PREFILL_BORROW_MB` budget is gone),
  one freeze window per prefill forward, whole-slab block planning, and wave
  width control (`LK_GPU_PREFILL_WAVE_EXPERTS=-1` auto / `0` single-shot /
  `>0` cap). Triton staging remains selectable.
- **GPU expert pool** — deterministic split (`LK_POOL_DET_SPLIT`: frozen
  snapshot + copy-fence wait before the GPU half reads slots), two-phase lease
  with a publish-side ledger, NVFP4 scale-stride fix (real-weight rel 0.86 ->
  1.2e-6), and expert-pool park around CUDA-graph capture/replay. `LK_POOL_SLOTS=0`
  now enables prefill-only staging on its own instead of disabling GPU prefill.
- **SM8x performance** — dense block-FP8 group-32 Marlin W8A16 decode
  (dual-retain repack, M<=16, kernels prewarmed before capture), deterministic
  FP8 (`SGLANG_DETERMINISTIC_FP8`: fixed split-K, SM12x facade bypass), and the
  inverse RoPE fused into the DeepSeek-V4 bf16 `wo_a` einsum on SM8 decode.
- **NVMe weight offload — real NVMe, no host-RAM residency** — DeepSeek-V4.1
  Engram and Qwen3.8 PLE tables are read from NVMe into the VRAM hot cache
  through a pinned ring; the full tables never enter host memory (boot cost is
  independent of checkpoint size). Includes the DeepSeek-V4.1-Flash-1017 model
  entry and NVMe launch scripts.
- **Determinism / quality** — run-deterministic persistent mix reduction, QSA
  top-k ordering, chunked-radix insert deferred to request completion
  (upstream #38319 race), deferred clears for recycled mamba ping-pong slots,
  and the custom-allreduce pull path for captured graphs on weak-P2P hosts.

## Fixes

- Pinned-host race eliminated tree-wide: async H2D sources park in a 16-deep
  ring (`park_pinned`/`h2d_async`); 11 sites — kills the greedy non-determinism /
  runaway-loop family (Qwen3.8, DeepSeek V4.x; cf. upstream #37343/#35632/#38319/#43956).
- Heterogeneous TP4 capture deadlock: graph eligibility via group-consensus
  arch check (`_tp_all`), capture gate `>=10` → `>=8`.
- First-request JIT stall on DeepSeek V4.x: flashinfer sampling kernel prewarmed
  at model load (V4 + dspark); boot-time prewarm also covers SWA page release,
  QSA graph metadata, small-copy variants, and bs=1 decode.

## Known Issues

- DeepSeek-V4.1 det512 dynamic split is not yet reproducible; prefill down-GEMM
  keeps a bf16 hi/lo ~3e-3 tail (plan recorded).
- lvllm decode GPU half is not yet ported.
- `reasoning_content` is dropped on `tool_calls` responses with speculative decoding.
- Three speculative kernels still JIT on the first request (`--warmups
  prefill_shapes` can absorb them).


# lsglang v1.6.0 — GPU Expert Pool, NVMe Weight Offload, Super-Mix Prefill

**Base:** upstream sglang @ `583fada954` (main, 2026-09-27)
**lk_moe:** v2.6.0
Everything on top of the base is the lk_moe integration and its upgrades.

## Model Support

All models run on SM80+. Every model supports the GPU Expert Pool and Super-Mix
Prefill; models with big embedding/Engram tables additionally support NVMe
Weight Offload.

| Model | GPU Expert Pool | Super-Mix Prefill | NVMe Weight Offload |
|---|---|---|---|
| Qwen3.8-Flash-Next (NVFP4) | ✓ | ✓ | ✓ (PLE table) |
| DeepSeek V4.1 (FP8) | ✓ | ✓ | ✓ (Engram table) |
| GLM-5.3 Flash (NVFP4) | ✓ | ✓ | — |
| DeepSeek V4 Flash (0731) | ✓ | ✓ | — |

## New Features

**GPU Expert Cache Pool — CPU and GPU compute in parallel**
Hot experts stay resident on GPU while the rest stream from CPU, and a decode
step runs **both halves concurrently**: pool-resident experts through the GPU
resident-gemv path while the remaining experts compute on CPU — two compute
engines serving one MoE layer at the same time. Staged GPU/CPU prefill and a
rewritten CPU expert-decode path; mailbox decode transport (takeover fix: no
more silent hangs on slow hosts at TP4).
*Enable:* `LK_POOL_SLOTS=64` (value is the switch: resident slots per MoE
layer, `0` = off), GPU decode half `LK_POOL_DECODE=1` (default follows pool).

**NVMe Weight Offload (new subsystem)**
- NVMe row filler inside lk_moe: sorted ring -> batched pread -> pinned staging
  -> H2D, serving both Engram (DSV4.1) and PLE embedding tables (Qwen3.8).
- Boot cost independent of checkpoint size; multi-hundred-GiB tables never
  enter RAM — VRAM hot cache + pinned ring only.
*Enable:* `LK_ENGRAM_NVME=1` (DSV4.1 Engram) or
`--ple-offload-embedding --ple-offload-backend nvme` (Qwen3.8 PLE).
Optional: `LK_NVME_DIRECT=1` — O_DIRECT reads bypassing page cache, backed by
a pinned direct-mapped host cache (`LK_NVME_CACHE_MB`, `LK_NVME_STAGING_MB`).

**Super-Mix Prefill**
Super-chunk scheduler + attention wrapper + per-model plan loops (DSV4,
Qwen3.8, GLM-5.3): long prefills pass through one expert-union staging with a
sub-M loop, bounding block-scale temporaries while keeping the GPU-fed pipeline
saturated.
*Enable:* `LK_GPU_PREFILL_MIN_TOKENS=1024` + `LK_GPU_PREFILL_SUB_M=16384`
(value itself switches; `0` = per-chunk staging) together with
`--chunked-prefill-size`; on fast-PCIe cards add `LK_GPU_PREFETCH_WINDOW=2`
(copy overlaps GEMM, ping-pong).

## Fixes & Notes

- Environment naming unified under `LK_*`; legacy `LVLLM*` names removed.
- Tuning parameters have been optimized and unified — see the
  [README tuning guide](https://github.com/guqiong96/Lsglang/blob/lsglang-v1.6.0/README.md)
  for the full `LK_*` reference and recommended presets per scenario.
