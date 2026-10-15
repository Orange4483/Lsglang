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
