# Prefill / decode sweep on .14 and .15 (1.7.3 + SWA Full-prefix fix)

Goal: keep the 665,535-token context (`max_total_num_tokens = 666,112`) while
making prefill usable (1000-2000 t/s) and decode as fast as possible.

Measured with `bench15.py`: a fresh (unique-prefix) ~32K and ~80K prompt with
`max_tokens=1` for prefill, and a 320-token generation, median of three runs, for
decode. `max_total_num_tokens` was 666,112 in every run.

## .15 — RTX 4090 48G + RTX PRO 5000, TP2 + DSPARK, 128 cores, 1 TB RAM

| # | chunk | `LK_GPU_PREFILL_SUB_M` | note | prefill 32K | prefill 80K | decode | KV |
|---|---|---|---|---|---|---|---|
| A | 8192 | 0 | previous default (per-chunk staging) | 2,129 | - | 74.6 | 666,112 |
| B | 1024 | 12288 | author's tp2 recipe values | 2,390 | - | 79.3 | 666,112 |
| C | 1024 | 32768 | super-chunk `CAP=32768, inner=1024` | 3,454 | 3,201 | 76.7 | 666,112 |
| D | 1024 | 65536 | `CAP=65536` | 3,441 | **CUDA OOM** | 86.0 | 666,112 |
| E | 1024 | 32768 + DSPARK 7/8 | `--speculative-dspark-block-size 7 --speculative-num-draft-tokens 8` | 3,457 | 3,203 | 79.0 | 666,112 |
| F | 1024 | 32768 + `LK_POWER_SAVING=0` | | 3,424 | 3,208 | 84.2 | 666,112 |
| H | 1024 | 32768, power saving off, tuned defaults | two repeat runs | 3,448 / 3,436 | 3,198 / 3,194 | 96.3 / 82.9 | 666,112 |

Notes:

- **Prefill** is where the win is: `SUB_M > chunk` arms the super-chunk wrapper
  (the boot line `Super-chunk prefill wrapper active: CAP=…, inner=…` confirms
  it), which cuts the number of staged MoE passes. 32K goes 2,129 -> 3,44x
  (+62%) and 80K lands at ~3,19x.
- **`SUB_M=65536` is not safe here**: the 32K prompt passed, the 80K one ran the
  rank out of VRAM (`torch.OutOfMemoryError: CUDA out of memory`, 70 MiB free of
  47.37 GiB) and the engine died. 32768 is the ceiling on this box.
- **Decode stays in the 83-96 t/s band** regardless of `chunk`/`SUB_M` (they only
  touch prefill). The spread between identical runs is ~13 t/s, so the
  difference against the old default (74.6, single run) is not conclusive; the
  `LK_POWER_SAVING=0` and DSPARK 7/8 variants were both inside the noise band.
  Decode is bound by the CPU-resident expert fetch; only ~2.1 GB (4090) and
  3.7 GB (PRO 5000) of VRAM are free while serving 666,112 tokens, so
  `LK_GPU_RESIDENT_MOE_LAYERS` cannot be raised without shrinking the KV.
- Cache reuse and long context were re-verified on the tuned config: a cold
  five-turn conversation hits 11,008 from turn 1, and a 121K-token needle test
  answers both facts correctly with 121,088 cached on the repeat.

Recommended command (now the default on `.15`, `bash start.sh` reproduces it):

```bash
DSV41_CHUNK=1024 DSV41_SUB_M=32768 LK_POWER_SAVING=0 \
DSV41_HICACHE_MODE=l2 DSV41_HICACHE_RATIO=2 \
  bash ~/project/Lsglang-egram-nvme-kv-nvme/lsglang173/start.sh
```

## .14 — 4x RTX 4090 24G, TP4, no DSPARK, 64 cores, 503 GB RAM

| # | chunk | `LK_GPU_PREFILL_SUB_M` | note | prefill 32K | prefill 80K | decode | KV |
|---|---|---|---|---|---|---|---|
| A | 8192 | 0 | previous default | 2,318 | 2,225 | 49.6 | 666,112 |
| B | 1024 | 32768 | `CAP=32768` | **CUDA OOM** | - | - | - |
| C | 1024 | 16384 | `CAP=16384` | 2,932 | 2,865 | 51.6 | 666,112 |
| D | 1024 | 24576 | `CAP=24576` | **CUDA OOM** | - | - | - |
| E | 1024 | 20480 | `CAP=20480, inner=1024` | 2,907 | 3,068 | 49.7 | 666,112 |
| F | 8192 | 20480 | `CAP=20480, inner=8192` | **3,022** | **3,149** | 49.8 | 666,112 |

- 24 GB cards cap the super-chunk at ~20K tokens: 24576 already OOMs a single
  32K prompt (`19 MiB free of 23.52 GiB`), so 20480 is the practical ceiling.
- Keeping `chunk 8192` (instead of 1024) is slightly faster here: fewer
  scheduling rounds for the same CAP.
- Two parallel 58.5K prefills on the tuned config: both complete in 42 s
  (~1,390 t/s each) without OOM, so `--max-running-requests 2` stays safe.
- Cache reuse and long context re-verified: cold five-turn conversation hits
  11,008 from turn 1, 121K needle test answers both facts (121,088 cached).

Recommended (now the default on `.14`):

```bash
DSV41_CHUNK=8192 DSV41_SUB_M=20480 DSV41_HICACHE_MODE=l2 DSV41_HICACHE_RATIO=1 \
  bash ~/project/Lsglang-egram-nvme-kv-nvme/lsglang173/start.sh
```

Conservative variant if the workload has bursts of concurrent long prefills:
`DSV41_SUB_M=16384` (2,932 / 2,865 t/s, more VRAM headroom).
