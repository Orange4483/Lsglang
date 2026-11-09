# lsglang 1.7.4 regression check (.14 / .15)

Environment: `~/envs/Lsglang174` unpacked from `E:\Lsglang174.tar.gz`
(lsglang 1.7.4, lk_moe 2.8.4, Python 3.12.11). The 1.7.3 site-packages patch is
gone; 1.7.4 carries the upstream fix.

## What changed upstream

- PR #33 (our report + first fix) is merged; the release notes credit it.
- `b48cc7d612 fix(mem_cache): keep the swa window behind the replay boundary`
  (back-port of sgl-project/sglang#46008) keeps the SWA window behind the replay
  boundary, so a real prefix match normally succeeds instead of degrading to a
  Full-only tombstone.
- `12d93188bb fix(mem_cache): gate tombstone reuse on restorable state` refines
  our patch: the Full-device prefix is only exposed when some component reports
  `reuse_alignment() > page_size`, i.e. a stateful component that can actually
  restore (Mamba's checkpoint grid). The comment states that DSV4.1's c4/c128
  rows and indexer keys live in per-request slots that nothing restores, so
  reusing a Full-only tombstone there fed those layers nothing (they measured
  `cached=6400` -> repetition). DSV4.1 therefore keeps the old no-reuse
  behaviour on that branch and relies on the SWA retention instead.
- `SGLANG_SWA_REPLAY_REBUILD_WINDOW` added (debug, off by default).
- Recipes: `sglang_serve_dsv41_tp4.sh` and `sglang_serve_dsv41_tp2_3090_dspark.sh`
  are byte-identical in their parameter lines to 1.7.3 (`SUB_M` 8192/12288,
  chunk 1024, max-total 262144, mem-frac 0.80/0.78, DSPARK 5);
  a new `sglang_serve_dsv41_tp2_5060ti.sh` targets different hardware.
  -> no recipe change for us, so the tuned parameters below stay.

## Cache regression matrix (1.7.4, no patch)

Cold five-turn captured conversation (11,246 initial tokens) and the pool-full
matrix (14 x 52,655-token fills = 737K > 666K capacity, then fresh work):

| host / tier | cold 5-turn convo | 78K conversation | 97.5K fact probe |
|---|---|---|---|
| .14 L1 | turn 1+ = 11,008 | - | - |
| .14 L2 | turn 1+ = 11,008 | 77,824 | 97,280, answers correct |
| .14 L3 | turn 1+ = 11,008 | 77,824 | 97,280, answers correct |
| .15 L2 + DSPARK | turn 1+ = 11,008 | 77,824 | 97,280, answers correct |

No zero-hit session, no wrong answer; `max_total_num_tokens` stays 666,112.
The L3 run even restored the 11K prefix on the first turn after a restart
(`cached=11,008` at turn 0), which the 1.7.3 build did only sometimes.

## Performance (tuned parameters unchanged)

| host | config | prefill 32K | prefill 80K | decode | KV |
|---|---|---|---|---|---|
| .14 | chunk 8192, SUB_M 20480, L2 ratio 1 | 2,976 | 3,040 | 50.9 | 666,112 |
| .15 | chunk 1024, SUB_M 32768, L2 ratio 2, power saving off | 3,443 | 3,193 | 75.2 | 666,112 |

Both are within noise of the 1.7.3 numbers (3,022 / 3,149 / 49.8 and
3,448 / 3,198 / 82.9-96.3), so the release does not change prefill/decode here.

## Boot-time regression on .15 (needs attention)

The .15 boot spent 7.6 minutes in `Capture draft verify CUDA graph begin`
(`Load weight end` 21:02:33 -> capture 21:04:54 -> `fired up` 21:12:31; the
whole start took about 22 minutes against ~6 for 1.7.3). GPU 0 sat at 100 %
utilisation for the whole capture. It does finish and serves normally
afterwards, but a restart on this host is now a ~20 minute operation. Worth
checking whether the DSPARK draft-verify graph recompiles per boot on the
SM89 + SM120 pair.

## Launcher change

`launch/173/serve_173.sh` now defaults to the newest packaged env on the host
(`Lsglang175` -> `Lsglang174` -> `Lsglang173`), so `bash start.sh` picks 1.7.4
without `ENV173=...`.
