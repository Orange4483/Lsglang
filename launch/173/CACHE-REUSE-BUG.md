# 1.7.3 cache reuse: the first follow-up turn misses (DeepSeek-V4.1, hybrid SWA)

Status: reproduced on .14 / .15 with lsglang 1.7.3 (lk_moe 2.8.4), TP4/TP2,
`--enable-decoder-swa-bounded-replay`, HiCache L1/L2/L3 alike.

## Permanent variant (the reported 5/5 zeros)

Replaying the captured DSH request body (`request-body-gid___axonhub_Request_42584.json`,
11,246 prompt tokens, 35 tools, `reasoning_effort: xhigh`) as a growing conversation
reproduces the "never hits" case:

| turn | msgs | prompt | cached | wall |
|---|---|---|---|---|
| 0 | 7 | 11,246 | 0 | 30.2 s |
| 1 | 9 | 11,282 | 0 | 10.0 s |
| 2 | 11 | 11,294 | 0 | 10.0 s |
| 3 | 13 | 11,306 | 0 | 10.0 s |
| 4 | 15 | 11,318 | 0 | 10.1 s |

Once the first request is cold, that conversation never reuses again. Two identical
(non-growing) requests of the same body do hit (0 -> 11,008), and the logged
`input_ids` of consecutive turns share a 11,246-token prefix, so the prompt is
deterministic: the loss is in the cache, not in the rendering.

Likely mechanism: the cold request's completion insert leaves a **tombstoned** node
(Full KV present, SWA window missing, see below). Every later turn matches that node
(`full_hit` > 0, `device_prefix_len` = 0), re-prefills, and its own insert matches the
same node length, so no new leaf is created and
`SwaComponent.commit_insert_component_data` returns early when `is_new_leaf` is
false -- the SWA tombstone is never filled, so the node stays unusable forever until
it is evicted.

Fix direction: when a re-insert covers a node whose SWA data is missing, attach the
request's fresh SWA window to it (the same `_populate_range` path used for new leaves,
limited to the sliding window) instead of returning early for every non-new leaf.

## Symptom

A long agent conversation (60K+ token context, growing each turn):

| turn | prompt tokens | cached_tokens | wall |
|---|---|---|---|
| 0 | 61,786 | 0 | 28.7 s |
| **1** | 61,817 | **0** | **28.1 s** |
| 2 | 61,848 | 61,696 | 3.3 s |
| 3+ | … | 61,696 | ~3.5 s |

Turn 0 is cold (expected). Turn 1 should reuse turn 0's prefix but re-prefills
everything; from turn 2 on it works. Same for a second conversation started
after the first: it also pays the extra cold turn.

## Evidence (SGLANG_RADIX_DEBUG=1)

```
19:17:25  match key_len=61658 full_hit=0          turn 0 (cold)
19:17:53  insert key_len=61696 prefix_len=0       turn 0 completion: new node
19:17:54  match key_len=61689 full_hit=61440      turn 1: tree reports 61,440 tokens
19:18:22  insert key_len=61440 prefix_len=61440   turn 1 completion: node re-adopted
```

So turn 0's completion insert creates a node whose key covers 61,440 tokens, but
the node has no usable device KV: the scheduler's `req.prefix_len` comes from
`match_result.device_prefix_len`, which is 0 while `full_kv_hit_length` is
61,440. The prefill therefore recomputes the whole prompt.

The node's SWA span is a tombstone: `SwaComponent.prepare_for_caching_req`
records `set_evicted_seqlen(...)` for the SWA-evicted prefix, and
`commit_insert_component_data` marks `swa_branch_inserted = len(key) >=
branching_seqlen`. The Full KV is present, the SWA window is not, and the match
finalizer keeps `device_prefix_len` at the SWA boundary (0) even though
`--enable-decoder-swa-bounded-replay` is on. `swa_branching_seqlen` is computed
(`SwaComponent.finalize_match_result_in_tree_core`) but the scheduler only uses
it for the post-prefill SWA window insert, not for choosing the prefill prefix.

## Control experiment

Set `SGLANG_DISABLE_CHUNKED_RADIX_INSERT=0` (re-enables the per-chunk insert
that lsglang commit `972824c91d` disabled by default):

```
turn 0: cached=0       28.5 s
turn 1: cached=57344    4.1 s   <- lag gone
turn 2: cached=61696    3.2 s
```

So the regression is in the deferred (completion-only) insert path: the chunk
pages are adopted by the tree during prefill when the per-chunk insert runs, but
the completion insert leaves a node whose KV is not retained.

## Attempted fix (did not work)

Exposing `swa_branching_seqlen` as `device_prefix_len` in
`SwaComponent.finalize_match_result_in_tree_core` trips the scheduler invariant

```
AssertionError: ('<rid>', 0, 61440)
  assert len(path) >= req.prefix_len
```

because the matched path does not cover the branching point; the SWA component
holds only a tombstone. A correct fix must either

1. make the completion insert retain the chunk pages (keep per-chunk page
   adoption, defer only the tree publish), or
2. teach the prefill path to use the Full prefix and replay the SWA window
   (`swa_branching_seqlen`) — a scheduler-side change.

Repairing the tombstone at commit time
(`SwaComponent.commit_insert_component_data`, non-new-leaf branch: rebuild the
node's SWA window from its Full value via `_populate_range`) was also tried:

- `start = end - window` trips `assert node_start >= start` in `_segments`,
  because `_populate_range` wants `start <= node_start` and walks parents;
- `start = end - len(node.key)` with a `live_start < end` guard runs without
  crashing, but the five-turn cold scenario still reports `cached=0`, so the
  commit-time repair alone does not make the node reusable — it needs
  instrumentation to see whether the repair runs and what the following match
  reports.

The remaining lever is the per-chunk adoption of the request's pages
(`checkpoint_kv_cache` in `mem_cache/common.py`), which is exactly what
`SGLANG_DISABLE_CHUNKED_RADIX_INSERT=0` restores.

## Instrumented result (decisive)

A temporary log in the non-new-leaf branch of
`SwaComponent.commit_insert_component_data` shows, for every turn of the cold
five-turn run:

```
SWADBG commit non-leaf key_len=11008 prefix_len=11008 node_len=11008 swa_none=True live=11008
```

- `swa_none=True`: the matched node really has no SWA data (tombstone);
- `live=11008` equals the full key length: `params.get_evicted_seqlen(SWA)` is at
  the *end* of the prefix, i.e. bounded replay evicted every SWA window of the
  request, as designed. There is no live SWA span for the commit path to attach,
  which is why the tombstone-repair patch could not change the outcome.

So the only way to reuse such a node is Full KV + replay, and the blocker is in
the match result: `TreeCore._match_post_processor` computes
`device_prefix_len = sum(len(v) for v in value[:best_match_device_value_len])`,
i.e. only the nodes where *every* component has a device value. With the SWA
component absent the sum is 0, and `last_device_node` collapses to the root, so
the scheduler's `req.prefix_len` is 0 (and raising it in the SWA finalizer trips
`assert len(path) >= req.prefix_len`).

Correct fix: in `_match_post_processor`, when
`enable_decoder_swa_bounded_replay` is on and only the SWA component is missing,
report `device_prefix_len = full_kv_hit_length` (page aligned) and point
`last_device_node` at the deepest node that still holds the Full value, so the
prefill replays the SWA window from it. That is a tree-core change, not a
component one.

## Fix validated

`launch/173/patches/0001-dsv4-bounded-replay-full-prefix.patch` implements that
change behind `LK_SWA_BOUNDED_FULL_PREFIX=1` (uses the existing
`match_full_device_prefix()` helper). Measured on .14, 1.7.3 + L1, same cold
five-turn captured body:

| turn | cached (stock) | cached (patched) | wall (patched) |
|---|---|---|---|
| 0 | 0 | 0 | 29.3 s |
| 1 | 0 | **11,008** | 5.2 s |
| 2 | 0 | **11,008** | 5.1 s |
| 3 | 0 | **11,008** | 5.5 s |
| 4 | 0 | **11,008** | 5.4 s |

Correctness probe on the same build (fact at 10/50/90% of a 28.7K prompt plus an
eight-turn memory check): all four answers correct
(`8821-OMEGA`, `ZX-4471`, `03:40 KST`, `PROJ-5566`), with the last three served
at 3.9-3.9-2.2 s from the replayed prefix.

Caveats before proposing it upstream: the SWA window is rebuilt from the Full
KV on the prefill path, so a wider validation (long-context quality, DSPARK,
MHA models) is still needed; the env gate keeps stock behaviour unless opted in.

Full validation matrix (same cold five-turn captured body, `cached` of turn 1):

| host | config | stock | patched |
|---|---|---|---|
| .14 TP4 | L1 | 0 | 11,008 (5.2 s) |
| .14 TP4 | L2 (`--hicache-ratio 1`) | 0 | 11,008 (5.9 s) |
| .14 TP4 | L3 (file backend) | 0 | 11,008 (5.7 s) |
| .15 TP2+DSPARK | L2 | 0 | 11,008 (3.0 s) |

Correctness probes on both patched hosts (28.7K-token prompt, facts at 10/50/90%
plus an eight-turn memory check) returned every answer correctly:
`8821-OMEGA`, `ZX-4471`, `03:40`, `PROJ-5566`.

Apply / revert:

```
bash launch/173/patches/apply.sh                  # idempotent, keeps .orig
bash launch/173/patches/apply.sh "" --revert      # back to stock
```

`serve_173.sh` now exports `LK_SWA_BOUNDED_FULL_PREFIX=1` by default and defaults
`DSV41_HICACHE_RATIO` to 1 on the small host (the 1.7.2 recipe's ratio 4 is what
OOM-killed the scheduler).

## Pool-full matrix (safety check before submitting)

Script: `launch/lan_14_15/two_turn_probe.py` is the minimal repro; the matrix
below used a local `t_full_matrix.py` that first over-fills the pool with
14 x 52,655-token prompts (737K > 666K capacity, so the pool is full and old
prefixes are evicted), then starts brand-new conversations and a 97.5K-token
fact-recall probe.

| host / cache mode | fill | new 11K convo turn 1 | new 78K convo turn 1 | 97.5K facts | answers |
|---|---|---|---|---|---|
| .14 L1 | 14 x 52.6K | 11,008 | 77,824 | 97,280 | all correct |
| .14 L2 | 14 x 52.6K | 11,008 | 77,824 | 97,280 | all correct |
| .14 L3 (file) | 14 x 52.6K | 11,008 | 77,824 | 97,280 | all correct |
| .15 L2 + DSPARK | 14 x 52.6K | 11,008 | 77,824 | 97,280 | all correct |
| .14 L3, two 39K convos in parallel | full | 38,912 each | - | - | markers correct |

- "turn 1" is the first follow-up of a conversation whose turn 0 was cold; the
  whole point of the fix is that it now reuses instead of re-prefilling.
- The 97.5K probe asks for facts placed at 10/50/90% of the prompt; the last two
  questions are served from the cached prefix (97,280 tokens) and answer
  `ZX-4471` / `03:40 KST` correctly, so replay does not corrupt attention.
- Server logs on both hosts contain no traceback/assertion/ERROR after the
  `fired up` line for any of these runs; no OOM (host pool now ratio 1).
- gsm8k-50 on the patched .14 (L3, pool already full): **50/50**, matching the
  1.7.3 baseline measured before the patch (99/100 on 100 items).

Known rough edge, unrelated to this fix: the L3 file backend does not reliably
restore a prefix after a *process restart* (one restart restored an 11K prefix
from disk, a later one did not). L1/L2 behaviour is deterministic.

Conclusion: with the patch, a full pool no longer turns new conversations into
permanent zero-hit sessions on any of the three tiers.


## Related: host OOM restarts

`--hicache-ratio 4` (the 1.7.2 recipe) sizes the DSV4 host pool at 4x the device
pool (~60 GB here); on a 503 GB host this OOM-kills the scheduler
(`Out of memory: Killed process ... (sglang::schedul) anon-rss:69.5GB`). DSV4
rejects both `--hicache-size` ("use --hicache-ratio instead") and
`--hicache-host-memory-fraction` ("HiCache auto-sizing does not support
DeepSeekV4TokenToKVPool"), so there is no way to bound the host pool by RAM.
Use `--hicache-ratio 1` on 500 GB hosts until auto-sizing supports DSV4.

## Repro

```
python launch/lan_14_15/two_turn_probe.py http://127.0.0.1:8070 3
```
