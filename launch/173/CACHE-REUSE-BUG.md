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
