# Upstream follow-up a3aa6c68ec (after 1.7.4): the tombstone path was dead

`a3aa6c68ec fix(mem_cache): keep Full-prefix reuse under bounded SWA replay complete`
(2026-11-09) fixes two things in the code that shipped in 1.7.4:

1. It removes the guard

   ```python
   and any(c.reuse_alignment() > self.page_size
           for c in self.components if hasattr(c, "reuse_alignment"))
   ```

   With the author's own comment: *"A previous `any(c.reuse_alignment() >
   page_size)` guard could never be true here -- only Mamba overrides
   reuse_alignment() and no Mamba model enables this feature -- so it silently
   disabled the whole path."* In 1.7.4 that meant the Full-prefix reuse added by
   PR #33 was never reached on DeepSeek-V4.1.
2. `SGLANG_SWA_REPLAY_REBUILD_WINDOW` now defaults to **true**: one compression
   window (>= 128 tokens, page aligned) is kept out of the reuse so prefill
   rebuilds the c4 ring / online c128 state at the boundary. Setting it to 0 is
   the "broken control arm" (reuse without the rebuild feeds the low-ratio
   layers an empty compressor).

It also adds a unit test (`test_unified_radix_cache_unittest.py`).

## Installed here

Both hosts run 1.7.4 with these two files replaced from the commit
(`python/sglang/srt/environ.py`,
`python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`, backups kept as
`*.bak.a3aa`):

```
bash /tmp/apply_a3aa.sh          # or scp ~/patch174/*.py and copy them in
```

## Validation on .15 (1.7.4 + a3aa6c68ec, RADIX debug on)

| test | result |
|---|---|
| 4-turn conversation | cached 4,096 -> 4,096 -> 4,352 (grows with the prompt) |
| 2,370-token generated reply, next turn | cached 6,400 (+2,078 over the previous prompt) |
| real tool call round-trip | cached 4,352, not broken by `tool_calls` |
| 121K-token needle, two questions | both correct, second served from 121,088 cached |

Caveat: the very first request after installing the patch stalled for more than
ten minutes (GPU 0/1 at 100 %, no `request.finished`), and the same request after
a restart completed in a few seconds. It looks like a one-time compile/warm-up of
the re-enabled path, but it is worth reporting upstream because a first request
that hangs after an upgrade is indistinguishable from a dead server.

## Still open

The DSH session that showed a frozen hit length (45,824 tokens, page-aligned
`previous prompt - 128`) cannot be replayed from the captured files (they hold
headers and response bodies only, and the .15 log has no entries for that
window). `.15` runs with `DSV41_LOG_REQ=1`; reproducing 2-3 turns will give the
`input_ids` to diff.
