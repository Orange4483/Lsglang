# Long-prompt OOM: super-chunk staging vs 665K context

Follow-up to `PERF-15-SWEEP.md`. The sweep only tested prompts up to 80K tokens; a
single long prompt changes the picture.

## What happens

| host | staging | prompt | result |
|---|---|---|---|
| .14 | chunk 8192, SUB_M 20480 | 390K tokens | **CUDA OOM, engine dies** (`512 MiB` request, 223 MiB free) |
| .14 | chunk 8192, SUB_M 0 | 585K tokens | OK, 2.2K t/s |
| .15 | chunk 1024, SUB_M 32768 | ~600K tokens | **CUDA OOM, engine dies** (`122 MiB`, 36 MiB free) |
| .15 | chunk 1024, SUB_M 8192 | 585K tokens | OK, ~2.0K t/s |

With `max_total_num_tokens = 666,112` and `--mem-fraction-static 0.86-0.88` the KV
pool already fills the cards, so the super-chunk staging scratch has nowhere to go
once a single prefill is long. Per-chunk staging (SUB_M <= chunk) keeps the
scratch small enough, at the cost of prefill throughput:

| host | staging | prefill 32K | prefill 80K | long prompt |
|---|---|---|---|---|
| .14 | SUB_M 0 | 2,318 | 2,225 | 585K OK |
| .14 | SUB_M 20480 | 3,022 | 3,149 | 400K OOM |
| .15 | SUB_M 8192 | 2,087 | 2,034 | 585K OK |
| .15 | SUB_M 32768 | 3,443 | 3,193 | 600K OOM |

## Defaults

Because the product requirement is a *usable* 665,535-token context, both hosts
now default to the long-context-safe staging:

- .14: `chunk 8192`, `SUB_M 0`
- .15: `chunk 1024`, `SUB_M 8192` (the wrapper still arms, `CAP=8192`)

Override with `DSV41_SUB_M=<n>` when only short prompts are expected and the
faster prefill matters. Any value above the chunk size reintroduces the OOM risk
for single prompts in the hundreds of thousands of tokens.

## Not yet explained

Why a DSH session on .15 reported `cached_tokens = 0` for every turn
(11,092 -> 37,769 tokens, 2026-11-09 23:23-23:31 UTC in the .15 log). A synthetic
tool-call conversation reuses fine, so the prefix must be changing between turns;
`.15` now runs with `DSV41_LOG_REQ=1` to capture `input_ids` for a live session and
diff them.
