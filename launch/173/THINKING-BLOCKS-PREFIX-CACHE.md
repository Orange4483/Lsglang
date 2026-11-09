# Why the prefix stops growing at the assistant turn (thinking mode)

Observed on .15 (1.7.4 + a3aa6c68ec): a production DSH session reused
38,912 -> 44,032 -> 45,824 tokens and then stayed at 45,824 while the prompt grew
46K -> 77K. 45,824 is exactly `page_align(previous prompt)`: the match stops at the
boundary where the assistant turn begins, so every turn re-prefills its own
assistant and tool output.

## What the encoder does with `reasoning_content`

`encoding_dsv4.encode_messages(..., drop_thinking=True)`:

```python
effective_drop_thinking = drop_thinking
if any(m.get("tools") for m in full_messages):
    effective_drop_thinking = False          # tools keep thinking
if thinking_mode == "thinking" and effective_drop_thinking:
    full_messages = _drop_thinking_messages(full_messages)
```

Rendering the same assistant message (`reasoning_content` + `content`) with
sglang's own encoder on this build:

```
tools=True : <｜Assistant｜><think>The belt drifts when ...</think>Most likely cause: ...
tools=None : <｜Assistant｜></think>Most likely cause: ...
```

The no-tools branch emits a **dangling `</think>` with no opening `<think>` and no
reasoning** - it is neither what the model generated nor a clean drop.

## Measured effect (reasoning_effort=xhigh)

| shape | turn-1 prompt | next-turn cached | note |
|---|---|---|---|
| no tools, reasoning echoed | 4,379 | 4,096-4,352 | stops at the prompt boundary |
| tools, reasoning echoed | 4,626 | 4,864 (+238) | only 238 tokens into a 2,282-char reasoning block |
| tools, content only (reasoning dropped by the client) | 4,626 | 4,096 (-530) | worse |
| **no thinking** (`reasoning_effort=none`), 4 turns | 4,308 | 4,096 -> 4,096 -> 4,096 -> 4,352 | grows with the prompt |
| no thinking, 2,370-token reply | 4,322 | 6,400 (+2,078) | generated content fully reusable |

So the generated assistant turn cannot be reproduced from the text the client
echoes back: the detokenized `reasoning_content` re-tokenizes to a slightly
different sequence than the tokens the model emitted, and the match stops right
there. With thinking disabled the assistant turn is plain `content`, which
round-trips, and reuse grows normally.

## Workarounds

1. **Cache-first**: serve that route with `reasoning_effort: none` (DSH:
   `reasoningEffort: off` for the pro5000 model). Verified above.
2. Keep `xhigh` and accept that each turn re-prefills from its own assistant
   message; hit rate then falls as the conversation grows.
3. Upstream (needs an issue): make the round-trip lossless - either render the
   assistant turn from the exact generated token ids (clients can send
   `token_ids`), or normalise the encoder so that
   `render(decode(generated)) == generated` for the thinking block, and fix the
   dangling `</think>` in the `drop_thinking` branch.
