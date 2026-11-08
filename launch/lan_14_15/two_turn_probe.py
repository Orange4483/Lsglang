#!/usr/bin/env python3
"""Minimal repro: the first follow-up turn of a long conversation gets no prefix reuse.

Run against a 1.7.3 server (DeepSeek-V4.1, hybrid SWA + HiCache) with the default
mid-chunk radix insert disabled (lsglang commit 972824c91d):

    python two_turn_probe.py http://127.0.0.1:8070
    turn 0: cached=0     (cold, expected)
    turn 1: cached=0     (BUG: the turn-0 prefix is in the tree but unusable)
    turn 2: cached=N     (the turn-1 completion insert finally adopts the pages)

With SGLANG_DISABLE_CHUNKED_RADIX_INSERT=0 the turn-1 line shows a hit instead,
which pins the regression to the deferred (completion-only) insert path.
"""
import json
import sys
import time
import urllib.request

FILLER = ("The shift supervisor reviewed the conveyor logs and confirmed that the tension readings "
          "stayed within tolerance during the night shift. ")


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8070"
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    context = "Conversation about the warehouse.\n" + (FILLER * 3500)[:400000]

    messages = [{"role": "user", "content": context + "\n\nQuestion 0: summarise the shift."}]
    for turn in range(turns):
        payload = {"model": "DeepSeek-V4.1-Flash-1017", "messages": messages,
                   "max_tokens": 32, "temperature": 0.0, "stream": False}
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as resp:
            out = json.loads(resp.read())
        usage = out.get("usage", {})
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        answer = out["choices"][0]["message"].get("content") or ""
        print(json.dumps({"turn": turn, "prompt": usage.get("prompt_tokens"),
                          "cached": cached, "s": round(time.time() - t0, 1)}), flush=True)
        messages = messages + [{"role": "assistant", "content": answer[:300] or "ok"},
                               {"role": "user", "content": f"Question {turn + 1}: continue."}]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
