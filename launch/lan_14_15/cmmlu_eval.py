#!/usr/bin/env python3
"""CMMLU-style multiple-choice eval against one or more OpenAI-compatible targets."""
import argparse
import json
import re
import time
import urllib.request

SYS = ("你是一位严谨的专家。回答单项选择题时，先简要推理，最后单独一行输出：答案：X"
       "（X 为 A、B、C、D 之一）。")


def build(it):
    return (f"题目：{it['question']}\n"
            f"A. {it['A']}\nB. {it['B']}\nC. {it['C']}\nD. {it['D']}\n"
            "请给出你的答案。")


def extract(text):
    m = re.findall(r"答案[:：]\s*([ABCD])", text)
    if m:
        return m[-1]
    m = re.findall(r"(?:^|\n)\s*([ABCD])\s*(?:\n|$)", text)
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1] if m else None


def run(target, items, max_tokens):
    name, base, key, model = target
    ok = 0
    rows = []
    for it in items:
        payload = {"model": model,
                   "messages": [{"role": "system", "content": SYS},
                                {"role": "user", "content": build(it)}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                out = json.loads(resp.read())
            text = out["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            text = ""
            print(json.dumps({"target": name, "i": it["i"], "error": str(exc)[:120]}), flush=True)
        pred = extract(text)
        good = pred == it["gold"]
        ok += good
        rows.append({"i": it["i"], "subject": it["subject"], "gold": it["gold"],
                     "pred": pred, "ok": good, "s": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    print(f"== {name}: {ok}/{len(items)}", flush=True)
    return rows


def parse_target(spec):
    parts = spec.split("|")
    name, base = parts[0], parts[1]
    key = parts[2] if len(parts) > 2 and parts[2] else ""
    model = parts[3] if len(parts) > 3 else "DeepSeek-V4.1-Flash-1017"
    return name, base, key, model


ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--target", action="append", required=True)
ap.add_argument("--out", default="")
args = ap.parse_args()

items = [json.loads(line) for line in open(args.items, encoding="utf-8")][: args.n]
all_rows = {}
for spec in args.target:
    target = parse_target(spec)
    all_rows[target[0]] = run(target, items, args.max_tokens)
if args.out:
    json.dump(all_rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
