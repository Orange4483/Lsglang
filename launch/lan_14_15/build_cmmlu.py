import csv, glob, json, os, random

root = r"E:\project\lsglang\_work\cmmlu"
tests = sorted(glob.glob(os.path.join(root, "test", "*.csv")))
rng = random.Random(20261108)
items = []
for path in tests:
    subject = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        continue
    rng.shuffle(rows)
    r = rows[0]
    items.append({"subject": subject,
                  "question": r["Question"].strip(),
                  "A": r["A"].strip(), "B": r["B"].strip(),
                  "C": r["C"].strip(), "D": r["D"].strip(),
                  "gold": r["Answer"].strip().upper()})
rng.shuffle(items)
items = items[:60]
out = r"E:\project\lsglang\_work\cmmlu60.jsonl"
with open(out, "w", encoding="utf-8") as fh:
    for i, it in enumerate(items):
        it["i"] = i
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")
print("wrote", out, len(items), "subjects:", len({i['subject'] for i in items}))