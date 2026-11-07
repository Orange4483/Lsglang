#!/usr/bin/env bash
# Who waits during prefill?  Samples per-GPU utilisation and host CPU with
# timestamps, then separates the prefill window (0..ttft) from decode.
#   bash prefill_probe2.sh 15 32768 4242 64
set -uo pipefail
H=${1:?host}
TOK=${2:?tokens}
NONCE=${3:-4242}
GEN=${4:-64}
cd "$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$H" = 14 ] && M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
OUT=/tmp/pf2_$$.txt
: >"$OUT"

START=$(date +%s.%N)
python launch/lan_14_15/client.py gen --model "$M" --max-new "$GEN" --tokens "$TOK" \
  --nonce "$NONCE" --timeout 7200 >/tmp/pf2_client_$$.jsonl 2>/dev/null &
C=$!
while kill -0 $C 2>/dev/null; do
  T=$(date +%s.%N)
  g=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '|')
  echo "$T|$g" >>"$OUT"
  sleep 0.5
done
wait $C || true
RES=$(tail -1 /tmp/pf2_client_$$.jsonl)
echo "$RES"
TTFT=$(printf '%s' "$RES" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("ttft_s") or 0)' 2>/dev/null || echo 0)
T0=$(printf '%s' "$RES" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("t0") or 0)' 2>/dev/null || echo 0)
python3 - "$OUT" "$T0" "$TTFT" <<'PY'
import re, statistics, sys
path, start, ttft = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
if start <= 0:
    start = 0.0
pre, dec = {}, {}
for line in open(path):
    parts = line.strip().split("|")
    if len(parts) < 2:
        continue
    t = float(parts[0]) - start
    for tok in parts[1:]:
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"(\d+),\s*(\d+)", tok)
        if not m:
            continue
        util = int(m.group(2))
        (pre if t < ttft else dec).setdefault(int(m.group(1)), []).append(util)
for idx in sorted(pre):
    v = pre[idx]
    idle = sum(1 for x in v if x < 10)
    d = dec.get(idx, [])
    print(f"GPU{idx} prefill(n={len(v)}): mean={statistics.mean(v):.0f}% "
          f"median={statistics.median(v):.0f}% p90={sorted(v)[max(0,int(len(v)*0.9)-1)]}% "
          f"idle={idle}/{len(v)}  | decode mean={statistics.mean(d):.0f}%" if d else
          f"GPU{idx} prefill(n={len(v)}): mean={statistics.mean(v):.0f}% "
          f"median={statistics.median(v):.0f}% p90={sorted(v)[max(0,int(len(v)*0.9)-1)]}% "
          f"idle={idle}/{len(v)}")
PY
