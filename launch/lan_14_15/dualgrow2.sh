#!/usr/bin/env bash
# Two concurrent streams at growing lengths, in two flavours:
#   same : identical prompts  -> the prefix (and its KV pages) are shared
#   diff : different nonces   -> no prefix sharing, both contexts stay resident
# Reports per-stage VRAM peak (sampled while the pair runs) and any failure.
#   bash dualgrow2.sh 14 same|diff "65536 131072 262144"
set -uo pipefail
HOST_NUM=${1:?usage: dualgrow2.sh 14|15 same|diff "lengths"}
MODE=${2:?same|diff}
cd "$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
if [ "$MODE" = same ]; then NA=0; NB=0; else NA=11; NB=22; fi
PEAK=/tmp/gpu_peak_$$.txt
: >"$PEAK"

for n in ${3:-65536 131072 262144}; do
  echo "== $MODE $n" >&2
  : >"$PEAK"
  python launch/lan_14_15/client.py gen --model "$M" --max-new 400 \
    --tokens "$n" --nonce "$NA" --timeout 7200 2>/dev/null >/tmp/dg_a.jsonl &
  A=$!
  python launch/lan_14_15/client.py gen --model "$M" --max-new 400 \
    --tokens $((n + 3)) --nonce "$NB" --timeout 7200 2>/dev/null >/tmp/dg_b.jsonl &
  B=$!
  while kill -0 $A 2>/dev/null || kill -0 $B 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader >>"$PEAK" 2>/dev/null
    sleep 5
  done
  wait $A || echo "stream A failed" >&2
  wait $B || echo "stream B failed" >&2
  cat /tmp/dg_a.jsonl /tmp/dg_b.jsonl
  python3 - "$PEAK" <<'PY'
import re, sys
vals = []
for line in open(sys.argv[1]):
    nums = [int(x) for x in re.findall(r"(\d+) MiB", line)]
    if nums:
        vals.append(nums)
if vals:
    peak = [max(col) for col in zip(*vals)]
    print("gpu_peak_mib:", peak)
else:
    print("gpu_peak_mib: n/a")
PY
done
