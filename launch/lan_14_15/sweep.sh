#!/usr/bin/env bash
# One configuration's measurement pass: prefill/decode at three lengths plus a
# same-domain six-turn decode run (single stream) and the pool stats.
#   bash sweep.sh 14 tag
set -uo pipefail
HOST_NUM=${1:?usage: sweep.sh 14|15 tag}; shift
TAG=${1:?tag}
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
OUT=$REPO/logs/${TAG}.jsonl
cd "$REPO"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
model=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && model=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
LOG=logs/serve_${HOST_NUM}.log
C="python launch/lan_14_15/client.py gen --model $model --max-new 64 --timeout 7200"

: >"$OUT"
for n in 32768 131072 524288; do
  echo "[$(date '+%T')] len=$n" >&2
  $C --tokens "$n" --nonce "$((900 + n % 7))" 2>/dev/null >>"$OUT" || echo "FAIL len=$n" >&2
  tail -1 "$OUT" >&2
done
echo "[$(date '+%T')] hot x6" >&2
python launch/lan_14_15/hotcold.py "$model" hot 6 2>/dev/null | tail -1 | tee -a "$OUT"
grep -a '\[pool\] dstep' "$LOG" | tail -2 | cut -c1-150 | tee -a "$OUT"
grep -a '\[pool\] slab' "$LOG" | tail -1 | tee -a "$OUT"
echo "results: $OUT" >&2
