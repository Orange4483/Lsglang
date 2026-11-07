#!/usr/bin/env bash
# Admission test: with a large --max-new, are two requests really admitted
# together (#running-req: 2) or does the scheduler keep one queued?
#   bash admit.sh 14 <ctx> <max_new> <seconds>
set -uo pipefail
HOST_NUM=${1:?usage: admit.sh 14|15 ctx max_new seconds}
CTX=${2:?ctx}
GEN=${3:?max_new}
SECS=${4:-90}
cd "$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
L=logs/serve_${HOST_NUM}.log
S=$(grep -a -n "serve_${HOST_NUM} pid=" "$L" | tail -1 | cut -d: -f1)
S=${S:-0}

echo "== launching 2 x (ctx=$CTX, max_new=$GEN)" >&2
python launch/lan_14_15/client.py gen --model "$M" --max-new "$GEN" \
  --tokens "$CTX" --nonce 11 --timeout 14400 >/tmp/adm_a.jsonl 2>/dev/null &
A=$!
python launch/lan_14_15/client.py gen --model "$M" --max-new "$GEN" \
  --tokens $((CTX + 3)) --nonce 22 --timeout 14400 >/tmp/adm_b.jsonl 2>/dev/null &
B=$!
sleep "$SECS"
echo "== scheduler state after ${SECS}s"
tail -n +"$S" "$L" | grep -a 'Decode batch' | tail -3 | cut -c1-170
tail -n +"$S" "$L" | grep -a 'Prefill batch' | tail -2 | cut -c1-150
kill $A $B 2>/dev/null
wait $A $B 2>/dev/null
echo "== retraction / queue hints"
tail -n +"$S" "$L" | grep -aiE 'retract|queue-req: [1-9]|out of memory' | tail -4 | cut -c1-160
