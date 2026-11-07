#!/usr/bin/env bash
# Multi-length stability matrix against a local DSV4.1 server.
#   bash launch/lan_14_15/bench_matrix.sh 15 [lengths...]
# Every run appends one JSON line to logs/<tag>_stability.jsonl; a failing run
# is recorded with ok=false instead of aborting the matrix.
set -uo pipefail
HOST_NUM=${1:?usage: bench_matrix.sh 14|15 [lengths...]}; shift || true
LENGTHS=("$@")
if [ ${#LENGTHS[@]} -eq 0 ]; then
  LENGTHS=(4096 32768 131072 262144 524288 600000)
fi
MAXNEW=${MAXNEW:-32}
REPEAT=${REPEAT:-1}
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
TAG=${TAG:-$HOST_NUM}
OUT=$REPO/logs/${TAG}_stability.jsonl
ERR=$REPO/logs/${TAG}_client_err.txt
mkdir -p "$REPO/logs"
cd "$REPO"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python

model=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && model=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017

for n in "${LENGTHS[@]}"; do
  for r in $(seq 1 "$REPEAT"); do
    echo "[$(date '+%T')] len=$n rep=$r" >&2
    out=$(python launch/lan_14_15/client.py gen --tokens "$n" --max-new "$MAXNEW" \
          --nonce "$((n + r))" --timeout 7200 --model "$model" 2>"$ERR")
    if ! printf '%s' "$out" | grep -q '"prompt_tokens"'; then
      msg=$(tail -c 400 "$ERR" | tr '\n\t' '  ' | tr -d '"\\')
      printf '{"ts":"%s","len":%s,"rep":%s,"ok":false,"err":"%s"}\n' \
        "$(date '+%FT%T')" "$n" "$r" "$msg" >>"$OUT"
      echo "FAIL len=$n rc=$? :: $(echo "$msg" | tail -c 200)" >&2
      continue
    fi
    printf '%s' "$out" | TAG_LEN=$n TAG_REP=$r python -c '
import json, os, sys
d = json.loads(sys.stdin.read())
d.update(ok=True, len=int(os.environ["TAG_LEN"]), rep=int(os.environ["TAG_REP"]))
print(json.dumps(d))' >>"$OUT"
    tail -1 "$OUT" >&2
  done
done
echo "results: $OUT" >&2
