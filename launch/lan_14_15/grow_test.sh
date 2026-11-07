#!/usr/bin/env bash
# Growing-context run: one constant prefix seed, increasing lengths, so every
# request extends the previous one -- the shape of a real agent session and the
# OOM check the user asked for.  Stops at the first failed length.
#   bash grow_test.sh 14 <tag> [lengths...]
set -uo pipefail
HOST_NUM=${1:?usage: grow_test.sh 14|15 tag [lengths...]}; shift
TAG=${1:?tag}; shift || true
LENGTHS=("$@")
[ ${#LENGTHS[@]} -eq 0 ] && LENGTHS=(32768 131072 524288 600000)
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
OUT=$REPO/logs/${TAG}.jsonl
ERR=$REPO/logs/${TAG}_err.txt
cd "$REPO"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
model=$HOME/models/DeepSeek-V4.1-Flash-1017
[ "$HOST_NUM" = 14 ] && model=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
NONCE=${PREFIX_NONCE:-4242}

for n in "${LENGTHS[@]}"; do
  echo "[$(date '+%T')] len=$n" >&2
  out=$(python launch/lan_14_15/client.py gen --tokens "$n" --max-new 32 \
        --prefix-nonce "$NONCE" --timeout 7200 --model "$model" 2>"$ERR")
  if ! printf '%s' "$out" | grep -q '"prompt_tokens"'; then
    printf '{"ts":"%s","len":%s,"ok":false,"err":"%s"}\n' "$(date '+%FT%T')" "$n" \
      "$(tail -c 300 "$ERR" | tr '\n\t' '  ' | tr -d '"\\')" >>"$OUT"
    echo "FAIL len=$n :: $(tail -c 200 "$ERR" | tr '\n' ' ')" >&2
    exit 1
  fi
  printf '%s\n' "$out" >>"$OUT"
  tail -1 "$OUT" >&2
done
echo "results: $OUT" >&2
