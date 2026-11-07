#!/usr/bin/env bash
# Hot-expert + chunk + concurrency probe, in the order the report needs it.
set -uo pipefail
REPO=$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165
cd "$REPO"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
model=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
C="python launch/lan_14_15/client.py gen --model $model --max-new 32 --timeout 7200"

echo "== prefill 32k chunk1024"
$C --tokens 32768 --nonce 901
echo "== prefill 131k chunk1024"
$C --tokens 131072 --nonce 902
echo "== hot domain x6"
python launch/lan_14_15/hotcold.py "$model" hot 6
echo "== cold domain (alternating) x6"
python launch/lan_14_15/hotcold.py "$model" cold 6
echo "== dual stream x4"
python launch/lan_14_15/hotcold.py "$model" dual 4
echo "== pool stats"
grep -a '\[pool\]' logs/serve_14.log | tail -8 | cut -c1-220
