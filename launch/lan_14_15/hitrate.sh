#!/usr/bin/env bash
# Expert hit rate for a continuous coding workload.
# The pool logs cumulative counts (h = hits, m = misses, p = pumps, resident =
# distinct resident experts); the coding session's own rate is the delta across
# the run, not the since-boot figure.
set -uo pipefail
cd "$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate Lsglang165
export PYTHONPATH=$PWD/python
M=/media/hello/Windows/models/DeepSeek-V4.1-Flash-1017
L=logs/serve_14.log
TURNS=${TURNS:-5}
GEN=${GEN:-400}
TOKENS=${TOKENS:-24000}

snap() { grep -a 'cum hit' "$L" | tail -1; }
before=$(snap)
echo "before: $before"

python launch/lan_14_15/scenario_agent.py --model "$M" --tokens "$TOKENS" \
  --turns "$TURNS" --gen "$GEN" --nonce "${NONCE:-555}" 2>/dev/null | tail -3

after=$(snap)
echo "after:  $after"

python3 - "$before" "$after" <<'PY'
import re, sys
def nums(line):
    m = re.search(r'cum hit=([\d.]+)% h=(\d+) m=(\d+) p=(\d+).*resident=(\d+)', line)
    return tuple(float(x) if '.' in x else int(x) for x in m.groups()) if m else None
b, a = nums(sys.argv[1]), nums(sys.argv[2])
if b and a:
    h, m = a[1] - b[1], a[2] - b[2]
    tot = h + m
    print(f"coding-session hit rate: {h}/{tot} = {100.0*h/tot:.1f}%"
          f"  (resident {b[4]} -> {a[4]}, pumps +{a[3]-b[3]})")
else:
    print("no cum line", sys.argv[1], sys.argv[2])
PY
