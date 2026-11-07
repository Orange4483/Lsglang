#!/usr/bin/env bash
# Who waits during prefill?  Samples per-GPU utilisation and host CPU while a
# single request runs, then reports the per-card profile.
#   bash prefill_probe.sh 15 32768 4242 64
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
OUT=/tmp/prefill_samples_$$.txt
: >"$OUT"

python launch/lan_14_15/client.py gen --model "$M" --max-new "$GEN" --tokens "$TOK" \
  --nonce "$NONCE" --timeout 7200 >/tmp/prefill_client_$$.jsonl 2>/dev/null &
C=$!
cpu_prev=$(awk '/^cpu /{print $5" "$2+$3+$4+$5+$6+$7+$8}' /proc/stat)
while kill -0 $C 2>/dev/null; do
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' '|')
  cpu_now=$(awk '/^cpu /{print $5" "$2+$3+$4+$5+$6+$7+$8}' /proc/stat)
  echo "GPU $gpu CPU $cpu_prev $cpu_now" >>"$OUT"
  cpu_prev=$cpu_now
  sleep 0.5
done
wait $C || true
tail -1 /tmp/prefill_client_$$.jsonl

python3 - "$OUT" "$(nproc)" <<'PY'
import re, statistics, sys
gpus, busy = {}, []
prev = None
for line in open(sys.argv[1]):
    m = re.search(r"GPU (.*?) CPU (\d+) (\d+) (\d+) (\d+)$", line.strip())
    if not m:
        continue
    for tok in m.group(1).split("|"):
        if not tok.strip():
            continue
        parts = [p.strip() for p in tok.split(",")]
        if len(parts) >= 2:
            gpus.setdefault(int(parts[0]), []).append(int(parts[1].split()[0]))
    cur = (int(m.group(2)), int(m.group(3)))
    if prev is not None:
        d_idle, d_tot = cur[0] - prev[0], cur[1] - prev[1]
        if d_tot > 0:
            busy.append(100.0 * (d_tot - d_idle) / d_tot)
    prev = cur
ncpu = int(sys.argv[2])
for idx in sorted(gpus):
    v = gpus[idx]
    zeros = sum(1 for x in v if x < 10)
    print(f"GPU{idx}: n={len(v)} mean={statistics.mean(v):.0f}% "
          f"median={statistics.median(v):.0f}% p90={sorted(v)[int(len(v)*0.9)-1]}% "
          f"max={max(v)}% idle_samples={zeros}/{len(v)}")
if busy:
    print(f"CPU: mean_busy={statistics.mean(busy):.0f}% of {ncpu} cores "
          f"=> {statistics.mean(busy)*ncpu/100:.1f} cores busy")
PY
