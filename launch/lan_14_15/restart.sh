#!/usr/bin/env bash
# Orderly shutdown of everything, with a status snapshot, for a clean restart.
#   bash launch/lan_14_15/restart.sh 15 [ENV=...] -- [extra args]
set -uo pipefail
HOST_NUM=${1:?usage: restart.sh 14|15}; shift || true
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
cd "$REPO"
echo "== stop $(date '+%T')"
bash launch/lan_14_15/stop.sh "$HOST_NUM" --all
sleep 5
echo "== shm/thp"; bash launch/lan_14_15/tune_nvme.sh show | tail -3
echo "== start $(date '+%T')"
bash launch/lan_14_15/start.sh "$HOST_NUM" "$@"
echo "== waiting for /model_info (up to 30 min)"
python3 - "$HOST_NUM" <<'PY'
import json, sys, time, urllib.request
host_num = sys.argv[1]
want = "DeepSeek-V4.1-Flash-1017"
t0 = time.time()
while time.time() - t0 < 1800:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8070/model_info", timeout=5) as r:
            if want in json.dumps(json.loads(r.read())):
                print(f"UP after {time.time()-t0:.0f}s")
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass
    time.sleep(10)
print("TIMEOUT")
raise SystemExit(1)
PY
