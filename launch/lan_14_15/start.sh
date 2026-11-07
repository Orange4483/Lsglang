#!/usr/bin/env bash
# Start the DSV4.1 server on this host in its own session.
#   bash launch/lan_14_15/start.sh 14|15 [extra sglang args...]
set -euo pipefail
HOST_NUM=${1:?usage: start.sh 14|15 [extra args]}; shift || true
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
PORT=${PORT:-8070}
cd "$REPO"
mkdir -p logs
# The Engram host table is a 94 GiB shmem mapping read row-by-row.  Two settings
# are required for it to be 2 MiB pages instead of 4 KiB ones (one TLB miss per
# row): the sysfs knob AND the tmpfs mount option -- sysfs alone is not enough,
# /dev/shm must be mounted with huge=advise.
echo advise | sudo -n tee /sys/kernel/mm/transparent_hugepage/shmem_enabled >/dev/null 2>&1 || \
  echo "WARN: could not set shmem_enabled (host table will use 4K pages)"
sudo -n mount -o remount,huge=advise /dev/shm 2>/dev/null || \
  echo "WARN: could not remount /dev/shm huge=advise"
PIDFILE=logs/serve_${HOST_NUM}.pid
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  echo "ERROR: port $PORT already in use:"; ss -ltnp "sport = :$PORT" || true; exit 1
fi
setsid nohup bash "launch/lan_14_15/serve_${HOST_NUM}.sh" "$@" >/dev/null 2>&1 &
echo $! >"$PIDFILE"
echo "started host=$HOST_NUM pid=$(cat "$PIDFILE") port=$PORT log=$REPO/logs/serve_${HOST_NUM}.log"
