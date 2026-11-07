#!/usr/bin/env bash
# Stop a DSV4.1 server on this host: whole session first, then any stray workers.
#   bash launch/lan_14_15/stop.sh [14|15] [--all]
# --all also kills anything else in this repo tree (e.g. a previous version's
# server that still owns port 8070).
set -uo pipefail
HOST_NUM=${1:-15}
MODE=${2:-}
REPO=${REPO:-$HOME/project/Lsglang-egram-nvme-kv-nvme/lsglang165}
cd "$REPO"
PIDFILE=logs/serve_${HOST_NUM}.pid

stop_pgid() {
  local pgid=$1
  kill -TERM -"$pgid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -"$pgid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -"$pgid" 2>/dev/null || true
}

if [ -f "$PIDFILE" ]; then
  PGID=$(cat "$PIDFILE")
  if kill -0 "$PGID" 2>/dev/null; then
    echo "stopping session pgid=$PGID"
    stop_pgid "$PGID"
  fi
  rm -f "$PIDFILE"
fi

# stray scheduler/detokenizer workers from an earlier run in this tree
pkill -TERM -f "sglang::scheduler" 2>/dev/null || true
pkill -TERM -f "sglang::detokenizer" 2>/dev/null || true
pkill -TERM -f "$REPO/python/sglang" 2>/dev/null || true
sleep 3
pkill -KILL -f "sglang::scheduler" 2>/dev/null || true
pkill -KILL -f "sglang::detokenizer" 2>/dev/null || true

if [ "$MODE" = "--all" ]; then
  pkill -TERM -f "sglang.launch_server" 2>/dev/null || true
  pkill -TERM -f "sglang serve" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
  pkill -KILL -f "sglang serve" 2>/dev/null || true
fi
echo "stopped (port ${PORT:-8070} check):"
ss -ltn "sport = :${PORT:-8070}" 2>/dev/null || true
