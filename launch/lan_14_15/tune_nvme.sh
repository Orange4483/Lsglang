#!/usr/bin/env bash
# NVMe/filesystem tuning for the model disk on .14 / .15.
#   bash tune_nvme.sh            apply
#   bash tune_nvme.sh show       print current values only
# Engram lookups are 264 B random reads; weight streaming is large sequential
# reads.  The knobs below serve both: bigger max_sectors for the streaming
# path, no readahead amplification and no atime writes for the random path.
set -uo pipefail
MODE=${1:-apply}
DEV=nvme0n1
Q=/sys/block/$DEV/queue

show() {
  echo "== $(hostname) $(date '+%T') =="
  findmnt -no SOURCE,TARGET,FSTYPE,OPTIONS / | sed 's/^/root   /'
  findmnt -no SOURCE,TARGET,FSTYPE,OPTIONS /media/hello/Windows 2>/dev/null | sed 's/^/win    /'
  for f in scheduler read_ahead_kb nr_requests max_sectors_kb nomerges rotational; do
    printf '%-16s %s\n' "$f" "$(cat "$Q/$f" 2>/dev/null)"
  done
  grep -E 'shmem_enabled' /sys/kernel/mm/transparent_hugepage/shmem_enabled | sed 's/^/thp    /'
  grep /dev/shm /proc/mounts | sed 's/^/shm    /'
}

if [ "$MODE" = show ]; then show; exit 0; fi

# 1. noatime: the checkpoint is read-only; atime updates are pure write traffic.
sudo -n mount -o remount,noatime / || echo "WARN remount root"
if mountpoint -q /media/hello/Windows; then
  sudo -n mount -o remount,noatime /media/hello/Windows || echo "WARN remount ntfs"
fi
# 2. sequential streaming (model load) gains from 1 MiB requests.
echo 1024 | sudo -n tee "$Q/max_sectors_kb" >/dev/null || echo "WARN max_sectors"
# 3. no I/O scheduler queueing on NVMe.
echo none | sudo -n tee "$Q/scheduler" >/dev/null 2>&1 || true
# 4. 264 B rows do not benefit from 128 KiB readahead; it only inflates traffic.
echo 0 | sudo -n tee "$Q/read_ahead_kb" >/dev/null || echo "WARN read_ahead"
# 5. keep request merging (adjacent engram rows do merge).
echo 0 | sudo -n tee "$Q/nomerges" >/dev/null 2>&1 || true
# 6. shmem THP for the Engram host table.
echo advise | sudo -n tee /sys/kernel/mm/transparent_hugepage/shmem_enabled >/dev/null 2>&1 || true
sudo -n mount -o remount,huge=advise /dev/shm 2>/dev/null || true

show
