#!/usr/bin/env bash
# Wipe the L3 (NVMe) KV cache directory.
#   bash clean_l3.sh
# Entries are tied to the build and the profile (DSPARK on/off, replay on/off,
# chunk size), so clean after any of those change or the engine may report
# "Short read" and fall back.
set -uo pipefail
D=${DSV41_KV_NVME:-$HOME/kvcache/l3}
before=$(ls "$D" 2>/dev/null | wc -l)
rm -rf "$D"
mkdir -p "$D"
echo "L3 cleaned: $D (removed $before entries)"
