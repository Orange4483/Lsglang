#!/usr/bin/env bash
# Apply the bounded-replay Full-prefix fix to an installed lsglang 1.7.3 tree.
#
#   bash launch/173/patches/apply.sh [env-prefix]      # default: $HOME/envs/Lsglang173
#
# The patch is idempotent and keeps a .orig copy so `--revert` restores stock
# behaviour. It only enables the fix when the launcher exports
# LK_SWA_BOUNDED_FULL_PREFIX=1 (see launch/173/serve_173.sh).
set -uo pipefail

ENV_PREFIX=${1:-$HOME/envs/Lsglang173}
PY=$ENV_PREFIX/bin/python
F=$ENV_PREFIX/lib/python3.12/site-packages/sglang/srt/mem_cache/unified_cache/unified_tree_core.py
MODE=${2:-apply}

if [ ! -f "$F" ]; then
  echo "not found: $F" >&2
  exit 1
fi

if [ "$MODE" = "--revert" ]; then
  if [ -f "$F.orig" ]; then
    cp -f "$F.orig" "$F"
    echo "reverted $F"
  else
    echo "no .orig backup next to $F" >&2
    exit 1
  fi
  exit 0
fi

if [ ! -f "$F.orig" ]; then
  cp -f "$F" "$F.orig"
fi

$PY - "$F" <<'PY'
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "LK_SWA_BOUNDED_FULL_PREFIX" in src:
    print("already patched")
    raise SystemExit(0)

if "from sglang.srt.runtime_context import get_exec" not in src:
    anchor = "from sglang.srt.mem_cache.base_prefix_cache import ("
    assert anchor in src, "import anchor missing"
    src = src.replace(anchor, "from sglang.srt.runtime_context import get_exec\n" + anchor, 1)
if "\nimport os\n" not in src:
    src = src.replace("\nimport logging\n", "\nimport logging\nimport os\n", 1)

anchor = """        # Expose only NodeIds outside TreeCore.
        return result._replace("""
patch = """        # Bounded replay keeps no SWA KV for a cached prefix (the request's SWA
        # eviction cursor sits at the end), so the component-based prefix length
        # collapses to 0 even though the Full KV is fully resident. Expose the
        # Full-device prefix and let prefill replay the SWA window from it.
        if (
            os.environ.get("LK_SWA_BOUNDED_FULL_PREFIX") == "1"
            and get_exec().features.enable_decoder_swa_bounded_replay
            and result.device_prefix_len == 0
            and result.full_kv_hit_length > 0
        ):
            matched_len, node_id, _pinned = self.match_full_device_prefix(params.key)
            aligned = matched_len // self.page_size * self.page_size
            if aligned > 0:
                result = result._replace(
                    device_prefix_len=aligned,
                    last_device_node=self.node_by_id(node_id),
                )

""" + anchor
assert anchor in src, "post-processor anchor missing"
src = src.replace(anchor, patch, 1)
open(path, "w", encoding="utf-8").write(src)
print("patched")
PY

$PY -c "import ast; ast.parse(open('$F').read()); print('syntax ok')"
