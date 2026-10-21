"""copyright (c) 2025-2026 guqiong9696@gmail.com, All Rights Reserved.
NVFP4 MoE backend auto-resolution on SM12x (sglang#35617 port): the method
__init__ must resolve auto -> flashinfer_cutlass BEFORE weight prep when
capability >= 12, leave explicit backends untouched, and leave other archs
alone. Run: python test/manual/test_nvfp4_sm120_backend_resolution.py
"""
import sys

sys.argv = [sys.argv[0]]

import sglang.srt.layers.quantization.modelopt_quant as mq
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_flags

fl = get_flags().moe


def _make_method():
    cfg = mq.ModelOptFp4Config.__new__(mq.ModelOptFp4Config)
    cfg.is_checkpoint_nvfp4_serialized = True
    return mq.ModelOptNvFp4FusedMoEMethod.__new__(mq.ModelOptNvFp4FusedMoEMethod)


def _run(cap, backend):
    fl.runner_backend = backend
    mq.is_cuda = lambda: True
    mq.get_device_capability = lambda: cap
    saved_plat = mq.get_platform
    class _P:
        is_blackwell = True
    mq.get_platform = lambda: _P()
    m = _make_method()
    try:
        mq.ModelOptNvFp4FusedMoEMethod.__init__(m, cfg=_cfg_stub())
    except TypeError:
        mq.ModelOptNvFp4FusedMoEMethod.__init__(m, _cfg_stub())
    mq.get_platform = saved_plat
    return fl.runner_backend


def _cfg_stub():
    c = mq.ModelOptFp4Config.__new__(mq.ModelOptFp4Config)
    return c


fails = []

def check(name, cond):
    print(f"{name}: {'OK' if cond else 'FAIL'}")
    if not cond:
        fails.append(name)

check("sm120 auto -> cutlass",
      _run((12, 0), None) == MoeRunnerBackend.FLASHINFER_CUTLASS)
check("sm121 auto -> cutlass",
      _run((12, 1), None) == MoeRunnerBackend.FLASHINFER_CUTLASS)
check("sm100 auto untouched",
      _run((10, 0), None) in (None, MoeRunnerBackend.AUTO))
check("sm89 auto untouched",
      _run((8, 9), None) in (None, MoeRunnerBackend.AUTO))
check("sm120 explicit trtllm kept",
      _run((12, 0), MoeRunnerBackend.FLASHINFER_TRTLLM) == MoeRunnerBackend.FLASHINFER_TRTLLM)
check("sm120 explicit marlin kept",
      _run((12, 0), MoeRunnerBackend.MARLIN) == MoeRunnerBackend.MARLIN)

print("ALL OK" if not fails else f"FAILS: {fails}")
sys.exit(1 if fails else 0)
