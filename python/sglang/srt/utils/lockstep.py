"""TP lockstep observability: per-rank all-reduce execution tallies.

A custom-allreduce deadlock (counter-equality spin in
``cross_device_reduce_1stage``) is always the tail of a call-count skew:
one rank executed one more (or one fewer) AR than its peer at some point,
and from then on every counter handshake mismatches forever. The skew
event itself is silent and the hang surfaces later, elsewhere.

This module keeps a cheap per-rank tally of everything that advances the
AR counters -- eager AR calls, calls captured into each graph shape, and
per-graph replay counts -- and fires one loud tripwire when a decode
step runs eagerly on a process that otherwise replays decode graphs.
It changes no control flow: it never catches, retries, switches paths,
or degrades anything. Counting is a module-global int increment (the GIL
makes these safe as advisory tallies); nothing here touches CUDA.
"""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Dict

logger = logging.getLogger(__name__)

_eager_ar_calls = 0
_captured_ar_calls = 0
_graph_ar_calls: Dict[str, int] = {}
_replay_counts: Dict[str, int] = {}

_graphs_replayed = threading.Event()
_decode_eager_logged = 0
_DECODE_EAGER_LOG_LIMIT = 3


def captured_count() -> int:
    return _captured_ar_calls


def note_eager_ar_call() -> None:
    """Count one eager (executing now) all-reduce kernel call."""
    global _eager_ar_calls
    _eager_ar_calls += 1


def note_captured_ar_call() -> None:
    """Count one all-reduce recorded into a cuda graph. Its execution is
    deferred to replays, so it must not count as eager."""
    global _captured_ar_calls
    _captured_ar_calls += 1


def note_graph_ar_calls(shape_key: Any, calls: int) -> None:
    _graph_ar_calls[str(shape_key)] = calls


def note_graph_replay(shape_key: Any) -> None:
    key = str(shape_key)
    _graphs_replayed.set()
    _replay_counts[key] = _replay_counts.get(key, 0) + 1


def note_decode_eager_fallback(reason: str) -> None:
    """Tripwire: a decode step is taking the eager path on a process that
    has replayed decode graphs before. If both ranks log it, this bs is
    simply uncaptured; if only one rank logs it, that step is exactly
    where the TP collective streams split -- the seed of an AR deadlock.
    Log loudly with the stack; change nothing."""
    global _decode_eager_logged
    if not _graphs_replayed.is_set():
        return
    if _decode_eager_logged >= _DECODE_EAGER_LOG_LIMIT:
        return
    _decode_eager_logged += 1
    site = "".join(traceback.format_stack(limit=14)[:-1])
    logger.error(
        "[lockstep] decode step running EAGER while decode graphs are in "
        "use (%s) -- if the peer rank does not log the same line, collective "
        "call counts now diverge:\n%s",
        reason,
        site,
    )


def snapshot() -> Dict[str, Any]:
    total = _eager_ar_calls + sum(
        n * _graph_ar_calls.get(k, 0) for k, n in _replay_counts.items()
    )
    return {
        "eager_ar": _eager_ar_calls,
        "captured_ar": _captured_ar_calls,
        "graph_ar_calls": dict(_graph_ar_calls),
        "replays": dict(_replay_counts),
        "total_ar_exec": total,
    }
