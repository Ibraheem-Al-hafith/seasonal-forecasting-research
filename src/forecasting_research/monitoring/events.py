"""Thread-safe publish/subscribe bus shared by CLI and GUI.

Execution components *publish* ProgressEvents; interfaces *subscribe* and render them.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from ..domain.contracts import ProgressEvent

log = logging.getLogger(__name__)
Subscriber = Callable[[ProgressEvent], None]

# Event types
EXPERIMENT_CREATED = "experiment_created"
EXPERIMENT_STARTED = "experiment_started"
PLAN_BUILT = "plan_built"
EVALUATION_STARTED = "evaluation_started"
EVALUATION_STAGE = "evaluation_stage"
EVALUATION_COMPLETED = "evaluation_completed"
EVALUATION_FAILED = "evaluation_failed"
EXPERIMENT_FINISHED = "experiment_finished"
EXPERIMENT_INTERRUPTED = "experiment_interrupted"
GENERATION_STARTED = "generation_started"
GENERATION_PROGRESS = "generation_progress"
GENERATION_FINISHED = "generation_finished"

# Persisted to SQLite (meaningful transitions only). Stage changes are live-only.
DURABLE_EVENTS = {
    EXPERIMENT_CREATED, EXPERIMENT_STARTED, PLAN_BUILT, EVALUATION_STARTED, EVALUATION_COMPLETED,
    EVALUATION_FAILED, EXPERIMENT_FINISHED, EXPERIMENT_INTERRUPTED, GENERATION_STARTED, GENERATION_FINISHED,
}


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._lock = threading.RLock()

    def subscribe(self, cb: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subs.append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._subs:
                    self._subs.remove(cb)

        return unsubscribe

    def publish(self, event: ProgressEvent) -> None:
        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(event)
            except Exception:  # a broken renderer must never break an experiment
                log.exception("progress subscriber failed")
