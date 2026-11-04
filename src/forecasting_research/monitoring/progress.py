"""Progress aggregation: overview counters, current operation, ETA.

Progress = (completed + failed) / total x 100. Failed work counts as processed but is shown separately.
ETA is shown only after at least `MIN_SAMPLES_FOR_ETA` evaluations finished in this session.
No fake percentages are invented for operations that cannot be measured: only stage names are shown.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from ..domain.contracts import ProgressEvent
from . import events as E

MIN_SAMPLES_FOR_ETA = 3


class ProgressTracker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.experiment_id: str | None = None
            self.status = "IDLE"
            self.total = self.completed = self.failed = 0
            self.running: dict[str, dict[str, Any]] = {}
            self.activity = "Idle"
            self.started_monotonic: float | None = None
            self.ended_monotonic: float | None = None
            self.session_durations: list[float] = []
            self.first_done: float | None = None  # monotonic time of the first evaluation finished this session
            self.generation = {"done": 0, "total": 0}
            self.log: deque[str] = deque(maxlen=200)
            self._eval_started: dict[str, float] = {}
            self.final_status: str | None = None
            self.final_outcome: str | None = None

    # ---- subscriber --------------------------------------------------------------------
    def on_event(self, ev: ProgressEvent) -> None:
        with self._lock:
            t = ev.event_type
            if ev.experiment_id and t in (E.EXPERIMENT_CREATED, E.EXPERIMENT_STARTED):
                if self.experiment_id != ev.experiment_id:
                    self.reset()
                    self.experiment_id = ev.experiment_id
            if t == E.EXPERIMENT_STARTED:
                self.status = "RUNNING"
                self.started_monotonic = time.monotonic()
                self.ended_monotonic = None
                self.activity = "Starting"
            elif t == E.PLAN_BUILT:
                p = ev.payload
                self.total, self.completed, self.failed = p["total"], p.get("completed", 0), p.get("failed", 0)
            elif t == E.EVALUATION_STARTED and ev.evaluation_id:
                self.running[ev.evaluation_id] = {"dataset_id": ev.dataset_id, "model_name": ev.model_name, "stage": "starting"}
                self._eval_started[ev.evaluation_id] = time.monotonic()
                self.activity = f"{ev.model_name} on {ev.dataset_id}"
            elif t == E.EVALUATION_STAGE and ev.evaluation_id in self.running:
                self.running[ev.evaluation_id]["stage"] = ev.stage
                self.activity = f"{ev.model_name} on {ev.dataset_id}: {ev.stage}"
            elif t in (E.EVALUATION_COMPLETED, E.EVALUATION_FAILED) and ev.evaluation_id:
                self.running.pop(ev.evaluation_id, None)
                if t == E.EVALUATION_COMPLETED:
                    self.completed += 1
                else:
                    self.failed += 1
                t0 = self._eval_started.pop(ev.evaluation_id, None)
                if t0 is not None:
                    self.session_durations.append(time.monotonic() - t0)
                    if self.first_done is None:
                        self.first_done = time.monotonic()
            elif t == E.GENERATION_STARTED:
                self.generation = {"done": 0, "total": ev.payload.get("total", 0)}
                self.activity = "Generating datasets"
                self.status = "GENERATING"
            elif t == E.GENERATION_PROGRESS:
                self.generation["done"] = ev.payload.get("done", 0)
                self.activity = f"Generating datasets ({self.generation['done']}/{self.generation['total']})"
            elif t == E.GENERATION_FINISHED:
                self.activity = "Generation finished"
                self.status = "IDLE" if self.experiment_id is None else self.status
            elif t in (E.EXPERIMENT_FINISHED, E.EXPERIMENT_INTERRUPTED):
                self.running.clear()
                self.status = ev.payload.get("status", "INTERRUPTED")
                self.final_status = self.status
                self.final_outcome = ev.payload.get("outcome")
                self.ended_monotonic = time.monotonic()
                self.activity = "Finished" if t == E.EXPERIMENT_FINISHED else "Interrupted"
            if ev.message:
                self.log.append(f"{ev.timestamp[11:19]}  {ev.message}")

    # ---- snapshot ----------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            running = len(self.running)
            processed = self.completed + self.failed
            pending = max(self.total - processed - running, 0)
            percent = (processed / self.total * 100.0) if self.total else 0.0
            elapsed = None
            if self.started_monotonic is not None:
                end = self.ended_monotonic if self.ended_monotonic is not None else time.monotonic()
                elapsed = end - self.started_monotonic
            eta = None
            n = len(self.session_durations)
            if n >= MIN_SAMPLES_FOR_ETA and self.first_done is not None and self.status == "RUNNING":
                # observed throughput (evaluations / wall-clock second): valid for sequential AND parallel runs
                span = time.monotonic() - self.first_done
                if span > 0:
                    eta = (pending + running) / ((n - 1) / span) if n > 1 else None
            return {
                "experiment_id": self.experiment_id,
                "status": self.status,
                "outcome": self.final_outcome,
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "running": running,
                "pending": pending,
                "percent": round(percent, 1),
                "activity": f"{running} evaluations running in parallel" if running > 1 else self.activity,
                "current": list(self.running.values()),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "generation": dict(self.generation),
                "log": list(self.log)[-30:],
            }


def format_seconds(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"
