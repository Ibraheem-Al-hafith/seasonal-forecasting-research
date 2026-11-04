"""Parallel execution strategy: a process pool computes, the parent process persists.

Design (why it is safe)
  * Workers run ONLY `run_work_item` (pure computation on plain data). They never touch SQLite or write
    artifacts. The parent is the single writer: `begin` -> submit -> `commit` happen in the parent, so the
    database / artifact invariants and the safe persistence sequence are exactly those of sequential mode.
  * At most `workers` evaluations are in flight, so the RUNNING rows in the database are real running work
    and a stop request / Ctrl-C never strands a long queue of "RUNNING" evaluations.
  * Stage updates travel from workers to the parent through a queue and are republished on the shared bus.
  * A crashed worker (segfault, OOM kill) fails only the evaluations that were in flight (category UNEXPECTED,
    retryable with --retry-failed); the pool is rebuilt up to `max_worker_restarts` times.
  * Ctrl-C is ignored by workers and handled by the parent, which kills the pool and leaves the in-flight
    evaluations INTERRUPTED (re-run on resume). A persistence failure kills the pool and aborts safely.
  * If the parent dies abruptly (kill -9, OOM), a watchdog thread in each worker makes it exit within ~2 s.
  * One BLAS thread per worker (`limit_blas_threads`) prevents N workers x M BLAS threads oversubscription.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from ..domain.contracts import EvaluationTask, TaskOutcome
from ..domain.enums import FailureCategory, Stage
from ..domain.exceptions import ConfigError
from ..experiments.work import WorkItem, WorkResult, failure_result, run_work_item
from .base import ExecutionStrategy, TaskExecutor
from .sequential import SequentialStrategy

log = logging.getLogger(__name__)

MAX_AUTO_WORKERS = 8
_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
POLL_SECONDS = 0.2


def resolve_workers(workers: int | str) -> int:
    """'auto' -> CPU cores - 1 (at least 1, at most 8: each worker holds roughly 150-250 MB)."""
    if workers == "auto":
        return max(1, min((os.cpu_count() or 2) - 1, MAX_AUTO_WORKERS))
    return max(1, int(workers))


def import_worker_modules(modules: Iterable[str]) -> None:
    """Import modules (that register custom forecasting methods) in THIS process; workers import them too."""
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as exc:
            raise ConfigError(f"Cannot import execution.worker_modules entry {m!r}: {exc}") from exc


@contextlib.contextmanager
def limit_blas_threads(enabled: bool) -> Iterator[None]:
    """Set BLAS/OpenMP thread variables to 1 while workers are spawned (they inherit the environment)."""
    changed: list[str] = []
    if enabled:
        for var in _BLAS_VARS:
            if var not in os.environ:  # never override an explicit user choice
                os.environ[var] = "1"
                changed.append(var)
    try:
        yield
    finally:
        for var in changed:
            os.environ.pop(var, None)


# ---------------------------------------------------------------- worker side (module-level for pickling)
_STAGE_QUEUE = None


def _exit_when_parent_dies() -> None:
    """Workers must never outlive the main process (e.g. after `kill -9` or an OOM kill of the parent)."""
    parent = mp.parent_process()
    if parent is None:
        return
    while True:
        parent.join(timeout=2.0)  # returns early when the parent exits
        if not parent.is_alive():
            os._exit(1)


def _init_worker(stage_queue, modules: list[str]) -> None:
    global _STAGE_QUEUE
    _STAGE_QUEUE = stage_queue
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the parent only
    threading.Thread(target=_exit_when_parent_dies, name="parent-watchdog", daemon=True).start()
    for m in modules:
        importlib.import_module(m)


def _worker_run(item: WorkItem) -> WorkResult:
    def on_stage(stage: Stage) -> None:
        with contextlib.suppress(Exception):
            _STAGE_QUEUE.put_nowait((item.task.evaluation_id, stage.value))

    return run_work_item(item, on_stage)


# ---------------------------------------------------------------- parent side
class ParallelStrategy(ExecutionStrategy):
    def __init__(
        self,
        workers: int,
        start_method: str | None = None,
        worker_modules: Sequence[str] = (),
        max_tasks_per_worker: int | None = None,
        limit_blas: bool = True,
        max_restarts: int = 3,
    ):
        self.workers = max(1, int(workers))
        self.start_method = start_method or "spawn"
        self.worker_modules = list(worker_modules)
        self.max_tasks_per_worker = max_tasks_per_worker
        self.limit_blas = limit_blas
        self.max_restarts = max_restarts

    def describe(self) -> str:
        return f"parallel ({self.workers} workers, {self.start_method})" if self.workers > 1 else "sequential"

    def _cannot_spawn_reason(self) -> str | None:
        """`spawn`/`forkserver` workers re-import the main script; that is impossible for `python -` (stdin)."""
        if self.start_method == "fork":
            return None
        path = getattr(sys.modules.get("__main__"), "__file__", None)
        if path and not os.path.exists(path):
            return f"the main program {path!r} cannot be re-imported by worker processes; run it from a script file"
        return None

    def _new_pool(self, ctx, stage_q) -> ProcessPoolExecutor:
        kw = {}
        if self.max_tasks_per_worker and sys.version_info >= (3, 11) and self.start_method != "fork":
            kw["max_tasks_per_child"] = self.max_tasks_per_worker
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_init_worker,
                                   initargs=(stage_q, self.worker_modules), **kw)

    @staticmethod
    def _shutdown(pool: ProcessPoolExecutor | None, kill: bool) -> None:
        if pool is None:
            return
        procs = list((getattr(pool, "_processes", None) or {}).values())
        if kill:
            pool.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                with contextlib.suppress(Exception):
                    p.terminate()
            for p in procs:
                with contextlib.suppress(Exception):
                    p.join(timeout=5)
        else:
            pool.shutdown(wait=True)

    @staticmethod
    def _drain(stage_q, active: dict[Future, tuple[EvaluationTask, int]], runner: TaskExecutor) -> None:
        """Republish worker stage updates (ignoring late events of evaluations that already finished)."""
        by_id = {t.evaluation_id: t for t, _ in active.values()}
        while True:
            try:
                eid, stage = stage_q.get_nowait()
            except (queue.Empty, OSError, ValueError):
                return
            task = by_id.get(eid)
            if task is not None:
                runner.stage(task, Stage(stage))

    @staticmethod
    def _worker_died(task: EvaluationTask) -> WorkResult:
        return failure_result(
            task.evaluation_id, FailureCategory.UNEXPECTED,
            "Worker process terminated unexpectedly (crash or out-of-memory kill) while this evaluation, "
            "or a concurrent one, was running. Retry with: research resume <id> --retry-failed")

    def execute(self, tasks: Sequence[EvaluationTask], runner: TaskExecutor, should_stop: Callable[[], bool]) -> list[TaskOutcome]:
        if self.workers <= 1 or len(tasks) <= 1:
            return SequentialStrategy().execute(tasks, runner, should_stop)
        problem = self._cannot_spawn_reason()
        if problem:
            log.warning("parallel execution unavailable (%s); running sequentially instead", problem)
            return SequentialStrategy().execute(tasks, runner, should_stop)
        import_worker_modules(self.worker_modules)
        ctx = mp.get_context(self.start_method)
        stage_q = ctx.Queue()
        outcomes: list[TaskOutcome] = []
        pending: deque[EvaluationTask] = deque(tasks)
        active: dict[Future, tuple[EvaluationTask, int]] = {}
        pool: ProcessPoolExecutor | None = None
        restarts, broken, clean = 0, False, False
        try:
            with limit_blas_threads(self.limit_blas):
                pool = self._new_pool(ctx, stage_q)
                while pending or active:
                    if broken and not active:  # every in-flight future of the dead pool has been resolved
                        self._shutdown(pool, kill=True)
                        restarts += 1
                        if restarts > self.max_restarts:
                            log.critical("worker pool crashed %d times; stopping with %d evaluations unstarted "
                                         "(resume to continue)", restarts, len(pending))
                            break
                        log.error("worker process died; rebuilding the pool (%d/%d)", restarts, self.max_restarts)
                        pool, broken = self._new_pool(ctx, stage_q), False

                    # keep exactly `workers` evaluations in flight (RUNNING rows == really running work)
                    while pending and not broken and len(active) < self.workers and not should_stop():
                        task = pending.popleft()
                        attempt = runner.begin(task)
                        item = runner.work_item(task)
                        if isinstance(item, WorkResult):  # could not even build the work item
                            outcomes.append(runner.commit(task, attempt, item))
                            continue
                        try:
                            active[pool.submit(_worker_run, item)] = (task, attempt)
                        except BrokenProcessPool:
                            outcomes.append(runner.commit(task, attempt, self._worker_died(task)))
                            broken = True

                    if not active:
                        if broken:
                            continue
                        break  # nothing in flight and nothing may be started (finished, or stop requested)

                    self._drain(stage_q, active, runner)
                    done, _ = wait(list(active), timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
                    self._drain(stage_q, active, runner)
                    for fut in done:
                        task, attempt = active.pop(fut)
                        try:
                            result = fut.result()
                        except BrokenProcessPool:
                            broken, result = True, self._worker_died(task)
                        except Exception as exc:  # e.g. result could not be pickled / worker raised unexpectedly
                            result = failure_result(task.evaluation_id, FailureCategory.UNEXPECTED,
                                                    f"Worker error: {type(exc).__name__}: {exc}")
                        outcomes.append(runner.commit(task, attempt, result))
                clean = True
        finally:
            # clean exit: wait for idle workers. KeyboardInterrupt / PersistenceError: kill them now.
            self._shutdown(pool, kill=not clean)
            with contextlib.suppress(Exception):
                stage_q.close()
                stage_q.cancel_join_thread()
        return outcomes
