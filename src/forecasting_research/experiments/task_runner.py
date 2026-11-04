"""Runs evaluation tasks with persistence. Parent-process side of the execution machinery.

An evaluation is split into three steps so a parallel strategy can overlap the expensive middle step:

  begin()      parent : PENDING/... -> RUNNING, attempt row, STARTED event
  work_item()  parent : build the plain-data WorkItem (reads the dataset record)
  run_work_item()     : pure computation (parent process OR worker process)  -> experiments/work.py
  commit()     parent : write artifact, then COMPLETED + metrics in ONE transaction (or record FAILED)

`run()` chains the three for the sequential strategy.

Error policy
  * model / data / forecast problems   -> evaluation FAILED (category recorded), run continues
  * persistence problems               -> PersistenceError propagates: the experiment stops safely
An evaluation is marked COMPLETED only after its artifact is safely written and verified.
"""

from __future__ import annotations

import logging
import time

from ..config.schemas import AppConfig
from ..domain.contracts import DatasetRecord, EvaluationTask, ProgressEvent, TaskOutcome
from ..domain.enums import EvaluationStatus, FailureCategory, Stage
from ..domain.exceptions import PersistenceError, ResearchError
from ..monitoring import events as E
from ..monitoring.events import EventBus
from ..monitoring.logging import log_ctx
from ..persistence.artifacts import ArtifactStore
from ..persistence.repositories import Repository
from .work import WorkItem, WorkResult, failure_result, run_work_item

log = logging.getLogger(__name__)


class TaskRunner:
    def __init__(self, repo: Repository, store: ArtifactStore, cfg: AppConfig, bus: EventBus):
        self.repo, self.store, self.cfg, self.bus = repo, store, cfg, bus
        self._eval_cfg = cfg.evaluation.model_dump(mode="json")
        self._fc_cfg = cfg.forecasting.model_dump(mode="json")
        self._record: DatasetRecord | None = None  # last dataset record (3 methods share one dataset)

    # ---- helpers -----------------------------------------------------------------------
    def __call__(self, task: EvaluationTask) -> TaskOutcome:
        return self.run(task)

    def _ctx(self, task: EvaluationTask) -> dict:
        return log_ctx(experiment_id=task.experiment_id, evaluation_id=task.evaluation_id,
                       dataset_id=task.dataset_id, model_name=task.model_name)

    def _emit(self, task: EvaluationTask, event_type: str, **kw) -> None:
        self.bus.publish(ProgressEvent(
            event_type=event_type, experiment_id=task.experiment_id, evaluation_id=task.evaluation_id,
            dataset_id=task.dataset_id, model_name=task.model_name, **kw))

    def stage(self, task: EvaluationTask, stage: Stage) -> None:
        self.repo.evaluations.set_stage(task.evaluation_id, stage.value)
        self._emit(task, E.EVALUATION_STAGE, stage=stage.value, status="RUNNING")

    # ---- step 1: begin -----------------------------------------------------------------
    def begin(self, task: EvaluationTask) -> int:
        attempt = self.repo.evaluations.begin_attempt(task.evaluation_id)
        self._emit(task, E.EVALUATION_STARTED, status="RUNNING",
                   message=f"Started {task.model_name} on {task.dataset_id} (attempt {attempt})")
        log.info("evaluation started (attempt %d)", attempt, extra=self._ctx(task))
        return attempt

    # ---- step 2: build the work item ---------------------------------------------------
    def work_item(self, task: EvaluationTask) -> WorkItem | WorkResult:
        """Returns a WorkItem, or a failure WorkResult if the dataset cannot even be looked up."""
        try:
            if self._record is None or self._record.dataset_id != task.dataset_id:
                self._record = self.repo.datasets.get_dataset(task.dataset_id)
            r = self._record
            return WorkItem(task, str(self.store.workspace), r.observed_path, r.observed_sha256,
                            r.seasonality_type, r.seasonal_period, r.length, self._eval_cfg, self._fc_cfg)
        except PersistenceError:
            raise
        except ResearchError as exc:
            return failure_result(task.evaluation_id, exc.category, f"{type(exc).__name__}: {exc}")

    # ---- step 4: commit ----------------------------------------------------------------
    def commit(self, task: EvaluationTask, attempt: int, result: WorkResult) -> TaskOutcome:
        ctx = self._ctx(task)
        if result.outcome is None:
            return self._record_failure(task, attempt, FailureCategory(result.failure_category or "UNEXPECTED"),
                                        result.failure_detail or "unknown failure", result.compute_seconds, ctx)
        out = result.outcome
        t0 = time.perf_counter()
        try:
            self.stage(task, Stage.PERSIST_RESULTS)
            rel, sha = self.store.save_evaluation(task.evaluation_id, out.actual, out.forecast)
            runtime = result.compute_seconds + (time.perf_counter() - t0)  # real work time, comparable across modes
            self.repo.evaluations.complete(
                task.evaluation_id, attempt, parameters=out.parameters, metrics=out.metrics, warnings=out.warnings,
                artifact_path=rel, artifact_sha256=sha, runtime_seconds=runtime,
                train_size=out.train_size, test_size=out.test_size, horizon=out.horizon)
        except PersistenceError as exc:
            self._fail_best_effort(task, attempt, exc, result.compute_seconds)
            log.error("persistence failure - stopping: %s", exc, extra=ctx)
            raise
        self._emit(task, E.EVALUATION_COMPLETED, status="COMPLETED", stage=Stage.COMPLETE.value,
                   message=f"Completed {task.model_name} on {task.dataset_id} in {runtime:.1f}s")
        log.info("evaluation completed in %.2fs", runtime, extra=ctx)
        return TaskOutcome(task, EvaluationStatus.COMPLETED.value, runtime_seconds=runtime)

    # ---- sequential convenience ----------------------------------------------------------
    def run(self, task: EvaluationTask) -> TaskOutcome:
        attempt = self.begin(task)
        item = self.work_item(task)
        result = item if isinstance(item, WorkResult) else run_work_item(item, lambda s: self.stage(task, s))
        return self.commit(task, attempt, result)

    # ---- failures --------------------------------------------------------------------------
    def _record_failure(self, task, attempt, category: FailureCategory, detail: str, runtime: float, ctx) -> TaskOutcome:
        self.repo.evaluations.fail(task.evaluation_id, attempt, category.value, detail, runtime)
        summary = detail.splitlines()[0][:200]
        self._emit(task, E.EVALUATION_FAILED, status="FAILED",
                   message=f"FAILED {task.model_name} on {task.dataset_id}: {summary}", payload={"category": category.value})
        log.warning("evaluation failed [%s]: %s", category.value, summary, extra=ctx)
        return TaskOutcome(task, EvaluationStatus.FAILED.value, category.value, summary, runtime)

    def _fail_best_effort(self, task, attempt, exc: Exception, runtime: float) -> None:
        try:
            self.repo.evaluations.fail(task.evaluation_id, attempt, FailureCategory.PERSISTENCE_ERROR.value, str(exc), runtime)
        except Exception:  # the database itself may be what is failing; stale RUNNING is recovered on resume
            pass
