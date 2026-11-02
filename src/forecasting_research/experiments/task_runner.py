"""Runs ONE evaluation task end-to-end (steps 1-13 of the evaluation contract) with persistence.

Error policy
  * model / data / forecast problems   -> evaluation FAILED (category recorded), run continues
  * persistence problems               -> PersistenceError propagates: the experiment stops safely
An evaluation is marked COMPLETED only after its artifact is safely written and verified.
"""

from __future__ import annotations

import logging
import time
import traceback

import numpy as np

from ..config.schemas import AppConfig
from ..domain.contracts import DatasetRecord, EvaluationTask, ProgressEvent, TaskOutcome
from ..domain.enums import EvaluationStatus, FailureCategory, Stage
from ..domain.exceptions import ArtifactIntegrityError, DataError, PersistenceError, ResearchError
from ..evaluation.evaluator import Evaluator
from ..forecasting.registry import create_method
from ..monitoring import events as E
from ..monitoring.events import EventBus
from ..monitoring.logging import log_ctx
from ..persistence.artifacts import ArtifactStore
from ..persistence.repositories import Repository

log = logging.getLogger(__name__)


class TaskRunner:
    def __init__(self, repo: Repository, store: ArtifactStore, cfg: AppConfig, bus: EventBus):
        self.repo, self.store, self.cfg, self.bus = repo, store, cfg, bus
        self._cache: tuple[str, DatasetRecord, np.ndarray] | None = None  # one dataset at a time

    # ---- helpers -----------------------------------------------------------------------
    def _emit(self, task: EvaluationTask, event_type: str, **kw) -> None:
        self.bus.publish(ProgressEvent(
            event_type=event_type, experiment_id=task.experiment_id, evaluation_id=task.evaluation_id,
            dataset_id=task.dataset_id, model_name=task.model_name, **kw))

    def _stage(self, task: EvaluationTask, stage: Stage) -> None:
        self.repo.evaluations.set_stage(task.evaluation_id, stage.value)
        self._emit(task, E.EVALUATION_STAGE, stage=stage.value, status="RUNNING")

    def _load(self, dataset_id: str) -> tuple[DatasetRecord, np.ndarray]:
        if self._cache and self._cache[0] == dataset_id:
            return self._cache[1], self._cache[2]
        rec = self.repo.datasets.get_dataset(dataset_id)
        try:  # only the OBSERVED artifact is ever read here; ground truth is never loaded
            y = self.store.load_observed(rec.observed_path, rec.observed_sha256)
        except ArtifactIntegrityError as exc:  # a bad dataset fails its evaluations, it does not abort the run
            raise DataError(str(exc)) from exc
        if y.shape != (rec.length,):
            raise DataError(f"Dataset {dataset_id} length {y.shape} does not match metadata ({rec.length})")
        self._cache = (dataset_id, rec, y)
        return rec, y

    # ---- main --------------------------------------------------------------------------
    def run(self, task: EvaluationTask) -> TaskOutcome:
        ctx = log_ctx(experiment_id=task.experiment_id, evaluation_id=task.evaluation_id,
                      dataset_id=task.dataset_id, model_name=task.model_name)
        attempt = self.repo.evaluations.begin_attempt(task.evaluation_id)
        t0 = time.perf_counter()
        self._emit(task, E.EVALUATION_STARTED, status="RUNNING",
                   message=f"Started {task.model_name} on {task.dataset_id} (attempt {attempt})")
        log.info("evaluation started (attempt %d)", attempt, extra=ctx)
        try:
            self._stage(task, Stage.LOAD_DATASET)
            rec, y = self._load(task.dataset_id)
            method = create_method(task.model_name, self.cfg.forecasting)
            outcome = Evaluator(self.cfg.evaluation).evaluate(
                y, rec.seasonality_type, rec.seasonal_period, method, on_stage=lambda s: self._stage(task, s))

            self._stage(task, Stage.PERSIST_RESULTS)
            rel, sha = self.store.save_evaluation(task.evaluation_id, outcome.actual, outcome.forecast)
            runtime = time.perf_counter() - t0
            self.repo.evaluations.complete(
                task.evaluation_id, attempt, parameters=outcome.parameters, metrics=outcome.metrics,
                warnings=outcome.warnings, artifact_path=rel, artifact_sha256=sha, runtime_seconds=runtime,
                train_size=outcome.train_size, test_size=outcome.test_size, horizon=outcome.horizon)
        except PersistenceError as exc:
            self._fail_best_effort(task, attempt, exc, time.perf_counter() - t0)
            log.error("persistence failure - stopping: %s", exc, extra=ctx)
            raise
        except ResearchError as exc:
            return self._record_failure(task, attempt, exc.category, f"{type(exc).__name__}: {exc}", t0, ctx)
        except Exception as exc:  # never let one bad evaluation kill the experiment
            detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            return self._record_failure(task, attempt, FailureCategory.UNEXPECTED, detail, t0, ctx)

        self._emit(task, E.EVALUATION_COMPLETED, status="COMPLETED", stage=Stage.COMPLETE.value,
                   message=f"Completed {task.model_name} on {task.dataset_id} in {runtime:.1f}s")
        log.info("evaluation completed in %.2fs", runtime, extra=ctx)
        return TaskOutcome(task, EvaluationStatus.COMPLETED.value, runtime_seconds=runtime)

    def _record_failure(self, task, attempt, category: FailureCategory, detail: str, t0: float, ctx) -> TaskOutcome:
        runtime = time.perf_counter() - t0
        self.repo.evaluations.fail(task.evaluation_id, attempt, category.value, detail, runtime)
        summary = detail.splitlines()[0][:200]
        self._emit(task, E.EVALUATION_FAILED, status="FAILED", message=f"FAILED {task.model_name} on {task.dataset_id}: {summary}",
                   payload={"category": category.value})
        log.warning("evaluation failed [%s]: %s", category.value, summary, extra=ctx)
        return TaskOutcome(task, EvaluationStatus.FAILED.value, category.value, summary, runtime)

    def _fail_best_effort(self, task, attempt, exc: Exception, runtime: float) -> None:
        try:
            self.repo.evaluations.fail(task.evaluation_id, attempt, FailureCategory.PERSISTENCE_ERROR.value, str(exc), runtime)
        except Exception:  # the database itself may be what is failing; stale RUNNING is recovered on resume
            pass
