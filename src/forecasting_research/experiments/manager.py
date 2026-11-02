"""Experiment manager / orchestrator.

Coordinates generation, planning, execution (through the ExecutionStrategy), lifecycle and recovery.
It implements none of the maths, model fitting, metric or storage logic itself.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..config.loader import config_from_dict
from ..config.schemas import AppConfig
from ..domain.contracts import DatasetRecord, ProgressEvent, utcnow
from ..domain.enums import ExperimentOutcome, ExperimentStatus
from ..domain.exceptions import ArtifactIntegrityError, DataError, PersistenceError, StopRequested
from ..execution.base import ExecutionStrategy
from ..forecasting.registry import validate_method_names
from ..generation.generator import DatasetGenerator
from ..monitoring import events as E
from ..monitoring.events import DURABLE_EVENTS, EventBus
from ..monitoring.logging import experiment_log
from ..persistence.artifacts import ArtifactStore
from ..persistence.repositories import Repository
from .environment import environment_info
from .lifecycle import derive_experiment_status
from .planner import build_plan, make_batch_id, make_experiment_id
from .recovery import plan_resume
from .task_runner import TaskRunner

log = logging.getLogger(__name__)


class ExperimentManager:
    def __init__(self, *, repo: Repository, store: ArtifactStore, workspace: Path, bus: EventBus, strategy: ExecutionStrategy):
        self.repo, self.store, self.workspace, self.bus, self.strategy = repo, store, Path(workspace), bus, strategy
        self.stop_event = threading.Event()
        bus.subscribe(self._persist_event)

    # ---- control -----------------------------------------------------------------------
    def request_stop(self) -> None:
        """Ask the running operation to stop after the current evaluation / dataset."""
        self.stop_event.set()

    def _persist_event(self, ev: ProgressEvent) -> None:
        if ev.event_type in DURABLE_EVENTS and ev.experiment_id:
            try:
                self.repo.events.add(ev)
            except PersistenceError:
                log.exception("could not persist progress event")  # state tables remain authoritative

    # ---- Mode A: generation only -------------------------------------------------------
    def generate_batch(self, cfg: AppConfig, name: str | None = None) -> str:
        """Generate every dataset of the design. Re-running is idempotent (existing datasets are verified)."""
        self.stop_event.clear()
        gen = DatasetGenerator(cfg.generation)
        batch_id = make_batch_id()
        total = cfg.generation.dataset_count()
        self.repo.datasets.create_batch(
            batch_id, name or cfg.experiment.name,
            {"generation": cfg.generation.model_dump(mode="json"), "environment": environment_info()})
        self.bus.publish(ProgressEvent(E.GENERATION_STARTED, message=f"Generating {total} datasets (batch {batch_id})",
                                       payload={"total": total, "batch_id": batch_id}))
        step = max(1, total // 100)
        try:
            for i, spec in enumerate(gen.iter_specs(), 1):
                if self.stop_event.is_set():
                    raise StopRequested(f"Generation stopped after {i - 1}/{total} datasets")
                self._persist_dataset(gen.generate(spec), batch_id)
                if i % step == 0 or i == total:
                    self.bus.publish(ProgressEvent(E.GENERATION_PROGRESS, payload={"done": i, "total": total},
                                                   message=f"Generated {i}/{total}" if i == total else ""))
            self.repo.datasets.finish_batch(batch_id, "COMPLETED")
        except BaseException:
            try:
                self.repo.datasets.finish_batch(batch_id, "FAILED")
            except PersistenceError:
                pass
            raise
        self.bus.publish(ProgressEvent(E.GENERATION_FINISHED, message=f"Batch {batch_id} complete", payload={"batch_id": batch_id}))
        return batch_id

    def _persist_dataset(self, ds, batch_id: str) -> None:
        """Artifacts first (atomic, verified), then the DB reference; never the other way round."""
        if self.repo.datasets.dataset_exists(ds.dataset_id):
            rec = self.repo.datasets.get_dataset(ds.dataset_id)
            observed = self.store.load_observed(rec.observed_path, rec.observed_sha256)
            truth = self.store.load_truth(rec.truth_path, rec.truth_sha256)
            same = (observed.shape == ds.observed.shape and (observed == ds.observed).all()
                    and (truth.trend == ds.truth.trend).all() and (truth.seasonal == ds.truth.seasonal).all()
                    and (truth.noise == ds.truth.noise).all())
            if not same:
                raise ArtifactIntegrityError(f"Dataset {ds.dataset_id} exists but differs from regenerated content")
            self.repo.datasets.link_to_batch(batch_id, ds.dataset_id)
            return
        o_rel, t_rel, o_sha, t_sha = self.store.save_dataset(ds)
        m = ds.metadata
        rec = DatasetRecord(
            dataset_id=ds.dataset_id, batch_id=batch_id, seasonality_type=ds.spec.seasonality_type,
            seasonal_period=ds.spec.seasonal_period, trend_type=ds.spec.trend_type, noise_level=ds.spec.noise_level,
            length=ds.spec.length, replication=ds.spec.replication, seed=str(ds.spec.seed), metadata=m,
            observed_path=o_rel, truth_path=t_rel, observed_sha256=o_sha, truth_sha256=t_sha,
            generator_version=m["generator_version"], created_at=utcnow())
        self.repo.datasets.add_dataset(rec, batch_id)

    # ---- Modes B/C: run evaluations on datasets -----------------------------------------
    def run_on_datasets(self, cfg: AppConfig, dataset_ids: list[str], batch_id: str | None = None) -> str:
        validate_method_names(cfg.forecasting.methods)
        if not dataset_ids:
            raise DataError("No datasets selected for the experiment")
        experiment_id = make_experiment_id()
        self.repo.experiments.create_experiment(
            experiment_id, cfg.experiment.name, batch_id, cfg.model_dump(mode="json"), environment_info(), dataset_ids)
        tasks = build_plan(experiment_id, dataset_ids, cfg.forecasting.methods)
        self.repo.evaluations.create_pending(tasks)
        self.bus.publish(ProgressEvent(E.EXPERIMENT_CREATED, experiment_id=experiment_id,
                                       message=f"Experiment {experiment_id} created: {len(tasks)} evaluations"))
        self._execute(experiment_id, cfg, tasks)
        return experiment_id

    # ---- Mode D: resume ------------------------------------------------------------------
    def resume(self, experiment_id: str, retry_failed: bool = False, retry_categories: set[str] | None = None) -> str:
        exp = self.repo.experiments.get(experiment_id)
        cfg = config_from_dict(exp["config"])
        validate_method_names(cfg.forecasting.methods)
        plan = plan_resume(self.repo, self.store, experiment_id, retry_failed, retry_categories)
        log.info("resume plan: %d to run, %d skipped, %d invalid-completed, %d retried-failed, %d permanent failures",
                 len(plan.tasks), plan.skipped_completed, len(plan.invalid_completed), plan.retried_failed,
                 plan.permanent_failed, extra={"experiment_id": experiment_id})
        self._execute(experiment_id, cfg, plan.tasks, resumed=True)
        return experiment_id

    # ---- core loop -----------------------------------------------------------------------
    def _execute(self, experiment_id: str, cfg: AppConfig, tasks: list, resumed: bool = False) -> None:
        self.stop_event.clear()
        with experiment_log(self.workspace, experiment_id, cfg.logging.level):
            run_ids = {t.evaluation_id for t in tasks}
            rows = self.repo.evaluations.list_for_experiment(experiment_id)
            done = sum(1 for r in rows if r["status"] == "COMPLETED" and r["evaluation_id"] not in run_ids)
            failed = sum(1 for r in rows if r["status"] == "FAILED" and r["evaluation_id"] not in run_ids)
            self.repo.experiments.set_status(experiment_id, ExperimentStatus.RUNNING.value, ExperimentOutcome.NONE.value, mark_started=True)
            self.bus.publish(ProgressEvent(E.EXPERIMENT_STARTED, experiment_id=experiment_id,
                                           message=f"{'Resumed' if resumed else 'Started'} {experiment_id}: {len(tasks)} evaluations to run"))
            self.bus.publish(ProgressEvent(E.PLAN_BUILT, experiment_id=experiment_id,
                                           payload={"total": len(rows), "completed": done, "failed": failed}))
            runner = TaskRunner(self.repo, self.store, cfg, self.bus)
            interrupted = aborted = False
            error: PersistenceError | None = None
            try:
                self.strategy.execute(tasks, runner.run, self.stop_event.is_set)
            except KeyboardInterrupt:
                interrupted = True
            except PersistenceError as exc:
                aborted, error = True, exc
                log.critical("persistence failure; experiment aborted: %s", exc, extra={"experiment_id": experiment_id})
            interrupted = interrupted or self.stop_event.is_set()
            self._finalize(experiment_id, aborted, interrupted)
            if error is not None:
                raise error

    def _finalize(self, experiment_id: str, aborted: bool, interrupted: bool) -> None:
        status, outcome = ExperimentStatus.INTERRUPTED, ExperimentOutcome.INCOMPLETE
        try:
            if aborted or interrupted:
                self.repo.evaluations.mark_running_interrupted(experiment_id)
            counts = self.repo.evaluations.counts(experiment_id)
            status, outcome = derive_experiment_status(counts, aborted=aborted, interrupted=interrupted)
            self.repo.experiments.set_status(
                experiment_id, status.value, outcome.value,
                mark_completed=status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED))
            msg = f"Experiment {experiment_id} {status.value} ({outcome.value}): {counts['COMPLETED']} ok, {counts['FAILED']} failed"
        except PersistenceError:
            log.critical("could not record final experiment status; resume will repair it", extra={"experiment_id": experiment_id})
            msg = f"Experiment {experiment_id} stopped; final state could not be recorded"
            if aborted:
                status, outcome = ExperimentStatus.FAILED, ExperimentOutcome.ABORTED
        etype = E.EXPERIMENT_INTERRUPTED if status == ExperimentStatus.INTERRUPTED else E.EXPERIMENT_FINISHED
        self.bus.publish(ProgressEvent(etype, experiment_id=experiment_id, message=msg,
                                       payload={"status": status.value, "outcome": outcome.value}))
        log.info(msg, extra={"experiment_id": experiment_id})
