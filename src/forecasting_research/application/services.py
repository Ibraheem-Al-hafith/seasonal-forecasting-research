"""Shared application services: the single entry point used by both the CLI and the GUI.

No research logic lives in the interfaces; they only call these methods and render results.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis import aggregation, comparison, export, plots
from ..config.schemas import AppConfig, ExecutionCfg
from ..domain.exceptions import ConfigError, DataError, UnknownEntityError
from ..execution.factory import create_strategy, import_worker_modules
from ..experiments.lifecycle import describe_status
from ..experiments.manager import ExperimentManager
from ..forecasting.registry import validate_method_names
from ..monitoring.events import EventBus
from ..monitoring.logging import setup_logging
from ..monitoring.progress import ProgressTracker
from ..persistence.artifacts import ArtifactStore
from ..persistence.backup import IntegrityReport, backup_database, check_workspace
from ..persistence.database import Database
from ..persistence.repositories import Repository

WORKSPACE_DIRS = ["database", "artifacts/datasets", "artifacts/evaluations", "logs/experiments", "exports", "backups"]


class ResearchApp:
    def __init__(self, workspace: str | Path | None = None, config: AppConfig | None = None, console_log_level: str = "WARNING"):
        self.config = config or AppConfig()
        self.workspace = Path(workspace or self.config.workspace.path).expanduser().resolve()
        for d in WORKSPACE_DIRS:
            (self.workspace / d).mkdir(parents=True, exist_ok=True)
        setup_logging(self.workspace, self.config.logging.level, console_log_level)
        self.db = Database(self.workspace / "database" / "research.sqlite")
        self.db.initialize()
        self.store = ArtifactStore(self.workspace)
        self.repo = Repository(self.db)
        self.bus = EventBus()
        self.tracker = ProgressTracker()
        self.bus.subscribe(self.tracker.on_event)
        self.manager = ExperimentManager(
            repo=self.repo, store=self.store, workspace=self.workspace, bus=self.bus,
            strategy_factory=create_strategy)
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    # ---- configuration -------------------------------------------------------------------
    def _cfg(self, cfg: AppConfig | None) -> AppConfig:
        cfg = cfg or self.config
        import_worker_modules(cfg.execution.worker_modules)  # custom methods must be registered first
        validate_method_names(cfg.forecasting.methods)
        return cfg

    # ---- workflows -----------------------------------------------------------------------
    def generate(self, cfg: AppConfig | None = None) -> str:
        """Mode A: generation only. Returns the batch id."""
        return self.manager.generate_batch(self._cfg(cfg))

    def run(self, cfg: AppConfig | None = None) -> str:
        """Mode B: generate datasets, then evaluate them. Returns the experiment id."""
        cfg = self._cfg(cfg)
        batch_id = self.manager.generate_batch(cfg)
        ids = [d.dataset_id for d in self.repo.datasets.list_datasets(batch_id=batch_id)]
        return self.manager.run_on_datasets(cfg, ids, batch_id=batch_id)

    def evaluate(
        self,
        cfg: AppConfig | None = None,
        batch_id: str | None = None,
        dataset_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Mode C: evaluate previously generated datasets."""
        cfg = self._cfg(cfg)
        filters = {k: v for k, v in (filters or {}).items() if v is not None}
        if dataset_ids:
            found = {d.dataset_id for d in self.repo.datasets.list_datasets(dataset_ids=dataset_ids)}
            missing = [d for d in dataset_ids if d not in found]
            if missing:
                raise UnknownEntityError(f"Unknown dataset(s): {missing}")
            ids = list(dict.fromkeys(dataset_ids))
        else:
            ids = [d.dataset_id for d in self.repo.datasets.list_datasets(batch_id=batch_id, **filters)]
        if not ids:
            raise DataError("No datasets match the selection. Generate datasets first (research generate).")
        return self.manager.run_on_datasets(cfg, ids, batch_id=batch_id)

    def resume(self, experiment_id: str, retry_failed: bool = False, retry_categories: list[str] | None = None,
               execution: ExecutionCfg | None = None) -> str:
        """Mode D: resume an interrupted experiment.

        Execution settings (sequential/parallel, workers) come from the CURRENTLY loaded configuration unless
        `execution` is given, so an experiment started sequentially can be resumed in parallel and vice versa.
        """
        execution = execution or self.config.execution
        import_worker_modules(execution.worker_modules)
        return self.manager.resume(experiment_id, retry_failed, set(retry_categories) if retry_categories else None, execution)

    def stop(self) -> None:
        self.manager.request_stop()

    # ---- background execution (GUI) ----------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_background(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Run a workflow in a worker thread; progress is read through `tracker`."""
        if self.is_running:
            return False
        self.last_error = None
        self.tracker.reset()

        def _target() -> None:
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=_target, name="research-worker", daemon=True)
        self._thread.start()
        return True

    # ---- inspection ----------------------------------------------------------------------------
    def list_experiments(self) -> pd.DataFrame:
        rows = []
        for e in self.repo.experiments.list():
            c = self.repo.evaluations.counts(e["experiment_id"])
            rows.append({**e, "status_display": describe_status(e["status"], e["outcome"], c["FAILED"]),
                         "total": c["TOTAL"], "completed": c["COMPLETED"], "failed": c["FAILED"]})
        return pd.DataFrame(rows)

    def experiment_details(self, experiment_id: str) -> dict:
        e = self.repo.experiments.get(experiment_id)
        c = self.repo.evaluations.counts(experiment_id)
        e["counts"] = c
        e["status_display"] = describe_status(e["status"], e["outcome"], c["FAILED"])
        e["percent"] = round((c["COMPLETED"] + c["FAILED"]) / c["TOTAL"] * 100, 1) if c["TOTAL"] else 0.0
        e["n_datasets"] = len(self.repo.experiments.dataset_ids(experiment_id))
        return e

    def evaluations_table(self, experiment_id: str) -> pd.DataFrame:
        self.repo.experiments.get(experiment_id)  # validates id
        rows = self.repo.evaluations.list_for_experiment(experiment_id)
        return pd.DataFrame([{
            "dataset_id": r["dataset_id"], "model": r["model_name"], "evaluation_id": r["evaluation_id"],
            "status": r["status"], "stage": r["current_stage"], "started": r["started_at"],
            "duration_s": round(r["runtime_seconds"], 2) if r["runtime_seconds"] is not None else None,
            "attempts": r["attempts"], "failure": r["failure_category"], "error": r["error_summary"],
        } for r in rows], columns=["dataset_id", "model", "evaluation_id", "status", "stage", "started",
                                    "duration_s", "attempts", "failure", "error"])

    def evaluation_details(self, evaluation_id: str) -> dict:
        ev = self.repo.evaluations.get(evaluation_id)
        ev["metrics"] = {k: {"value": v.value, "status": v.status.value} for k, v in self.repo.evaluations.metrics(evaluation_id).items()}
        ev["attempt_history"] = self.repo.evaluations.attempts(evaluation_id)
        return ev

    def list_datasets(self, **filters: Any) -> pd.DataFrame:
        recs = self.repo.datasets.list_datasets(**{k: v for k, v in filters.items() if v is not None})
        return pd.DataFrame([{
            "dataset_id": d.dataset_id, "batch_id": d.batch_id, "seasonality": d.seasonality_type, "period": d.seasonal_period,
            "trend": d.trend_type, "noise": d.noise_level, "length": d.length, "replication": d.replication, "seed": d.seed,
        } for d in recs])

    def dataset_details(self, dataset_id: str) -> dict:
        d = self.repo.datasets.get_dataset(dataset_id)
        return {**d.__dict__}

    def list_batches(self) -> pd.DataFrame:
        return pd.DataFrame(self.repo.datasets.list_batches())

    # ---- analysis ---------------------------------------------------------------------------------
    def results(self, experiment_ids: list[str] | None = None) -> pd.DataFrame:
        for e in experiment_ids or []:
            self.repo.experiments.get(e)
        return self.repo.evaluations.results_table(experiment_ids)

    def aggregate(self, experiment_ids: list[str] | None = None, group_by: list[str] | str | None = None,
                  metrics: list[str] | None = None) -> pd.DataFrame:
        return aggregation.aggregate_metrics(self.results(experiment_ids), group_by, metrics)

    def best_methods(self, experiment_ids: list[str] | None = None, group_by: list[str] | str | None = None,
                     metrics: list[str] | None = None, stat: str = "mean") -> pd.DataFrame:
        metrics = metrics or ["MAE", "RMSE", "MAPE"]
        agg = self.aggregate(experiment_ids, group_by, None)
        frames = [comparison.best_method_per_group(agg, aggregation.normalize_dims(group_by), m, stat) for m in metrics]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def forecast_figure(self, experiment_id: str, dataset_id: str):
        return plots.plot_actual_vs_forecast(self.repo, self.store, experiment_id, dataset_id)

    def metric_figure(self, experiment_ids: list[str] | None, dimension: str, metric: str = "RMSE"):
        return plots.plot_metric_by_dimension(self.results(experiment_ids), dimension, metric)

    def export_results(self, experiment_id: str, kind: str = "evaluations", group_by: list[str] | str | None = None,
                       metric: str = "RMSE", out: str | Path | None = None) -> Path:
        """kind: evaluations | aggregate | best | forecasts. Writes a CSV into workspace/exports (or `out`)."""
        self.repo.experiments.get(experiment_id)
        if kind == "evaluations":
            df = self.results([experiment_id])
        elif kind == "aggregate":
            df = self.aggregate([experiment_id], group_by)
        elif kind == "best":
            df = self.best_methods([experiment_id], group_by, [metric])
        elif kind == "forecasts":
            df = export.forecasts_long_table(self.repo, self.store, experiment_id)
        else:
            raise ConfigError(f"Unknown export kind {kind!r}: evaluations | aggregate | best | forecasts")
        path = Path(out) if out else self.workspace / "exports" / f"{experiment_id}_{kind}.csv"
        return export.write_csv(df, path)

    # ---- maintenance -----------------------------------------------------------------------------------
    def check_workspace(self) -> IntegrityReport:
        return check_workspace(self.db, self.repo, self.store)

    def backup(self) -> Path:
        return backup_database(self.db.path, self.workspace / "backups")
