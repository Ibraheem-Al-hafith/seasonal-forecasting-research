"""Resume policy: which evaluations to skip, retry or re-run."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..domain.contracts import EvaluationTask
from ..domain.enums import FailureCategory
from ..domain.exceptions import PersistenceError
from ..persistence.artifacts import ArtifactStore
from ..persistence.repositories import Repository

log = logging.getLogger(__name__)
REQUIRED_METRICS = {"MAE", "MSE", "RMSE", "MAPE"}
# Infrastructure failures are not model failures: they are retried automatically on resume.
AUTO_RETRY_CATEGORIES = {FailureCategory.PERSISTENCE_ERROR.value}


@dataclass
class ResumePlan:
    tasks: list[EvaluationTask] = field(default_factory=list)
    skipped_completed: int = 0
    invalid_completed: list[str] = field(default_factory=list)  # COMPLETED in DB but artifacts bad
    retried_failed: int = 0
    permanent_failed: int = 0
    interrupted: int = 0
    pending: int = 0


def plan_resume(
    repo: Repository,
    store: ArtifactStore,
    experiment_id: str,
    retry_failed: bool = False,
    retry_categories: set[str] | None = None,
) -> ResumePlan:
    """Decide what to run. Valid COMPLETED evaluations are skipped; everything else is classified.

    * COMPLETED + verified artifact + all metrics -> skip
    * COMPLETED but artifact missing/corrupt      -> re-run (never trusted on DB status alone)
    * RUNNING (stale) / INTERRUPTED / PENDING     -> run
    * FAILED                                      -> run only if retry requested (or infrastructure failure)
    """
    repo.evaluations.mark_running_interrupted(experiment_id)  # stale RUNNING rows from a crash
    plan = ResumePlan()
    for row in repo.evaluations.list_for_experiment(experiment_id):
        status = row["status"]
        task = EvaluationTask(row["evaluation_id"], experiment_id, row["dataset_id"], row["model_name"])
        if status == "COMPLETED":
            if _completed_is_valid(repo, store, row):
                plan.skipped_completed += 1
            else:
                plan.invalid_completed.append(row["evaluation_id"])
                plan.tasks.append(task)
                log.warning("completed evaluation %s has missing/corrupt results; will re-run", row["evaluation_id"])
        elif status == "FAILED":
            cat = row["failure_category"]
            wanted = retry_failed and (not retry_categories or cat in retry_categories)
            if wanted or cat in AUTO_RETRY_CATEGORIES:
                plan.retried_failed += 1
                plan.tasks.append(task)
            else:
                plan.permanent_failed += 1
        elif status == "INTERRUPTED":
            plan.interrupted += 1
            plan.tasks.append(task)
        else:  # PENDING
            plan.pending += 1
            plan.tasks.append(task)
    return plan


def _completed_is_valid(repo: Repository, store: ArtifactStore, row: dict) -> bool:
    if not row.get("artifact_path") or not row.get("artifact_sha256"):
        return False
    try:
        store.verify(row["artifact_path"], row["artifact_sha256"])
    except PersistenceError:
        return False
    return REQUIRED_METRICS <= set(repo.evaluations.metrics(row["evaluation_id"]))
