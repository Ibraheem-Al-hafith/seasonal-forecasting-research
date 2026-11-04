"""Pure, picklable computation of ONE evaluation (no database access, no persistence).

This is the unit of work that can run in the main process (sequential strategy) or in a worker
process (parallel strategy). Everything a worker needs travels in a WorkItem of plain data; everything it
produces comes back in a WorkResult. All database and artifact *writes* stay in the parent process
(single writer), which keeps SQLite and the safe-persistence sequence simple and correct.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from ..config.schemas import EvaluationCfg, ForecastingCfg
from ..domain.contracts import EvaluationOutcome, EvaluationTask
from ..domain.enums import FailureCategory, Stage
from ..domain.exceptions import ArtifactIntegrityError, DataError, ResearchError
from ..evaluation.evaluator import Evaluator
from ..forecasting.registry import create_method
from ..persistence.artifacts import ArtifactStore


@dataclass(frozen=True)
class WorkItem:
    task: EvaluationTask
    workspace: str
    observed_path: str
    observed_sha256: str
    seasonality_type: str
    seasonal_period: int
    length: int
    evaluation_cfg: dict
    forecasting_cfg: dict


@dataclass
class WorkResult:
    evaluation_id: str
    outcome: EvaluationOutcome | None = None
    failure_category: str | None = None
    failure_detail: str | None = None
    compute_seconds: float = 0.0


def failure_result(evaluation_id: str, category: FailureCategory | str, detail: str, seconds: float = 0.0) -> WorkResult:
    cat = category.value if isinstance(category, FailureCategory) else category
    return WorkResult(evaluation_id, None, cat, detail, seconds)


def run_work_item(item: WorkItem, on_stage: Callable[[Stage], None] | None = None) -> WorkResult:
    """Steps 1-11 of the evaluation contract. Never raises for model/data problems: returns a failure result."""
    stage = on_stage or (lambda s: None)
    eid = item.task.evaluation_id
    t0 = time.perf_counter()
    try:
        stage(Stage.LOAD_DATASET)
        try:  # only the OBSERVED artifact is ever read; ground truth is never loaded
            y = ArtifactStore(item.workspace).load_observed(item.observed_path, item.observed_sha256)
        except ArtifactIntegrityError as exc:  # a bad dataset fails its own evaluations, not the whole run
            raise DataError(str(exc)) from exc
        if y.shape != (item.length,):
            raise DataError(f"Dataset {item.task.dataset_id} length {y.shape} does not match metadata ({item.length})")
        method = create_method(item.task.model_name, ForecastingCfg.model_validate(item.forecasting_cfg))
        outcome = Evaluator(EvaluationCfg.model_validate(item.evaluation_cfg)).evaluate(
            y, item.seasonality_type, item.seasonal_period, method, on_stage=stage)
        return WorkResult(eid, outcome=outcome, compute_seconds=time.perf_counter() - t0)
    except ResearchError as exc:
        return failure_result(eid, exc.category, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)
    except Exception as exc:  # never let one bad evaluation take down a worker or the experiment
        return failure_result(eid, FailureCategory.UNEXPECTED,
                              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.perf_counter() - t0)
