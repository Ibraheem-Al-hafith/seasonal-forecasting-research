"""Plain data contracts exchanged between components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .enums import MetricStatus, Stage

StageCallback = Callable[[Stage], None]


def utcnow() -> str:
    """ISO-8601 UTC timestamp string (the format stored in SQLite)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DatasetSpec:
    """Everything (besides generator params) that identifies one dataset."""

    seasonality_type: str
    seasonal_period: int
    trend_type: str
    noise_level: str
    length: int
    replication: int
    seed: int  # derived, deterministic

    def canonical_key(self) -> str:
        return "|".join(
            [self.seasonality_type, str(self.seasonal_period), self.trend_type, self.noise_level, str(self.length)]
        )


@dataclass
class GroundTruth:
    trend: np.ndarray
    seasonal: np.ndarray
    noise: np.ndarray  # additive noise values OR multiplicative noise factors
    noise_kind: str  # "additive" | "multiplicative_factor"


@dataclass
class GeneratedDataset:
    dataset_id: str
    spec: DatasetSpec
    observed: np.ndarray
    truth: GroundTruth
    metadata: dict[str, Any]


@dataclass
class DatasetRecord:
    """Dataset row as stored in SQLite (no arrays)."""

    dataset_id: str
    batch_id: str
    seasonality_type: str
    seasonal_period: int
    trend_type: str
    noise_level: str
    length: int
    replication: int
    seed: str
    metadata: dict[str, Any]
    observed_path: str
    truth_path: str
    observed_sha256: str
    truth_sha256: str
    generator_version: str
    created_at: str


@dataclass(frozen=True)
class ForecastRequest:
    """The ONLY information a forecasting method ever receives."""

    train: np.ndarray
    seasonal_period: int
    horizon: int
    seasonal_form: str  # "additive" | "multiplicative"


@dataclass
class ForecastResult:
    forecast: np.ndarray
    parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    status: MetricStatus = MetricStatus.OK


@dataclass(frozen=True)
class SplitResult:
    train: np.ndarray
    test_full: np.ndarray
    test_eval: np.ndarray  # first `horizon` test observations
    train_size: int
    test_size: int
    horizon: int


@dataclass
class EvaluationOutcome:
    """Structured result of evaluating one method on one dataset."""

    actual: np.ndarray
    forecast: np.ndarray
    metrics: Mapping[str, MetricValue]
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    warnings: list[str]
    train_size: int
    test_size: int
    horizon: int
    runtime_seconds: float


@dataclass(frozen=True)
class EvaluationTask:
    evaluation_id: str
    experiment_id: str
    dataset_id: str
    model_name: str


@dataclass(frozen=True)
class TaskOutcome:
    task: EvaluationTask
    status: str  # EvaluationStatus value
    failure_category: str | None = None
    error_summary: str | None = None
    runtime_seconds: float = 0.0


@dataclass(frozen=True)
class ProgressEvent:
    """Shared by CLI and GUI. Published by execution components, rendered by interfaces."""

    event_type: str
    experiment_id: str | None = None
    evaluation_id: str | None = None
    dataset_id: str | None = None
    model_name: str | None = None
    stage: str | None = None
    status: str | None = None
    message: str = ""
    timestamp: str = field(default_factory=utcnow)
    payload: dict[str, Any] = field(default_factory=dict)
