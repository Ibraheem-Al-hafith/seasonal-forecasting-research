"""Enumerations shared across the system (str-valued so they serialise cleanly)."""

from __future__ import annotations

from enum import Enum


class SeasonalityType(str, Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class TrendType(str, Enum):
    NONE = "none"
    LINEAR = "linear"


class NoiseLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExperimentStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ExperimentOutcome(str, Enum):
    """Derived, stored summary so partial runs are never shown as plain success."""

    NONE = "NONE"  # not yet finished
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ALL_FAILED = "ALL_FAILED"
    ABORTED = "ABORTED"  # persistence failure
    INCOMPLETE = "INCOMPLETE"  # interrupted with work remaining


class EvaluationStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class BatchStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FailureCategory(str, Enum):
    DATA_ERROR = "DATA_ERROR"
    MODEL_FIT_ERROR = "MODEL_FIT_ERROR"
    INVALID_FORECAST = "INVALID_FORECAST"
    TIMEOUT = "TIMEOUT"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"
    UNEXPECTED = "UNEXPECTED"


class Stage(str, Enum):
    """Stages of one model evaluation (Level C progress)."""

    LOAD_DATASET = "load_dataset"
    PREPARE_SPLIT = "prepare_split"
    SELECT_PARAMETERS = "select_parameters"
    FIT_MODEL = "fit_model"
    GENERATE_FORECAST = "generate_forecast"
    CALCULATE_METRICS = "calculate_metrics"
    PERSIST_RESULTS = "persist_results"
    COMPLETE = "complete"


class MetricStatus(str, Enum):
    OK = "OK"
    UNDEFINED_NEAR_ZERO = "UNDEFINED_NEAR_ZERO"


class SeasonalForm(str, Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
