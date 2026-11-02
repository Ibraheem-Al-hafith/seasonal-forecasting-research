"""Exception hierarchy. Each evaluation-level error maps to a FailureCategory."""

from __future__ import annotations

from .enums import FailureCategory


class ResearchError(Exception):
    """Base class for all expected, human-readable errors."""

    category: FailureCategory = FailureCategory.UNEXPECTED


class ConfigError(ResearchError):
    """Invalid configuration (rejected before execution)."""

    category = FailureCategory.DATA_ERROR


class DataError(ResearchError):
    """Invalid, missing or corrupt input data."""

    category = FailureCategory.DATA_ERROR


class ModelFitError(ResearchError):
    """A forecasting method could not be fitted."""

    category = FailureCategory.MODEL_FIT_ERROR


class InvalidForecastError(ResearchError):
    """Forecast has the wrong shape or contains non-finite values."""

    category = FailureCategory.INVALID_FORECAST


class EvaluationTimeout(ResearchError):
    """Soft time budget exhausted before any usable model was found."""

    category = FailureCategory.TIMEOUT


class PersistenceError(ResearchError):
    """Database or artifact persistence failed. Execution must stop safely."""

    category = FailureCategory.PERSISTENCE_ERROR


class ArtifactIntegrityError(PersistenceError):
    """An artifact is missing, unreadable, corrupt or would be overwritten."""


class UnknownEntityError(ResearchError):
    """Requested experiment / dataset / evaluation does not exist."""

    category = FailureCategory.DATA_ERROR


class StopRequested(ResearchError):
    """Raised when the user asks a long-running operation to stop (not a failure)."""
