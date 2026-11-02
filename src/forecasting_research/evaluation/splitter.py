"""Chronological train/test split and horizon resolution."""

from __future__ import annotations

import math

import numpy as np

from ..config.defaults import MIN_TRAIN_CYCLES
from ..domain.contracts import SplitResult
from ..domain.exceptions import DataError


def resolve_horizon(configured: int | None, seasonal_period: int) -> int:
    """Default = one complete seasonal cycle; an integer override replaces it."""
    return int(configured) if configured else int(seasonal_period)


def split_series(
    y: np.ndarray, train_fraction: float, seasonal_period: int, horizon_override: int | None = None
) -> SplitResult:
    y = np.asarray(y, dtype=float)
    n = len(y)
    train_size = int(math.floor(train_fraction * n))
    test_size = n - train_size
    horizon = resolve_horizon(horizon_override, seasonal_period)
    if train_size < MIN_TRAIN_CYCLES * seasonal_period:
        raise DataError(f"Training size {train_size} is below {MIN_TRAIN_CYCLES} seasonal cycles")
    if horizon > test_size:
        raise DataError(f"Horizon {horizon} exceeds test size {test_size}")
    train = y[:train_size]
    test_full = y[train_size:]
    return SplitResult(train, test_full, test_full[:horizon], train_size, test_size, horizon)
