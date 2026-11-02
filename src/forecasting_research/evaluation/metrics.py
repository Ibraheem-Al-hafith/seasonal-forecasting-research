"""Evaluation metrics (plain NumPy).

MAPE policy: percent scale. If scale = mean(|actual|) is zero, or any |actual| < 1e-3 * scale,
MAPE is *undefined*: value None with status UNDEFINED_NEAR_ZERO. It is never replaced by 0 or inf.
"""

from __future__ import annotations

import numpy as np

from ..domain.contracts import MetricValue
from ..domain.enums import MetricStatus
from ..domain.exceptions import DataError

MAPE_NEAR_ZERO_REL = 1e-3


def _check(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape or a.ndim != 1 or a.size == 0:
        raise DataError(f"Actual and forecast must be equal-length 1-D arrays; got {a.shape} vs {f.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(f))):
        raise DataError("Metrics require finite actual and forecast values")
    return a, f


def mae(actual, forecast) -> float:
    a, f = _check(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def mse(actual, forecast) -> float:
    a, f = _check(actual, forecast)
    return float(np.mean((a - f) ** 2))


def rmse(actual, forecast) -> float:
    return float(np.sqrt(mse(actual, forecast)))


def mape(actual, forecast) -> MetricValue:
    a, f = _check(actual, forecast)
    scale = float(np.mean(np.abs(a)))
    if scale == 0.0 or np.any(np.abs(a) < MAPE_NEAR_ZERO_REL * scale):
        return MetricValue(None, MetricStatus.UNDEFINED_NEAR_ZERO)
    return MetricValue(float(np.mean(np.abs((a - f) / a)) * 100.0), MetricStatus.OK)


def compute_metrics(actual, forecast) -> dict[str, MetricValue]:
    return {
        "MAE": MetricValue(mae(actual, forecast)),
        "MSE": MetricValue(mse(actual, forecast)),
        "RMSE": MetricValue(rmse(actual, forecast)),
        "MAPE": mape(actual, forecast),
    }
