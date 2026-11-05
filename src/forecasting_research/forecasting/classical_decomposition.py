"""Method A: classical decomposition with an explicit forecasting procedure.

1. Trend  : centred moving average (m for odd m, 2xm for even m) on TRAIN only.
2. Season : mean detrended value per phase; additive indices sum to 0, multiplicative mean to 1.
3. Extrap.: OLS straight line fitted on the deseasonalised training series.
4. Forecast: T_hat(t) + S[t mod m]  (additive)   or   T_hat(t) * S[t mod m]  (multiplicative).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..config.schemas import ForecastingCfg
from ..domain.contracts import ForecastRequest, StageCallback
from ..domain.enums import Stage
from ..domain.exceptions import DataError, ModelFitError
from .base import Forecaster
from .registry import register


def check_decomposition_input(y: np.ndarray, m: int) -> np.ndarray:
    """Validate (y, m) for moving-average decomposition and return y as a float array.

    `np.convolve` cannot cope with missing values (one NaN contaminates up to m+1 windows), so non-finite
    data are rejected explicitly instead of silently producing a partly-NaN trend.
    """
    if isinstance(m, bool) or not isinstance(m, int | np.integer) or m < 2:
        raise DataError(f"Seasonal period must be an integer >= 2; got {m!r}")
    try:
        arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataError(f"Series must be numeric: {exc}") from exc
    if arr.ndim != 1:
        raise DataError("Series must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise DataError("Series contains NaN or infinite values")
    window = m if m % 2 == 1 else m + 1
    if len(arr) < window:
        raise DataError(f"Series of length {len(arr)} is too short for a centred moving average with period {m}")
    return arr


def centered_moving_average(y: np.ndarray, m: int) -> np.ndarray:
    """Centred MA trend; NaN where the window is incomplete (m//2 points at each end)."""
    y = check_decomposition_input(y, m)
    n = len(y)
    if m % 2 == 1:
        kernel = np.ones(m) / m
    else:
        kernel = np.concatenate(([0.5], np.ones(m - 1), [0.5])) / m
    half = len(kernel) // 2
    valid = np.convolve(y, kernel, mode="valid")
    out = np.full(n, np.nan)
    out[half : n - half] = valid
    return out


MIN_OBS_PER_PHASE_WARNING = 5  # same heuristic as the seasonal-strength reliability flag
_TINY = 1e-12


def seasonal_indices(y: np.ndarray, trend: np.ndarray, m: int, form: str) -> np.ndarray:
    """Per-phase mean of the detrended series, normalised (additive: sum 0; multiplicative: mean 1).

    Guards: multiplicative detrending divides by the trend, so a trend that is zero/near zero is rejected;
    the multiplicative normaliser (mean of the raw indices) must be non-zero.
    """
    if form not in ("additive", "multiplicative"):
        raise DataError(f"Unknown seasonal form: {form}")
    if form == "additive":
        detr = y - trend
    else:
        finite = trend[np.isfinite(trend)]
        scale = float(np.mean(np.abs(y))) or 1.0
        if finite.size == 0 or np.min(np.abs(finite)) <= _TINY * scale:
            raise ModelFitError("Trend is zero or near zero; multiplicative detrending (y / trend) is undefined")
        detr = y / trend
    idx = np.empty(m)
    min_obs = None
    for k in range(m):
        vals = detr[k::m]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            raise ModelFitError(f"No detrended observations for seasonal phase {k}")
        min_obs = len(vals) if min_obs is None else min(min_obs, len(vals))
        idx[k] = vals.mean()
    if min_obs is not None and min_obs < MIN_OBS_PER_PHASE_WARNING:
        warnings.warn(
            f"only {min_obs} detrended observations per seasonal phase (period {m}); seasonal indices are statistically unstable",
            UserWarning, stacklevel=2)
    if form == "additive":
        return idx - idx.mean()
    mean = idx.mean()
    if not np.isfinite(mean) or abs(mean) <= _TINY:
        raise ModelFitError("Mean of raw multiplicative seasonal indices is zero; cannot normalise")
    return idx / mean


def deseasonalise(y: np.ndarray, idx: np.ndarray, form: str) -> np.ndarray:
    s = idx[np.arange(len(y)) % len(idx)]
    return y - s if form == "additive" else y / s


def decompose_train(y: np.ndarray, m: int, form: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (moving-average trend, seasonal indices)."""
    trend = centered_moving_average(y, m)
    if form == "multiplicative" and np.nanmin(trend) <= 0:
        raise ModelFitError("Moving-average trend is non-positive; multiplicative decomposition undefined")
    return trend, seasonal_indices(y, trend, m, form)


class ClassicalDecompositionForecaster(Forecaster):
    name = "classical_decomposition"

    @classmethod
    def from_config(cls, cfg: ForecastingCfg) -> ClassicalDecompositionForecaster:
        return cls()

    def _fit_forecast(self, req: ForecastRequest, on_stage: StageCallback):
        y, m, h, form = req.train, req.seasonal_period, req.horizon, req.seasonal_form
        n = len(y)
        on_stage(Stage.SELECT_PARAMETERS)  # no free hyper-parameters: stage kept for uniform progress
        on_stage(Stage.FIT_MODEL)
        trend, idx = decompose_train(y, m, form)
        adjusted = deseasonalise(y, idx, form)
        slope, intercept = np.polyfit(np.arange(n, dtype=float), adjusted, 1)
        on_stage(Stage.GENERATE_FORECAST)
        t_future = np.arange(n, n + h, dtype=float)
        trend_hat = intercept + slope * t_future
        s_future = idx[t_future.astype(int) % m]
        fc = trend_hat + s_future if form == "additive" else trend_hat * s_future
        params = {
            "seasonal_form": form,
            "trend_estimator": f"centered_moving_average(m={m})",
            "extrapolation": "ols_linear_on_deseasonalised_train",
            "slope": float(slope),
            "intercept": float(intercept),
            "seasonal_indices": [float(v) for v in idx],
        }
        return fc, params, {}


register("classical_decomposition", ClassicalDecompositionForecaster.from_config)
