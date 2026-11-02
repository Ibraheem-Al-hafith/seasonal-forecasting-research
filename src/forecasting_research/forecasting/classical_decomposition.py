"""Method A: classical decomposition with an explicit forecasting procedure.

1. Trend  : centred moving average (m for odd m, 2xm for even m) on TRAIN only.
2. Season : mean detrended value per phase; additive indices sum to 0, multiplicative mean to 1.
3. Extrap.: OLS straight line fitted on the deseasonalised training series.
4. Forecast: T_hat(t) + S[t mod m]  (additive)   or   T_hat(t) * S[t mod m]  (multiplicative).
"""

from __future__ import annotations

import numpy as np

from ..config.schemas import ForecastingCfg
from ..domain.contracts import ForecastRequest, StageCallback
from ..domain.enums import Stage
from ..domain.exceptions import ModelFitError
from .base import Forecaster
from .registry import register


def centered_moving_average(y: np.ndarray, m: int) -> np.ndarray:
    """Centred MA trend; NaN where the window is incomplete (m//2 points at each end)."""
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


def seasonal_indices(y: np.ndarray, trend: np.ndarray, m: int, form: str) -> np.ndarray:
    detr = y - trend if form == "additive" else y / trend
    idx = np.empty(m)
    for k in range(m):
        vals = detr[k::m]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            raise ModelFitError(f"No detrended observations for seasonal phase {k}")
        idx[k] = vals.mean()
    return idx - idx.mean() if form == "additive" else idx / idx.mean()


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
