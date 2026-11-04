"""Method B: Holt-Winters triple exponential smoothing (statsmodels).

Seasonal form follows the request; trend (none/add) is chosen by AICc on TRAIN data only.
Initialisation: statsmodels "estimated". No damping, no Box-Cox.
"""

from __future__ import annotations

import math

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..config.schemas import ForecastingCfg, HoltWintersCfg
from ..domain.contracts import ForecastRequest, StageCallback
from ..domain.enums import Stage
from ..domain.exceptions import ModelFitError
from .base import Forecaster
from .registry import register


class HoltWintersForecaster(Forecaster):
    name = "holt_winters"

    def __init__(self, cfg: HoltWintersCfg | None = None):
        self.cfg = cfg or HoltWintersCfg()

    @classmethod
    def from_config(cls, cfg: ForecastingCfg) -> HoltWintersForecaster:
        return cls(cfg.holt_winters)

    def _fit_forecast(self, req: ForecastRequest, on_stage: StageCallback):
        y, m, h = req.train, req.seasonal_period, req.horizon
        on_stage(Stage.SELECT_PARAMETERS)
        best = None
        tried: list[dict] = []
        for trend_opt in self.cfg.trend_options:
            trend = None if trend_opt == "none" else trend_opt
            try:
                model = ExponentialSmoothing(
                    y,
                    trend=trend,
                    seasonal=req.seasonal_form,
                    seasonal_periods=m,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                aicc = float(fit.aicc)
            except Exception as exc:  # candidate-level failure: record and continue
                tried.append({"trend": trend_opt, "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            score = aicc if math.isfinite(aicc) else math.inf
            tried.append({"trend": trend_opt, "aicc": aicc if math.isfinite(aicc) else None})
            if best is None or score < best[0]:
                best = (score, trend_opt, fit)
        if best is None:
            raise ModelFitError(f"Holt-Winters: all candidate configurations failed: {tried}")
        _, trend_opt, fit = best
        on_stage(Stage.FIT_MODEL)  # fitting already done during selection; announce for uniform progress
        on_stage(Stage.GENERATE_FORECAST)
        fc = np.asarray(fit.forecast(h), dtype=float)
        p = fit.params
        params = {
            "trend": trend_opt,
            "seasonal": req.seasonal_form,
            "seasonal_periods": m,
            "initialization_method": "estimated",
            "smoothing_level": _f(p.get("smoothing_level")),
            "smoothing_trend": _f(p.get("smoothing_trend")),
            "smoothing_seasonal": _f(p.get("smoothing_seasonal")),
            "aicc": _f(fit.aicc),
            "sse": _f(fit.sse),
            "selection_criterion": "aicc",
            "candidates": tried,
        }
        return fc, params, {}


def _f(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


register("holt_winters", HoltWintersForecaster.from_config)
