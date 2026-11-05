"""Method C: SARIMA with bounded, two-stage automatic order selection (statsmodels SARIMAX), method version 2.

Components (see forecasting/sarima_components/):
  differencing.py : seasonal strength diagnostic, choice of D and d, deterministic trend
  candidates.py   : candidate generation (full grid / stage 1 / stage 2)
  fitting.py      : fit one candidate, strict convergence handling, Ljung-Box diagnostic
  search.py       : stages, soft time budget, expansion decision

All decisions use TRAINING data only.
"""

from __future__ import annotations

import numpy as np

from ..config.schemas import SARIMA_METHOD_VERSION, ForecastingCfg, SarimaCfg
from ..domain.contracts import ForecastRequest, StageCallback
from ..domain.enums import Stage
from ..domain.exceptions import DataError, InvalidForecastError
from .base import Forecaster
from .registry import register
from .sarima_components.candidates import full_grid
from .sarima_components.differencing import (
    choose_regular_differencing,
    choose_seasonal_differencing,
    require_strategy_dependencies,
)
from .sarima_components.search import run_search

# Largest magnitude (relative to the training range) a back-transformed forecast may have before it is rejected.
_MAX_REL_MAGNITUDE = 1e6


class SarimaForecaster(Forecaster):
    name = "sarima"

    def __init__(self, cfg: SarimaCfg | None = None):
        self.cfg = cfg or SarimaCfg()
        require_strategy_dependencies(self.cfg)  # fail at construction, not mid-experiment

    @classmethod
    def from_config(cls, cfg: ForecastingCfg) -> SarimaForecaster:
        return cls(cfg.sarima)

    # ---- optional log transformation, validated in both directions -------------------------------------
    @staticmethod
    def _transform(train: np.ndarray, use_log: bool) -> np.ndarray:
        if not use_log:
            return np.asarray(train, dtype=float)
        if not np.all(np.isfinite(train)) or np.any(train <= 0):
            raise DataError("log transform requires strictly positive, finite training data")
        return np.log(train)

    @staticmethod
    def _inverse(fc: np.ndarray, use_log: bool, train: np.ndarray) -> np.ndarray:
        if not use_log:
            return fc
        if not np.all(np.isfinite(fc)):
            raise InvalidForecastError("non-finite forecast in the log domain")
        with np.errstate(over="ignore"):
            out = np.exp(fc)
        span = float(np.ptp(train)) or float(np.max(np.abs(train))) or 1.0
        if not np.all(np.isfinite(out)) or np.max(np.abs(out)) > _MAX_REL_MAGNITUDE * span:
            raise InvalidForecastError("inverse log transform overflowed (forecast is not numerically meaningful)")
        return out

    def _fit_forecast(self, req: ForecastRequest, on_stage: StageCallback):
        cfg, m, h = self.cfg, req.seasonal_period, req.horizon
        use_log = cfg.log_transform_multiplicative and req.seasonal_form == "multiplicative"
        y = self._transform(req.train, use_log)

        on_stage(Stage.SELECT_PARAMETERS)
        D, d_seasonal = choose_seasonal_differencing(y, m, "additive" if use_log else req.seasonal_form, cfg)
        d, trend, d_regular = choose_regular_differencing(y, m, D, cfg)
        outcome = run_search(y, d, D, m, trend, cfg)
        best = outcome.best

        on_stage(Stage.FIT_MODEL)  # the selected model was already fitted during the search
        on_stage(Stage.GENERATE_FORECAST)
        fc = self._inverse(np.asarray(outcome.best_result.forecast(h), dtype=float), use_log, np.asarray(req.train, dtype=float))

        params = {
            "method_version": SARIMA_METHOD_VERSION,
            "order": list(best.order),
            "seasonal_order": list(best.seasonal_order),
            "trend": trend,
            "deterministic_trend": d_regular["deterministic_trend"],
            "selection_criterion": cfg.selection_criterion,
            "criterion_value": best.score,
            "aic": best.aic, "aicc": best.aicc, "bic": best.bic,
            "log_transform": use_log,
            "differencing": {"D": D, "d": d, "seasonal": d_seasonal, "regular": d_regular},
            "search_strategy": cfg.search_strategy,
            "search_limits": {
                "max_pq_sum": cfg.max_pq_sum, "max_seasonal_pq_sum": cfg.max_seasonal_pq_sum,
                "stage1_max_pq_sum": cfg.stage1_pq(), "stage1_max_seasonal_pq_sum": cfg.stage1_spq(),
                "max_iterations": cfg.max_iterations, "time_budget_seconds": cfg.time_budget_seconds,
            },
            "n_candidates_planned": len(full_grid(cfg)),
            "n_candidates_considered": len(outcome.records),
            "search_truncated": outcome.truncated,
            "stopped_reason": outcome.stopped_reason,
            "search_seconds": round(outcome.seconds, 3),
            "stages": outcome.stages,
            "expansion": outcome.expansion,
            **outcome.counts(),
            "best_stage": best.stage,
            "candidates": [r.to_dict() for r in outcome.records],
        }
        return fc, params, {}


register("sarima", SarimaForecaster.from_config)
