"""Evaluation protocol for one (dataset, method) pair. Pure computation: no persistence."""

from __future__ import annotations

import time

import numpy as np

from ..config.schemas import EvaluationCfg
from ..domain.contracts import EvaluationOutcome, ForecastRequest, StageCallback
from ..domain.enums import Stage
from ..domain.exceptions import DataError
from ..forecasting.base import Forecaster
from .metrics import compute_metrics
from .splitter import split_series


def seasonal_form_for(seasonality_type: str, mode: str) -> str:
    """`matched` hands the method the true seasonal form; otherwise a deliberate override."""
    return seasonality_type if mode == "matched" else mode


class Evaluator:
    def __init__(self, cfg: EvaluationCfg):
        self.cfg = cfg

    def evaluate(
        self,
        observed: np.ndarray,
        seasonality_type: str,
        seasonal_period: int,
        method: Forecaster,
        on_stage: StageCallback | None = None,
    ) -> EvaluationOutcome:
        """Steps 2-9 and 10-11 of the evaluation contract.

        Only `seasonality_type` (to derive the seasonal *form*) and `seasonal_period` are used as
        metadata. Ground truth is not an argument, so it cannot leak.
        """
        stage = on_stage or (lambda s: None)
        t0 = time.perf_counter()
        observed = np.asarray(observed, dtype=float)
        if observed.ndim != 1 or not np.all(np.isfinite(observed)):
            raise DataError("Observed series must be a finite 1-D array")

        stage(Stage.PREPARE_SPLIT)
        split = split_series(observed, self.cfg.train_fraction, seasonal_period, self.cfg.forecast_horizon)
        request = ForecastRequest(
            train=split.train,
            seasonal_period=seasonal_period,
            horizon=split.horizon,
            seasonal_form=seasonal_form_for(seasonality_type, self.cfg.seasonal_mode),
        )
        result = method.forecast(request, on_stage=stage)  # stages: select/fit/forecast

        stage(Stage.CALCULATE_METRICS)
        metrics = compute_metrics(split.test_eval, result.forecast)
        return EvaluationOutcome(
            actual=np.array(split.test_eval, copy=True),
            forecast=result.forecast,
            metrics=metrics,
            parameters=result.parameters,
            metadata={**result.metadata, "seasonal_form": request.seasonal_form, "method_runtime_seconds": result.runtime_seconds},
            warnings=result.warnings,
            train_size=split.train_size,
            test_size=split.test_size,
            horizon=split.horizon,
            runtime_seconds=time.perf_counter() - t0,
        )
