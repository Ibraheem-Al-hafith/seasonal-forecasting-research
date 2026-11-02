"""Common forecasting interface.

A forecaster receives a :class:`ForecastRequest` (training data, period, horizon, seasonal form)
and nothing else: no test observations, no ground truth.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from ..config.defaults import MIN_TRAIN_CYCLES
from ..domain.contracts import ForecastRequest, ForecastResult, StageCallback
from ..domain.enums import SeasonalForm, Stage
from ..domain.exceptions import (
    DataError,
    InvalidForecastError,
    ModelFitError,
    ResearchError,
)

_NOOP: StageCallback = lambda stage: None  # noqa: E731


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def _fit_forecast(self, req: ForecastRequest, on_stage: StageCallback) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """Return (forecast, selected_parameters, metadata). Must only use `req`."""

    def forecast(self, request: ForecastRequest, on_stage: StageCallback | None = None) -> ForecastResult:
        on_stage = on_stage or _NOOP
        self.validate_request(request)
        train = np.array(request.train, dtype=float, copy=True)
        train.flags.writeable = False  # methods must not mutate the shared training data
        safe_req = ForecastRequest(train, request.seasonal_period, request.horizon, request.seasonal_form)

        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                forecast, params, meta = self._fit_forecast(safe_req, on_stage)
            except ResearchError:
                raise
            except Exception as exc:  # library failures become identifiable model-fit errors
                raise ModelFitError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        runtime = time.perf_counter() - t0

        forecast = np.asarray(forecast, dtype=float).reshape(-1)
        self.validate_forecast(forecast, request.horizon)
        msgs: list[str] = []
        for w in caught:
            text = f"{w.category.__name__}: {w.message}"
            if text not in msgs:
                msgs.append(text[:300])
        return ForecastResult(forecast, params, meta, runtime, msgs[:20])

    # ---- validation helpers ------------------------------------------------------
    @staticmethod
    def validate_request(req: ForecastRequest) -> None:
        y = np.asarray(req.train)
        if y.ndim != 1:
            raise DataError("Training data must be one-dimensional")
        if not np.all(np.isfinite(y)):
            raise DataError("Training data contains non-finite values")
        if req.seasonal_period < 2:
            raise DataError("Seasonal period must be >= 2")
        if req.horizon < 1:
            raise DataError("Forecast horizon must be >= 1")
        if len(y) < MIN_TRAIN_CYCLES * req.seasonal_period:
            raise DataError(
                f"Need at least {MIN_TRAIN_CYCLES} full seasonal cycles "
                f"({MIN_TRAIN_CYCLES * req.seasonal_period} observations); got {len(y)}"
            )
        if req.seasonal_form not in {f.value for f in SeasonalForm}:
            raise DataError(f"Unknown seasonal form: {req.seasonal_form}")
        if req.seasonal_form == "multiplicative" and np.any(y <= 0):
            raise DataError("Multiplicative seasonality requires strictly positive training data")

    @staticmethod
    def validate_forecast(forecast: np.ndarray, horizon: int) -> None:
        if forecast.shape != (horizon,):
            raise InvalidForecastError(f"Forecast length {forecast.shape} != requested horizon {horizon}")
        if not np.all(np.isfinite(forecast)):
            raise InvalidForecastError("Forecast contains non-finite values")


class _Stages:
    """Tiny helper so methods can announce stages tersely."""

    def __init__(self, cb: Callable[[Stage], None]):
        self.cb = cb

    def select(self) -> None:
        self.cb(Stage.SELECT_PARAMETERS)

    def fit(self) -> None:
        self.cb(Stage.FIT_MODEL)

    def predict(self) -> None:
        self.cb(Stage.GENERATE_FORECAST)
