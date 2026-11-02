import numpy as np

from forecasting_research.domain.enums import Stage
from forecasting_research.forecasting.base import Forecaster
from forecasting_research.forecasting.registry import register, unregister


class AlwaysFails(Forecaster):
    name = "always_fails"

    def _fit_forecast(self, req, on_stage):
        raise RuntimeError("synthetic model failure")


class NanForecast(Forecaster):
    name = "nan_forecast"

    def _fit_forecast(self, req, on_stage):
        return np.full(req.horizon, np.nan), {}, {}


class Naive(Forecaster):
    """Seasonal-naive; records every request it receives."""

    name = "spy_naive"
    requests: list = []

    def _fit_forecast(self, req, on_stage):
        type(self).requests.append(req)
        on_stage(Stage.FIT_MODEL)
        m = req.seasonal_period
        return np.resize(req.train[-m:], req.horizon).astype(float), {"kind": "seasonal_naive"}, {}


class Registered:
    """Context manager registering temporary methods."""

    def __init__(self, **classes):
        self.classes = classes

    def __enter__(self):
        for n, c in self.classes.items():
            register(n, lambda cfg, c=c: c())
        return self

    def __exit__(self, *a):
        for n in self.classes:
            unregister(n)
