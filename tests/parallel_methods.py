"""Test-only forecasting methods, registered at import time.

Listed under `execution.worker_modules` so that spawned worker processes (which start with a fresh
interpreter) register them too.
"""

import multiprocessing as mp
import os

import numpy as np

from forecasting_research.forecasting.base import Forecaster
from forecasting_research.forecasting.registry import register


class ParAlwaysFails(Forecaster):
    name = "par_always_fails"

    def _fit_forecast(self, req, on_stage):
        raise RuntimeError("synthetic model failure")


class ParNan(Forecaster):
    name = "par_nan"

    def _fit_forecast(self, req, on_stage):
        return np.full(req.horizon, np.nan), {}, {}


class ParCrash(Forecaster):
    """Kills its own process, simulating a segfault / OOM kill. Only ever in a worker, never in pytest."""

    name = "par_crash"

    def _fit_forecast(self, req, on_stage):
        if mp.parent_process() is not None:
            os._exit(13)
        raise RuntimeError("par_crash is only lethal inside a worker process")


register("par_always_fails", lambda cfg: ParAlwaysFails())
register("par_nan", lambda cfg: ParNan())
register("par_crash", lambda cfg: ParCrash())
