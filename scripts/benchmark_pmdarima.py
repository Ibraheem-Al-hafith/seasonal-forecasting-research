"""Benchmark the custom statsmodels search against pmdarima.auto_arima on identical training data.

Requires:  pip install pmdarima
Usage:     python scripts/benchmark_pmdarima.py --out bench.csv [--length 200] [--periods 4,7,12] [--base-seed 777]

Two comparisons per dataset:
  fixed_d_D : d and D are FIXED to the values the custom differencing logic chose, so only the ORDER SEARCH differs
  auto_d_D  : each side chooses its own differencing (custom: configured strategy; pmdarima: KPSS d + OCSB D)
Search spaces differ slightly: pmdarima's stepwise search is bounded by max_p/q/P/Q (here 2/2/1/1, i.e. a superset of
p+q<=2, P+Q<=1) whereas the custom grid imposes the sum limits. Treat results as a benchmark, not a proof.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from forecasting_research.config.schemas import GenerationCfg, SarimaCfg
from forecasting_research.domain.contracts import ForecastRequest
from forecasting_research.forecasting.sarima import SarimaForecaster
from forecasting_research.generation.generator import DatasetGenerator


def pm_fit(train, m, h, d, D, crit="aicc"):
    import pmdarima as pm

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(train, seasonal=True, m=m, d=d, D=D, start_p=0, max_p=2, start_q=0, max_q=2, start_P=0, max_P=2,
                              start_Q=0, max_Q=2, information_criterion=crit, stepwise=True, error_action="ignore",
                              suppress_warnings=True, test="kpss", seasonal_test="ocsb", max_d=1, max_D=1, maxiter=100)
        fc = model.predict(n_periods=h)
    return np.asarray(fc, dtype=float), model, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_pmdarima.csv")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--periods", default="4,7,12")
    ap.add_argument("--noise", default="low,high")
    ap.add_argument("--base-seed", type=int, default=777)
    args = ap.parse_args()
    gen = DatasetGenerator(GenerationCfg(seasonal_periods=[int(x) for x in args.periods.split(",")], dataset_lengths=[args.length],
                                         noise_levels=args.noise.split(","), replications=1, base_seed=args.base_seed))
    rows = []
    for spec in gen.iter_specs():
        ds = gen.generate(spec)
        n_train, m = int(0.8 * spec.length), spec.seasonal_period
        train, test = ds.observed[:n_train], ds.observed[n_train:n_train + m]
        cfg = SarimaCfg()
        t0 = time.perf_counter()
        res = SarimaForecaster(cfg).forecast(ForecastRequest(train, m, m, spec.seasonality_type))
        sec = time.perf_counter() - t0
        p = res.parameters
        d, D = p["order"][1], p["seasonal_order"][1]

        ctx = dict(dataset=ds.dataset_id, seasonality=spec.seasonality_type, period=m, trend=spec.trend_type, noise=spec.noise_level)

        def rec(variant, mode, fc, seconds, order, sorder, fits=None, crit=None, ctx=ctx, train=train, test=test):
            rows.append(dict(ctx, comparison=mode, implementation=variant, seconds=seconds,
                             rmse=float(np.sqrt(np.mean((fc - test) ** 2))),
                             extreme=bool(np.any(np.abs(fc - train.mean()) > 5 * np.ptp(train))), order=str(tuple(order)),
                             seasonal_order=str(tuple(sorder)), fits=fits, aicc=crit, finite=bool(np.all(np.isfinite(fc)))))

        rec("custom_two_stage", "fixed_d_D", res.forecast, sec, p["order"], p["seasonal_order"], p["n_candidates_considered"], p["aicc"])
        try:
            fc, mod, s = pm_fit(train, m, m, d, D)
            rec("pmdarima_stepwise", "fixed_d_D", fc, s, mod.order, mod.seasonal_order, None, float(mod.aicc()))
        except Exception as exc:
            print("pmdarima failed:", exc)
        try:
            fc, mod, s = pm_fit(train, m, m, None, None)
            rec("pmdarima_stepwise", "auto_d_D", fc, s, mod.order, mod.seasonal_order, None, float(mod.aicc()))
        except Exception as exc:
            print("pmdarima auto failed:", exc)
        rec("custom_two_stage", "auto_d_D", res.forecast, sec, p["order"], p["seasonal_order"], p["n_candidates_considered"], p["aicc"])
        print(spec.canonical_key(), "done", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
