"""Compare SARIMA configurations on identical training data (offline study; not part of the experiment pipeline).

Usage:  python scripts/sarima_study.py --out study.csv [--length 200] [--reps 1] [--variants v1,v2,...]

Selection uses TRAINING data only (as in the real pipeline). Test RMSE is reported for information.
Use a generator seed that differs from the experiment's base_seed (default here: 777), otherwise the study would look
at the test windows of the very datasets you will later report on.
`rmse_ratio` = test RMSE / the standard deviation of the generator's noise in the test window (an
oracle-based normalisation that is only possible because the data are synthetic; 1.0 = irreducible error).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from forecasting_research.config.schemas import GenerationCfg, SarimaCfg
from forecasting_research.domain.contracts import ForecastRequest
from forecasting_research.forecasting.sarima import SarimaForecaster
from forecasting_research.generation.generator import DatasetGenerator

VARIANTS: dict[str, dict] = {
    # legacy approximation: strength-threshold D, level-only KPSS, exhaustive grid, AIC
    "legacy_like": dict(seasonal_differencing_strategy="seasonal_strength", kpss_regression="c",
                        search_strategy="full_grid", selection_criterion="aic"),
    "D0_kpss_c": dict(seasonal_differencing_strategy="fixed_zero", kpss_regression="c", search_strategy="full_grid"),
    "D0_auto_full": dict(seasonal_differencing_strategy="fixed_zero", kpss_regression="auto", search_strategy="full_grid"),
    "D1_auto_full": dict(seasonal_differencing_strategy="fixed_one", kpss_regression="auto", search_strategy="full_grid"),
    "D0_auto_two_stage": dict(seasonal_differencing_strategy="fixed_zero", kpss_regression="auto", search_strategy="two_stage"),
    "ocsb_auto_two_stage": dict(seasonal_differencing_strategy="ocsb", kpss_regression="auto", search_strategy="two_stage"),
    # stage-2 trigger study, with D = 1
    "D1_two_stage_lb": dict(seasonal_differencing_strategy="fixed_one", search_strategy="two_stage", stage2_trigger="ljung_box"),
    "D1_two_stage_both": dict(seasonal_differencing_strategy="fixed_one", search_strategy="two_stage", stage2_trigger="boundary_or_ljung_box"),
    "D1_stage1_only": dict(seasonal_differencing_strategy="fixed_one", max_pq_sum=1, max_seasonal_pq_sum=1, stage1_max_pq_sum=1,
                           stage1_max_seasonal_pq_sum=1, search_strategy="two_stage"),
    "D0_two_stage_lb": dict(seasonal_differencing_strategy="fixed_zero", search_strategy="two_stage", stage2_trigger="ljung_box"),
}
BASELINES = ["holt_winters", "classical_decomposition"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sarima_study.csv")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--periods", default="4,7,12")
    ap.add_argument("--noise", default="low,high")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seasonality", default="additive,multiplicative", help="comma list of seasonality types to include")
    ap.add_argument("--base-seed", type=int, default=777,
                    help="generator seed; keep it DIFFERENT from your experiment's base_seed so the study uses independent data")
    ap.add_argument("--baselines", action="store_true", help="also evaluate Holt-Winters and classical decomposition")
    args = ap.parse_args()

    gen_cfg = GenerationCfg(seasonality_types=args.seasonality.split(","), seasonal_periods=[int(x) for x in args.periods.split(",")], dataset_lengths=[args.length],
                            noise_levels=args.noise.split(","), replications=args.reps,
                            base_seed=args.base_seed)
    gen = DatasetGenerator(gen_cfg)
    rows = []
    for spec in gen.iter_specs():
        ds = gen.generate(spec)
        n_train = int(0.8 * spec.length)
        m = spec.seasonal_period
        train, test = ds.observed[:n_train], ds.observed[n_train:n_train + m]
        # noise sd in the test window (oracle information, used ONLY for the normalised error below)
        noise_sd = float(np.std(ds.truth.noise[n_train:n_train + m] if ds.truth.noise_kind == "additive"
                                else (ds.truth.noise[n_train:n_train + m] - 1) * ds.truth.trend[n_train:n_train + m]))
        if args.baselines:
            from forecasting_research.config.schemas import ForecastingCfg
            from forecasting_research.forecasting.registry import create_method
            for b in BASELINES:
                r = create_method(b, ForecastingCfg()).forecast(ForecastRequest(train, m, m, spec.seasonality_type))
                rm = float(np.sqrt(np.mean((r.forecast - test) ** 2)))
                rows.append(dict(dataset=ds.dataset_id, seasonality=spec.seasonality_type, period=m, trend=spec.trend_type,
                                 noise=spec.noise_level, variant=b, rmse=rm, rmse_ratio=rm / noise_sd, seconds=0.0, status="ok"))
        for name in [v for v in args.variants.split(",") if v]:
            cfg = SarimaCfg(**VARIANTS[name])
            t0 = time.perf_counter()
            try:
                res = SarimaForecaster(cfg).forecast(ForecastRequest(train, m, m, spec.seasonality_type))
                rmse = float(np.sqrt(np.mean((res.forecast - test) ** 2)))
                p = res.parameters
                rows.append(dict(dataset=ds.dataset_id, seasonality=spec.seasonality_type, period=m, trend=spec.trend_type,
                                 noise=spec.noise_level, variant=name, rmse=rmse, rmse_ratio=rmse / noise_sd,
                                 seconds=time.perf_counter() - t0, fits=p.get("n_candidates_considered", p.get("n_candidates_planned")),
                                 order=str(p["order"]), seasonal_order=str(p["seasonal_order"]), D=p["seasonal_order"][1], d=p["order"][1],
                                 trend_spec=p.get("deterministic_trend", p.get("trend")), stage2=p.get("expansion", {}).get("triggered"),
                                 converged_pct=100 * p.get("n_selectable", 0) / max(p.get("n_candidates_fitted", 1), 1), status="ok"))
            except Exception as exc:
                rows.append(dict(dataset=ds.dataset_id, seasonality=spec.seasonality_type, period=m, trend=spec.trend_type,
                                 noise=spec.noise_level, variant=name, seconds=time.perf_counter() - t0,
                                 status=f"FAILED {type(exc).__name__}"))
        print(f"{spec.canonical_key():45s} done", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
