"""Method C: SARIMA with bounded automatic order selection (statsmodels SARIMAX).

Everything below uses TRAIN data only:
  D : seasonal differencing if seasonal strength (Wang-Smith-Hyndman) > threshold
  d : 1 if KPSS rejects stationarity (after seasonal differencing), else 0
  (p,q) with p+q <= max_pq_sum  x  (P,Q) with P+Q <= max_seasonal_pq_sum  -> grid search by AIC/BIC
A constant is included when d + D <= 1 (level when 0, drift when 1).
Non-converged candidates are skipped; the search stops early once the soft time budget is spent.
"""

from __future__ import annotations

import math
import time

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

from ..config.schemas import ForecastingCfg, SarimaCfg
from ..domain.contracts import ForecastRequest, StageCallback
from ..domain.enums import Stage
from ..domain.exceptions import EvaluationTimeout, ModelFitError
from .base import Forecaster
from .classical_decomposition import centered_moving_average, seasonal_indices
from .registry import register


def seasonal_strength(y: np.ndarray, m: int) -> float:
    """F_s = max(0, 1 - Var(remainder) / Var(seasonal + remainder)), additive decomposition."""
    trend = centered_moving_average(y, m)
    idx = seasonal_indices(y, trend, m, "additive")
    seasonal = idx[np.arange(len(y)) % m]
    ok = np.isfinite(trend)
    remainder = (y - trend - seasonal)[ok]
    denom = np.var(seasonal[ok] + remainder)
    if denom <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(remainder) / denom))


def choose_differencing(y: np.ndarray, m: int, cfg: SarimaCfg) -> tuple[int, int, dict]:
    strength = seasonal_strength(y, m)
    D = 1 if strength > cfg.seasonal_strength_threshold else 0
    z = y[m:] - y[:-m] if D == 1 else y
    d, pval = 0, None
    try:
        with np.errstate(all="ignore"):
            _, pval, *_ = kpss(z, regression="c", nlags="auto")
        pval = float(pval)
        d = 1 if pval < cfg.kpss_alpha else 0
    except Exception:  # KPSS failure -> conservative: no regular differencing
        d = 0
    return d, D, {"seasonal_strength": strength, "kpss_pvalue": pval}


def candidate_grid(cfg: SarimaCfg) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Ordered simplest-first so a time-budget cut-off still leaves sensible candidates."""
    pq = [(p, q) for p in range(cfg.max_pq_sum + 1) for q in range(cfg.max_pq_sum + 1) if p + q <= cfg.max_pq_sum]
    spq = [
        (p, q)
        for p in range(cfg.max_seasonal_pq_sum + 1)
        for q in range(cfg.max_seasonal_pq_sum + 1)
        if p + q <= cfg.max_seasonal_pq_sum
    ]
    grid = [(a, b) for a in pq for b in spq]
    grid.sort(key=lambda ab: (sum(ab[0]) + sum(ab[1]), ab))
    return grid


class SarimaForecaster(Forecaster):
    name = "sarima"

    def __init__(self, cfg: SarimaCfg | None = None):
        self.cfg = cfg or SarimaCfg()

    @classmethod
    def from_config(cls, cfg: ForecastingCfg) -> SarimaForecaster:
        return cls(cfg.sarima)

    def _fit_forecast(self, req: ForecastRequest, on_stage: StageCallback):
        cfg = self.cfg
        m, h = req.seasonal_period, req.horizon
        use_log = cfg.log_transform_multiplicative and req.seasonal_form == "multiplicative"
        y = np.log(req.train) if use_log else np.asarray(req.train)

        on_stage(Stage.SELECT_PARAMETERS)
        d, D, diag = choose_differencing(y, m, cfg)
        trend = "c" if d + D <= 1 else "n"
        t0 = time.monotonic()
        candidates: list[dict] = []
        best = None  # (score, order, sorder, result)
        truncated = False
        for order_pq, sorder_pq in candidate_grid(cfg):
            if best is not None and time.monotonic() - t0 > cfg.time_budget_seconds:
                truncated = True
                break
            if best is None and time.monotonic() - t0 > cfg.time_budget_seconds:
                raise EvaluationTimeout(
                    f"SARIMA search exceeded {cfg.time_budget_seconds}s without a usable candidate"
                )
            order = (order_pq[0], d, order_pq[1])
            sorder = (sorder_pq[0], D, sorder_pq[1], m)
            rec: dict = {"order": list(order), "seasonal_order": list(sorder)}
            try:
                model = SARIMAX(y, order=order, seasonal_order=sorder, trend=trend,
                                enforce_stationarity=True, enforce_invertibility=True)
                res = model.fit(disp=False, maxiter=cfg.max_iterations)
                converged = bool(res.mle_retvals.get("converged", True)) if hasattr(res, "mle_retvals") and res.mle_retvals else True
                score = float(res.aic if cfg.selection_criterion == "aic" else res.bic)
                rec.update(converged=converged, score=score if math.isfinite(score) else None)
                if not converged or not math.isfinite(score):
                    rec["skipped"] = "non-converged" if not converged else "non-finite criterion"
                elif best is None or score < best[0]:
                    best = (score, order, sorder, res)
            except Exception as exc:
                rec.update(converged=False, error=f"{type(exc).__name__}: {exc}"[:200])
            candidates.append(rec)

        if best is None:
            raise ModelFitError(f"SARIMA: all {len(candidates)} candidate configurations failed")
        score, order, sorder, res = best
        on_stage(Stage.FIT_MODEL)  # best candidate already fitted during search
        on_stage(Stage.GENERATE_FORECAST)
        fc = np.asarray(res.forecast(h), dtype=float)
        if use_log:
            fc = np.exp(fc)
        params = {
            "order": list(order),
            "seasonal_order": list(sorder),
            "trend": trend,
            "selection_criterion": cfg.selection_criterion,
            "criterion_value": score,
            "log_transform": use_log,
            "search_limits": {
                "max_pq_sum": cfg.max_pq_sum,
                "max_seasonal_pq_sum": cfg.max_seasonal_pq_sum,
                "max_iterations": cfg.max_iterations,
                "time_budget_seconds": cfg.time_budget_seconds,
            },
            "n_candidates_considered": len(candidates),
            "n_candidates_planned": len(candidate_grid(cfg)),
            "search_truncated": truncated,
            "differencing_diagnostics": diag,
            "candidates": candidates,
        }
        return fc, params, {}


register("sarima", SarimaForecaster.from_config)
