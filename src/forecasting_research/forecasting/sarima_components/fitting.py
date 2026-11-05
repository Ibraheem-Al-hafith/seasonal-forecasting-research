"""Fit ONE candidate and classify the outcome strictly.

A candidate is *selectable* only if it fitted, reported convergence, and has a finite criterion value.
Missing convergence metadata is never assumed to mean "converged".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ...config.schemas import SarimaCfg
from .candidates import Candidate

CRITERIA = ("aic", "aicc", "bic")

OK = "ok"
NON_CONVERGED = "non_converged"
NO_CONVERGENCE_INFO = "no_convergence_info"
NON_FINITE = "non_finite_criterion"
ERROR = "error"


def criterion_value(res: Any, criterion: str) -> float:
    """Read the configured information criterion from a SARIMAXResults-like object (explicit mapping)."""
    if criterion not in CRITERIA:
        raise ValueError(f"Unsupported selection criterion {criterion!r}; choose one of {CRITERIA}")
    return float(getattr(res, criterion))


@dataclass
class CandidateRecord:
    candidate: Candidate
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    stage: int
    status: str = OK
    converged: bool | None = None
    score: float | None = None
    aic: float | None = None
    aicc: float | None = None
    bic: float | None = None
    fit_seconds: float = 0.0
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def selectable(self) -> bool:
        return self.status == OK

    def to_dict(self) -> dict:
        return {"order": list(self.order), "seasonal_order": list(self.seasonal_order), "stage": self.stage,
                "status": self.status, "converged": self.converged, "score": self.score, "aic": self.aic,
                "aicc": self.aicc, "bic": self.bic, "fit_seconds": round(self.fit_seconds, 3), "error": self.error}


def _finite(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fit_candidate(y: np.ndarray, cand: Candidate, d: int, D: int, m: int, trend: str, cfg: SarimaCfg, stage: int) -> tuple[CandidateRecord, Any]:
    """Fit SARIMAX(p,d,q)(P,D,Q,m). Returns (record, result-or-None); never raises for fit problems."""
    order, sorder = (cand.p, d, cand.q), (cand.P, D, cand.Q, m)
    rec = CandidateRecord(cand, order, sorder, stage)
    t0 = time.perf_counter()
    res = None
    try:
        model = SARIMAX(y, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=True, enforce_invertibility=True)
        res = model.fit(disp=False, maxiter=cfg.max_iterations)
        retvals = getattr(res, "mle_retvals", None)
        rec.aic, rec.aicc, rec.bic = (_finite(getattr(res, k, None)) for k in CRITERIA)
        if not isinstance(retvals, dict) or "converged" not in retvals:
            rec.status = NO_CONVERGENCE_INFO
        else:
            rec.converged = bool(retvals["converged"])
            score = _finite(criterion_value(res, cfg.selection_criterion))
            rec.score = score
            if not rec.converged:
                rec.status = NON_CONVERGED
            elif score is None:
                rec.status = NON_FINITE
    except Exception as exc:
        rec.status, rec.error = ERROR, f"{type(exc).__name__}: {exc}"[:200]
    rec.fit_seconds = time.perf_counter() - t0
    return rec, (res if rec.selectable else None)


def residual_autocorrelation(res: Any, cand: Candidate, d: int, D: int, m: int, alpha: float) -> dict:
    """Ljung-Box test on TRAINING residuals (burn-in of the differencing operators removed).

    A small p-value means autocorrelation remains: a diagnostic signal, not proof that a bigger model forecasts better.
    """
    out: dict = {"alpha": alpha}
    try:
        resid = np.asarray(res.resid, dtype=float)[d + D * m:]
        model_df = cand.p + cand.q + cand.P + cand.Q
        lags = min(2 * m, len(resid) // 4)
        if lags <= model_df or len(resid) < 10:
            out["note"] = "too few residuals for a Ljung-Box test"
            return out
        lb = acorr_ljungbox(resid, lags=[lags], model_df=model_df)
        p = _finite(lb["lb_pvalue"].iloc[0])
        out.update(lags=int(lags), model_df=model_df, pvalue=p, significant=bool(p is not None and p < alpha))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out
