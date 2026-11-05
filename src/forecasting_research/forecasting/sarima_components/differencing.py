"""Differencing selection and the seasonal-strength DIAGNOSTIC (training data only).

Seasonal strength measures how prominent the seasonal pattern is. It says nothing about whether the series
has a seasonal UNIT ROOT, so by default it no longer drives the choice of D (see SarimaCfg).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss

from ...config.schemas import SarimaCfg
from ...domain.exceptions import ConfigError, DataError
from ..classical_decomposition import centered_moving_average, check_decomposition_input, seasonal_indices

MIN_RELIABLE_OBS_PER_PHASE = 5  # heuristic: below this the seasonal indices (hence the strength) are noisy


@dataclass(frozen=True)
class StrengthResult:
    value: float
    domain: str  # "additive" (raw data) | "log" (multiplicative data, decomposed additively after log)
    obs_per_phase: float
    reliable: bool

    def to_dict(self) -> dict:
        return {"value": self.value, "domain": self.domain, "obs_per_phase": round(self.obs_per_phase, 2), "reliable": self.reliable}


def seasonal_strength(y: np.ndarray, m: int, form: str = "additive") -> StrengthResult:
    """F_s = max(0, 1 - Var(R) / Var(S + R)) from a classical decomposition in the correct domain.

    Multiplicative data are log-transformed first (log Y = log T + log S + log R), so the strength is computed
    consistently with the additive structure. Constant series (zero variance) return 0.0, not an error.
    """
    y = check_decomposition_input(y, m)
    if form == "multiplicative":
        if np.any(y <= 0):
            raise DataError("Multiplicative seasonal strength needs strictly positive data (log transform)")
        z, domain = np.log(y), "log"
    elif form == "additive":
        z, domain = y, "additive"
    else:
        raise DataError(f"Unknown seasonal form: {form}")
    trend = centered_moving_average(z, m)
    with warnings.catch_warnings():  # the instability warning is reported through `reliable` instead
        warnings.simplefilter("ignore", UserWarning)
        idx = seasonal_indices(z, trend, m, "additive")
    seasonal = idx[np.arange(len(z)) % m]
    ok = np.isfinite(trend)
    remainder = (z - trend - seasonal)[ok]
    denom = float(np.var(seasonal[ok] + remainder))
    value = 0.0 if denom <= 1e-300 else float(max(0.0, 1.0 - np.var(remainder) / denom))
    per_phase = ok.sum() / m
    return StrengthResult(value, domain, float(per_phase), bool(per_phase >= MIN_RELIABLE_OBS_PER_PHASE))


# ---------------------------------------------------------------------------- seasonal differencing (D)
def choose_seasonal_differencing(y: np.ndarray, m: int, form: str, cfg: SarimaCfg) -> tuple[int, dict]:
    """Returns (D, diagnostics). The strength diagnostic is always computed and stored."""
    strength = seasonal_strength(y, m, form)
    info: dict = {"strategy": cfg.seasonal_differencing_strategy, "seasonal_strength": strength.to_dict()}
    strategy = cfg.seasonal_differencing_strategy
    if strategy == "fixed_zero":
        return 0, info
    if strategy == "fixed_one":
        return 1, info
    if strategy == "seasonal_strength":  # legacy rule, kept for comparison studies only
        info["threshold"] = cfg.seasonal_strength_threshold
        return (1 if strength.value > cfg.seasonal_strength_threshold else 0), info
    # ocsb / ch: seasonal unit-root tests (optional dependency)
    nsdiffs = _load_nsdiffs()
    try:
        D = int(nsdiffs(np.asarray(y, dtype=float), m=m, test=strategy, max_D=1))
        info["test"] = strategy
        return D, info
    except Exception as exc:  # recorded, never silent
        info["test_error"] = f"{type(exc).__name__}: {exc}"[:200]
        warnings.warn(f"seasonal unit-root test '{strategy}' failed ({info['test_error']}); using D=0", UserWarning, stacklevel=2)
        return 0, info


def _load_nsdiffs():
    try:
        from pmdarima.arima import nsdiffs
    except ImportError as exc:
        raise ConfigError(
            "seasonal_differencing_strategy 'ocsb'/'ch' needs the optional pmdarima package: pip install pmdarima") from exc
    return nsdiffs


def require_strategy_dependencies(cfg: SarimaCfg) -> None:
    """Fail at construction time, not in the middle of an experiment."""
    if cfg.seasonal_differencing_strategy in ("ocsb", "ch"):
        _load_nsdiffs()


# ---------------------------------------------------------------------------- regular differencing (d) + trend
def _kpss(z: np.ndarray, regression: str) -> dict:
    out: dict = {"regression": regression}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stat, pval, lags, _ = kpss(z, regression=regression, nlags="auto")
        out.update(statistic=float(stat), pvalue=float(pval), lags=int(lags),
                   pvalue_bounded=any(issubclass(w.category, InterpolationWarning) for w in caught))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def choose_regular_differencing(y: np.ndarray, m: int, D: int, cfg: SarimaCfg) -> tuple[int, str, dict]:
    """Returns (d, sarimax_trend, diagnostics) from KPSS tests on the (seasonally differenced) training series.

    A series that is stationary around a deterministic linear trend does not need differencing: with
    kpss_regression 'auto' or 'ct' it keeps d = 0 and receives an explicit deterministic trend ('ct').
    After seasonal differencing a linear trend is merely a constant, so the 'ct' branch applies only if D = 0.
    """
    z = y[m:] - y[:-m] if D == 1 else np.asarray(y, dtype=float)
    mode, alpha = cfg.kpss_regression, cfg.kpss_alpha
    tests: list[dict] = []

    def rejects(t: dict) -> bool:
        return t.get("pvalue") is not None and t["pvalue"] < alpha

    chosen = "c"
    if mode == "ct" and D == 0:
        t = _kpss(z, "ct")
        tests.append(t)
        d, chosen = (1, "c") if rejects(t) else (0, "ct")
        if "error" in t:
            d, chosen = 0, "c"
    else:
        t_c = _kpss(z, "c")
        tests.append(t_c)
        if "error" in t_c or not rejects(t_c):
            d = 0
        elif mode == "auto" and D == 0:
            t_ct = _kpss(z, "ct")
            tests.append(t_ct)
            d, chosen = (0, "ct") if ("error" not in t_ct and not rejects(t_ct)) else (1, "c")
        else:
            d = 1
    if chosen == "ct":
        trend = "ct"
    else:
        trend = "c" if d + D <= 1 else "n"
    for t in tests:
        if "error" in t:
            warnings.warn(f"KPSS({t['regression']}) failed ({t['error']}); defaulting to d=0 with a constant", UserWarning, stacklevel=2)
    return d, trend, {"kpss_regression": mode, "kpss_alpha": alpha, "kpss_tests": tests, "d": d,
                      "deterministic_trend": describe_trend(d, D, trend)}


def describe_trend(d: int, D: int, trend: str) -> str:
    """Human-readable meaning of the SARIMAX `trend` argument under the chosen differencing."""
    if trend == "ct":
        return "linear_time_trend"
    if trend == "n":
        return "none"
    return {(0, 0): "intercept", (1, 0): "drift", (0, 1): "seasonal_drift"}.get((d, D), "constant")
