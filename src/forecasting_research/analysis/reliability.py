"""Reliability and search diagnostics (the report's dimensions B and C), computed from STORED results.

Forecast accuracy answers "how good are the forecasts when a method works"; these tables answer
"how often does it work, how much does it cost, and what did the search do". Failures are never dropped:
`success_pct` is computed over all evaluations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..persistence.artifacts import ArtifactStore
from ..persistence.repositories import Repository
from .aggregation import normalize_dims


def _dims(group_by) -> list[str]:
    return [] if group_by is None else normalize_dims(group_by)


def is_extreme_forecast(train: np.ndarray, forecast: np.ndarray) -> bool:
    """Extreme = any forecast value further than one full training range outside the observed training range."""
    lo, hi = float(np.min(train)), float(np.max(train))
    span = (hi - lo) or abs(hi) or 1.0
    return bool(np.any(forecast < lo - span) or np.any(forecast > hi + span))


def reliability_table(repo: Repository, store: ArtifactStore, experiment_ids: list[str] | None = None,
                      group_by: list[str] | str | None = None) -> pd.DataFrame:
    """Per (group, model): success rate, extreme-forecast rate and runtime statistics."""
    dims = _dims(group_by)
    keys = dims + ["model_name"]
    res = repo.evaluations.results_table(experiment_ids)
    if res.empty:
        return pd.DataFrame(columns=keys)
    extreme, cache = [], {}
    for r in res.itertuples():
        if r.status != "COMPLETED":
            extreme.append(np.nan)
            continue
        ev = repo.evaluations.get(r.evaluation_id)
        if r.dataset_id not in cache:
            rec = repo.datasets.get_dataset(r.dataset_id)
            cache[r.dataset_id] = store.load_observed(rec.observed_path, rec.observed_sha256)
        _, fc = store.load_evaluation(ev["artifact_path"], ev["artifact_sha256"])
        extreme.append(float(is_extreme_forecast(cache[r.dataset_id][: int(r.train_size)], fc)))
    res = res.assign(_extreme=extreme, _completed=(res["status"] == "COMPLETED").astype(int),
                     _failed=(res["status"] == "FAILED").astype(int))
    g = res.groupby(keys, dropna=False)
    out = g.agg(n_total=("evaluation_id", "size"), n_completed=("_completed", "sum"), n_failed=("_failed", "sum"),
                extreme_forecasts=("_extreme", "sum"),
                mean_runtime_s=("runtime_seconds", "mean"), median_runtime_s=("runtime_seconds", "median"),
                total_runtime_s=("runtime_seconds", "sum")).reset_index()
    out["success_pct"] = 100.0 * out["n_completed"] / out["n_total"]
    out["extreme_forecast_pct"] = 100.0 * out["extreme_forecasts"] / out["n_completed"].where(out["n_completed"] > 0)
    cols = keys + ["n_total", "n_completed", "n_failed", "success_pct", "extreme_forecasts", "extreme_forecast_pct",
                   "mean_runtime_s", "median_runtime_s", "total_runtime_s"]
    return out[cols]


def _get(d: dict | None, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def sarima_search_table(repo: Repository, experiment_ids: list[str] | None = None,
                        group_by: list[str] | str | None = None) -> pd.DataFrame:
    """Per group: what the SARIMA searches did (fits, stage-2 rate, truncation, convergence, differencing)."""
    dims = _dims(group_by)
    df = repo.evaluations.parameter_rows(experiment_ids, "sarima")
    if df.empty:
        return pd.DataFrame(columns=dims + ["n_searches"])
    P = df["parameters"]
    ok = P.notna()
    n_fit = P.map(lambda p: _get(p, "n_candidates_considered"))
    df = df.assign(
        _ok=ok.astype(int),
        _fits=pd.to_numeric(n_fit, errors="coerce"),
        _stage2=P.map(lambda p: _get(p, "expansion", "triggered")).map(lambda v: np.nan if v is None else float(bool(v))),
        _trunc=P.map(lambda p: _get(p, "search_truncated")).map(lambda v: np.nan if v is None else float(bool(v))),
        _selectable=P.map(lambda p: (_get(p, "n_selectable", default=np.nan) / _get(p, "n_candidates_fitted", default=np.nan) * 100)
                          if _get(p, "n_candidates_fitted") else np.nan),
        _D1=P.map(lambda p: _get(p, "differencing", "D")).map(lambda v: np.nan if v is None else float(v == 1)),
        _d1=P.map(lambda p: _get(p, "differencing", "d")).map(lambda v: np.nan if v is None else float(v == 1)),
        _strength=P.map(lambda p: _get(p, "differencing", "seasonal", "seasonal_strength", "value")),
        _unreliable=P.map(lambda p: _get(p, "differencing", "seasonal", "seasonal_strength", "reliable")).map(
            lambda v: np.nan if v is None else float(not v)),
        _version=P.map(lambda p: _get(p, "method_version", default=1)),
        _search_s=P.map(lambda p: _get(p, "search_seconds")),
        _failed=(df["status"] == "FAILED").astype(int),
    )
    keys = dims if dims else None
    grouped = df.groupby(keys, dropna=False) if keys else df.groupby(lambda _: "all")
    out = grouped.agg(
        n_searches=("evaluation_id", "size"), n_failed=("_failed", "sum"),
        mean_fits=("_fits", "mean"), median_fits=("_fits", "median"),
        pct_stage2=("_stage2", lambda s: 100 * s.mean()), pct_truncated=("_trunc", lambda s: 100 * s.mean()),
        mean_selectable_pct=("_selectable", "mean"),
        pct_D1=("_D1", lambda s: 100 * s.mean()), pct_d1=("_d1", lambda s: 100 * s.mean()),
        mean_seasonal_strength=("_strength", "mean"), pct_strength_unreliable=("_unreliable", lambda s: 100 * s.mean()),
        median_search_s=("runtime_seconds", "median"), mean_search_s=("runtime_seconds", "mean"),
        total_s=("runtime_seconds", "sum"), method_versions=("_version", lambda s: ",".join(sorted({str(v) for v in s}))),
    ).reset_index()
    if not keys:
        out = out.drop(columns=["index"], errors="ignore")
    return out
