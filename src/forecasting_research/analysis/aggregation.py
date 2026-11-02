"""Grouped aggregation of stored evaluation results.

Aggregation policy (explicit):
  * Metric statistics are computed over COMPLETED evaluations only, and only over *defined* values
    (an undefined MAPE is excluded from MAPE statistics but counted in `MAPE_n_undefined`).
  * Failed evaluations are never dropped silently: every row reports n_total, n_completed, n_failed.
  * Grouping by a subset of dimensions pools the remaining dimensions (marginal means).
"""

from __future__ import annotations

import pandas as pd

from ..config.defaults import METRIC_NAMES
from ..domain.exceptions import ConfigError

ALL_DIMS = ["seasonality_type", "seasonal_period", "trend_type", "noise_level", "length"]
_ALIASES = {
    "seasonality": "seasonality_type", "seasonality_type": "seasonality_type",
    "period": "seasonal_period", "seasonal_period": "seasonal_period",
    "trend": "trend_type", "trend_type": "trend_type",
    "noise": "noise_level", "noise_level": "noise_level",
    "length": "length", "dataset_length": "length",
}
STATS = ["mean", "std", "median", "min", "max"]


def normalize_dims(names: list[str] | str | None) -> list[str]:
    if names is None:
        return list(ALL_DIMS)
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    out = []
    for n in names:
        if n.lower() not in _ALIASES:
            raise ConfigError(f"Unknown grouping dimension {n!r}. Valid: {sorted(set(_ALIASES))}")
        out.append(_ALIASES[n.lower()])
    return out


def normalize_metric(name: str) -> str:
    up = name.upper()
    if up not in METRIC_NAMES:
        raise ConfigError(f"Unknown metric {name!r}. Valid: {METRIC_NAMES}")
    return up


def aggregate_metrics(results: pd.DataFrame, group_by: list[str] | None = None, metrics: list[str] | None = None) -> pd.DataFrame:
    """One row per (group, model) with counts and metric statistics."""
    keys = normalize_dims(group_by) + ["model_name"]
    metrics = [normalize_metric(m) for m in (metrics or METRIC_NAMES)]
    if results.empty:
        return pd.DataFrame(columns=keys + ["n_total", "n_completed", "n_failed"])
    df = results.copy()
    g = df.groupby(keys, dropna=False)
    out = g.agg(n_total=("evaluation_id", "size")).reset_index()
    status_counts = (
        df.assign(_c=(df["status"] == "COMPLETED").astype(int), _f=(df["status"] == "FAILED").astype(int))
        .groupby(keys, dropna=False)[["_c", "_f"]].sum().rename(columns={"_c": "n_completed", "_f": "n_failed"}).reset_index()
    )
    out = out.merge(status_counts, on=keys, how="left")
    comp = df[df["status"] == "COMPLETED"]
    for metric in metrics:
        if comp.empty:
            for s in STATS + ["n_valid"]:
                out[f"{metric}_{s}"] = float("nan") if s != "n_valid" else 0
            continue
        stats = comp.groupby(keys, dropna=False)[metric].agg(STATS + ["count"]).reset_index()
        stats = stats.rename(columns={s: f"{metric}_{s}" for s in STATS} | {"count": f"{metric}_n_valid"})
        out = out.merge(stats, on=keys, how="left")
        out[f"{metric}_n_valid"] = out[f"{metric}_n_valid"].fillna(0).astype(int)
    if "MAPE" in metrics:
        und = comp[comp["MAPE"].isna()].groupby(keys, dropna=False).size().rename("MAPE_n_undefined").reset_index() if not comp.empty else None
        out = out.merge(und, on=keys, how="left") if und is not None and not und.empty else out.assign(MAPE_n_undefined=0)
        out["MAPE_n_undefined"] = out["MAPE_n_undefined"].fillna(0).astype(int)
    return out
