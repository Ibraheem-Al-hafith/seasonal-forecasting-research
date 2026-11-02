"""Model comparison: best method per group and metric."""

from __future__ import annotations

import pandas as pd

from .aggregation import STATS, normalize_dims, normalize_metric


def best_method_per_group(agg: pd.DataFrame, group_by: list[str] | None, metric: str = "RMSE", stat: str = "mean") -> pd.DataFrame:
    """For each group, the model with the lowest `stat` of `metric` (ties broken by model name).

    `complete_coverage` is False when any method in the group lacks as many valid results as the
    best-covered method (failures / undefined values) - such comparisons should be read with care.
    """
    dims = normalize_dims(group_by)
    metric = normalize_metric(metric)
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}")
    col, ncol = f"{metric}_{stat}", f"{metric}_n_valid"
    cols = dims + ["metric", "stat", "best_model", "best_value", "runner_up_model", "runner_up_value",
                   "n_methods", "n_methods_with_results", "complete_coverage", "n_failed_total"]
    if agg.empty or col not in agg.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, sub in agg.groupby(dims, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        valid = sub.dropna(subset=[col]).sort_values([col, "model_name"])
        max_valid = int(sub[ncol].max()) if len(sub) else 0
        coverage = bool(len(valid) == len(sub) and (sub[ncol] == max_valid).all() and sub["n_failed"].sum() == 0)
        rec = dict(zip(dims, key, strict=True))
        rec.update(metric=metric, stat=stat, n_methods=len(sub), n_methods_with_results=len(valid),
                   complete_coverage=coverage, n_failed_total=int(sub["n_failed"].sum()),
                   best_model=None, best_value=None, runner_up_model=None, runner_up_value=None)
        if len(valid) >= 1:
            rec.update(best_model=valid.iloc[0]["model_name"], best_value=float(valid.iloc[0][col]))
        if len(valid) >= 2:
            rec.update(runner_up_model=valid.iloc[1]["model_name"], runner_up_value=float(valid.iloc[1][col]))
        rows.append(rec)
    return pd.DataFrame(rows, columns=cols)


def win_counts(best: pd.DataFrame) -> pd.DataFrame:
    """How often each model is best, per metric."""
    if best.empty:
        return pd.DataFrame(columns=["metric", "best_model", "groups_won"])
    return (best.dropna(subset=["best_model"]).groupby(["metric", "best_model"]).size()
            .rename("groups_won").reset_index().sort_values(["metric", "groups_won"], ascending=[True, False]))
