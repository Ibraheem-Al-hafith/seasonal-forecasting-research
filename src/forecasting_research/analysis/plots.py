"""Matplotlib figures built exclusively from stored results (OO API: no global backend state)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..persistence.artifacts import ArtifactStore
from ..persistence.repositories import Repository
from .aggregation import normalize_dims, normalize_metric


def plot_actual_vs_forecast(repo: Repository, store: ArtifactStore, experiment_id: str, dataset_id: str) -> Figure:
    rec = repo.datasets.get_dataset(dataset_id)
    y = store.load_observed(rec.observed_path, rec.observed_sha256)
    evals = [e for e in repo.evaluations.list_for_experiment(experiment_id, statuses=["COMPLETED"]) if e["dataset_id"] == dataset_id]
    fig = Figure(figsize=(9, 4.5))
    ax = fig.subplots()
    if not evals:
        ax.text(0.5, 0.5, "No completed evaluations for this dataset", ha="center", va="center")
        return fig
    train_size, horizon = evals[0]["train_size"], evals[0]["horizon"]
    start = max(0, train_size - 3 * horizon)
    ax.plot(range(start, train_size), y[start:train_size], color="0.5", label="train (tail)")
    xs = range(train_size, train_size + horizon)
    actual, _ = store.load_evaluation(evals[0]["artifact_path"], evals[0]["artifact_sha256"])
    ax.plot(xs, actual, color="black", lw=2, label="actual")
    for e in evals:
        _, fc = store.load_evaluation(e["artifact_path"], e["artifact_sha256"])
        ax.plot(xs, fc, marker="o", ms=3, label=e["model_name"])
    ax.axvline(train_size - 0.5, color="0.7", ls="--")
    ax.set(title=f"Actual vs forecast - {dataset_id} ({rec.seasonality_type}, m={rec.seasonal_period}, "
                 f"{rec.trend_type} trend, {rec.noise_level} noise)", xlabel="t", ylabel="value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_metric_by_dimension(results: pd.DataFrame, dimension: str, metric: str = "RMSE") -> Figure:
    dim = normalize_dims([dimension])[0]
    metric = normalize_metric(metric)
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    comp = results[results["status"] == "COMPLETED"].dropna(subset=[metric])
    if comp.empty:
        ax.text(0.5, 0.5, "No completed results to plot", ha="center", va="center")
        return fig
    mean = comp.groupby([dim, "model_name"])[metric].mean().unstack("model_name")
    std = comp.groupby([dim, "model_name"])[metric].std().unstack("model_name").reindex_like(mean)
    n_models = len(mean.columns)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(mean.index))
    for i, model in enumerate(mean.columns):
        ax.bar(x + i * width - 0.4 + width / 2, mean[model], width, yerr=std[model].fillna(0), capsize=2, label=model)
    ax.set_xticks(x, [str(v) for v in mean.index])
    ax.set(xlabel=dim, ylabel=f"mean {metric}", title=f"Mean {metric} by {dim} (error bars: std across evaluations)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def figure_to_png(fig: Figure, path) -> None:
    fig.savefig(path, dpi=130)
