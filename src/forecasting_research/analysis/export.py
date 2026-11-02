"""CSV export (atomic writes)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from ..domain.exceptions import PersistenceError
from ..persistence.artifacts import ArtifactStore
from ..persistence.repositories import Repository


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        tmp = None
    except OSError as exc:
        raise PersistenceError(f"Could not write export {path}: {exc}") from exc
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return path


def forecasts_long_table(repo: Repository, store: ArtifactStore, experiment_id: str) -> pd.DataFrame:
    """Actual and predicted values for every COMPLETED evaluation, straight from the stored artifacts."""
    rows = []
    for ev in repo.evaluations.list_for_experiment(experiment_id, statuses=["COMPLETED"]):
        actual, forecast = store.load_evaluation(ev["artifact_path"], ev["artifact_sha256"])
        for i, (a, f) in enumerate(zip(actual, forecast, strict=True), 1):
            rows.append({"experiment_id": experiment_id, "evaluation_id": ev["evaluation_id"], "dataset_id": ev["dataset_id"],
                         "model_name": ev["model_name"], "step": i, "actual": float(a), "forecast": float(f)})
    return pd.DataFrame(rows, columns=["experiment_id", "evaluation_id", "dataset_id", "model_name", "step", "actual", "forecast"])
