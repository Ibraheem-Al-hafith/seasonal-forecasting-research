"""Identifiers and evaluation planning."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime

from ..domain.contracts import EvaluationTask


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


def make_experiment_id() -> str:
    return f"exp_{_stamp()}"


def make_batch_id() -> str:
    return f"batch_{_stamp()}"


def make_evaluation_id(experiment_id: str, dataset_id: str, model_name: str) -> str:
    h = hashlib.sha256(f"{experiment_id}|{dataset_id}|{model_name}".encode()).hexdigest()[:16]
    return f"ev_{h}"


def build_plan(experiment_id: str, dataset_ids: list[str], models: list[str]) -> list[EvaluationTask]:
    """Dataset-major order: each dataset is loaded once and every method evaluated on it in turn."""
    return [
        EvaluationTask(make_evaluation_id(experiment_id, d, m), experiment_id, d, m)
        for d in dataset_ids
        for m in models
    ]
