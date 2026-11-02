"""Centralised state-transition and outcome policy (single source of truth)."""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.enums import EvaluationStatus as E
from ..domain.enums import ExperimentOutcome, ExperimentStatus
from ..domain.exceptions import ResearchError

# An evaluation may only move along these edges.
EVALUATION_TRANSITIONS: dict[E, set[E]] = {
    E.PENDING: {E.RUNNING},
    E.RUNNING: {E.COMPLETED, E.FAILED, E.INTERRUPTED},
    E.INTERRUPTED: {E.RUNNING},
    E.FAILED: {E.RUNNING},  # only when a retry is explicitly requested (see recovery.py)
    E.COMPLETED: {E.RUNNING},  # only when integrity verification invalidates the stored result
}


class IllegalTransition(ResearchError):
    pass


def assert_evaluation_transition(old: str, new: str) -> None:
    if E(new) not in EVALUATION_TRANSITIONS[E(old)]:
        raise IllegalTransition(f"Illegal evaluation transition {old} -> {new}")


def derive_experiment_status(
    counts: Mapping[str, int], *, aborted: bool = False, interrupted: bool = False
) -> tuple[ExperimentStatus, ExperimentOutcome]:
    """Map evaluation outcomes to the experiment's final (status, outcome).

    * aborted by persistence failure           -> FAILED / ABORTED
    * work remaining (stopped or crashed)      -> INTERRUPTED / INCOMPLETE
    * all COMPLETED                            -> COMPLETED / SUCCESS
    * all processed, some FAILED               -> COMPLETED / PARTIAL
    * every evaluation FAILED                  -> FAILED / ALL_FAILED
    """
    if aborted:
        return ExperimentStatus.FAILED, ExperimentOutcome.ABORTED
    remaining = counts.get("PENDING", 0) + counts.get("RUNNING", 0) + counts.get("INTERRUPTED", 0)
    if interrupted or remaining > 0:
        return ExperimentStatus.INTERRUPTED, ExperimentOutcome.INCOMPLETE
    failed, completed = counts.get("FAILED", 0), counts.get("COMPLETED", 0)
    if failed == 0:
        return ExperimentStatus.COMPLETED, ExperimentOutcome.SUCCESS
    if completed == 0:
        return ExperimentStatus.FAILED, ExperimentOutcome.ALL_FAILED
    return ExperimentStatus.COMPLETED, ExperimentOutcome.PARTIAL


def describe_status(status: str, outcome: str, failed: int = 0) -> str:
    """Human display string: partial runs are never shown as plain success."""
    if status == "COMPLETED" and outcome == "PARTIAL":
        return f"COMPLETED (partial: {failed} failed)"
    return status
