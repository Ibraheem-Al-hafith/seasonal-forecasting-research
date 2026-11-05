"""Search engine: stages, time budget, expansion decision. Knows nothing about SARIMAX internals.

Everything here uses TRAINING data only (the evaluator never hands test data to a forecaster), so the
decision to run stage 2 cannot leak test information.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...config.schemas import SarimaCfg
from ...domain.exceptions import EvaluationTimeout, ModelFitError
from .candidates import Candidate, full_grid, stage1_grid, stage2_grid
from .fitting import CandidateRecord, fit_candidate, residual_autocorrelation

FitFn = Callable[..., tuple[CandidateRecord, Any]]
DiagFn = Callable[..., dict]


@dataclass
class SearchOutcome:
    best: CandidateRecord
    best_result: Any
    records: list[CandidateRecord]
    stages: list[dict] = field(default_factory=list)
    expansion: dict = field(default_factory=dict)
    truncated: bool = False
    stopped_reason: str = "completed"
    seconds: float = 0.0

    def counts(self) -> dict:
        return count_records(self.records)


def count_records(records: list[CandidateRecord]) -> dict:
    by: dict[str, int] = {}
    for r in records:
        by[r.status] = by.get(r.status, 0) + 1
    return {"n_candidates_fitted": len(records), "n_selectable": by.get("ok", 0),
            "n_non_converged": by.get("non_converged", 0), "n_no_convergence_info": by.get("no_convergence_info", 0),
            "n_non_finite": by.get("non_finite_criterion", 0), "n_errors": by.get("error", 0)}


def _at_stage1_boundary(c: Candidate, cfg: SarimaCfg) -> dict:
    """Best stage-1 model sits on a stage-1 limit that is lower than the final limit (so expansion is possible)."""
    s1_pq, s1_spq = cfg.stage1_pq(), cfg.stage1_spq()
    nonseasonal = s1_pq < cfg.max_pq_sum and c.p + c.q >= s1_pq > 0
    seasonal = s1_spq < cfg.max_seasonal_pq_sum and c.P + c.Q >= s1_spq > 0
    return {"nonseasonal": bool(nonseasonal), "seasonal": bool(seasonal), "any": bool(nonseasonal or seasonal)}


def decide_expansion(best: CandidateRecord | None, best_result: Any, cfg: SarimaCfg, d: int, D: int, m: int,
                     diag_fn: DiagFn = residual_autocorrelation) -> dict:
    """Decide whether stage 2 runs and record why (indicators + reasons)."""
    info: dict = {"trigger_policy": cfg.stage2_trigger, "triggered": False, "reasons": []}
    if best is None:
        info.update(triggered=True, reasons=["no_selectable_stage1_model"])
        return info
    boundary = _at_stage1_boundary(best.candidate, cfg)
    info["boundary"] = boundary
    lb = diag_fn(best_result, best.candidate, d, D, m, cfg.ljung_box_alpha)
    info["ljung_box"] = lb
    policy = cfg.stage2_trigger
    reasons = []
    if policy == "always":
        reasons.append("always")
    if policy in ("boundary", "boundary_or_ljung_box") and boundary["any"]:
        reasons.append("best_stage1_model_at_boundary")
    if policy in ("ljung_box", "boundary_or_ljung_box") and lb.get("significant"):
        reasons.append("residual_autocorrelation")
    info.update(triggered=bool(reasons), reasons=reasons)
    return info


def run_search(y: np.ndarray, d: int, D: int, m: int, trend: str, cfg: SarimaCfg, *,
               fit_fn: FitFn = fit_candidate, diag_fn: DiagFn = residual_autocorrelation,
               clock: Callable[[], float] = time.monotonic) -> SearchOutcome:
    """Stage 1 (screening) then, if warranted, stage 2 (remaining candidates). Soft time budget between fits."""
    t_start = clock()
    records: list[CandidateRecord] = []
    stages: list[dict] = []
    best: tuple[float, CandidateRecord, Any] | None = None
    truncated = False

    def run_stage(stage: int, cands: list[Candidate]) -> bool:
        nonlocal best, truncated
        t0, n0 = clock(), len(records)
        for cand in cands:
            if clock() - t_start > cfg.time_budget_seconds:
                if best is None:
                    raise EvaluationTimeout(f"SARIMA search exceeded {cfg.time_budget_seconds}s without a usable candidate")
                truncated = True
                break
            rec, res = fit_fn(y, cand, d, D, m, trend, cfg, stage)
            records.append(rec)
            if rec.selectable and (best is None or rec.score < best[0]):  # type: ignore[operator]
                best = (rec.score, rec, res)  # type: ignore[assignment]
        fitted = records[n0:]
        stages.append({"stage": stage, "n_planned": len(cands), "n_fitted": len(fitted),
                       "n_selectable": sum(r.selectable for r in fitted), "seconds": round(clock() - t0, 3)})
        return not truncated

    first = stage1_grid(cfg)
    run_stage(1, first)
    expansion: dict = {"strategy": cfg.search_strategy, "triggered": False, "reasons": []}
    remaining = stage2_grid(cfg) if cfg.search_strategy == "two_stage" else []
    if remaining and not truncated:
        expansion = {"strategy": cfg.search_strategy,
                     **decide_expansion(best[1] if best else None, best[2] if best else None, cfg, d, D, m, diag_fn)}
        if expansion["triggered"]:
            run_stage(2, remaining)
    elif remaining:
        expansion["note"] = "stage 2 skipped: time budget exhausted in stage 1"
    if best is None:
        raise ModelFitError(f"SARIMA: none of the {len(records)} fitted candidates was selectable "
                            f"({count_records(records)})")
    return SearchOutcome(best=best[1], best_result=best[2], records=records, stages=stages, expansion=expansion,
                         truncated=truncated, stopped_reason="budget_exhausted" if truncated else "completed",
                         seconds=clock() - t_start)


__all__ = ["SearchOutcome", "run_search", "decide_expansion", "full_grid"]
