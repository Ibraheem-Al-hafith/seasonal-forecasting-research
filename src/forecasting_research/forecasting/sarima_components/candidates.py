"""Candidate generation (independent of model fitting, so the search boundaries can be tested alone)."""

from __future__ import annotations

from typing import NamedTuple

from ...config.schemas import SarimaCfg


class Candidate(NamedTuple):
    p: int
    q: int
    P: int
    Q: int

    @property
    def complexity(self) -> int:
        return self.p + self.q + self.P + self.Q

    def label(self) -> str:
        return f"({self.p},q={self.q})x({self.P},Q={self.Q})"


def _pairs(max_sum: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(max_sum + 1) for b in range(max_sum + 1) if a + b <= max_sum]


def _grid(max_pq: int, max_spq: int) -> list[Candidate]:
    grid = [Candidate(p, q, P, Q) for (p, q) in _pairs(max_pq) for (P, Q) in _pairs(max_spq)]
    return sorted(grid, key=lambda c: (c.complexity, c))  # simplest first: a time-budget cut-off keeps sensible models


def full_grid(cfg: SarimaCfg) -> list[Candidate]:
    """All candidates with p+q <= max_pq_sum and P+Q <= max_seasonal_pq_sum."""
    return _grid(cfg.max_pq_sum, cfg.max_seasonal_pq_sum)


def stage1_grid(cfg: SarimaCfg) -> list[Candidate]:
    """Screening candidates. For the full-grid strategy stage 1 *is* the full grid."""
    if cfg.search_strategy == "full_grid":
        return full_grid(cfg)
    return _grid(cfg.stage1_pq(), cfg.stage1_spq())


def stage2_grid(cfg: SarimaCfg) -> list[Candidate]:
    """The remaining candidates: full grid minus stage 1 (disjoint from it, union equals the full grid)."""
    done = set(stage1_grid(cfg))
    return [c for c in full_grid(cfg) if c not in done]
