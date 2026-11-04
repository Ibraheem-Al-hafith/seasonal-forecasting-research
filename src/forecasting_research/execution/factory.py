"""Build the execution strategy from configuration."""

from __future__ import annotations

from ..config.schemas import ExecutionCfg
from .base import ExecutionStrategy
from .parallel import ParallelStrategy, import_worker_modules, resolve_workers
from .sequential import SequentialStrategy

__all__ = ["create_strategy", "import_worker_modules"]


def create_strategy(cfg: ExecutionCfg) -> ExecutionStrategy:
    if cfg.strategy == "sequential":
        return SequentialStrategy()
    return ParallelStrategy(
        workers=resolve_workers(cfg.workers),
        start_method=cfg.start_method,
        worker_modules=cfg.worker_modules,
        max_tasks_per_worker=cfg.max_tasks_per_worker,
        limit_blas=cfg.limit_blas_threads,
        max_restarts=cfg.max_worker_restarts,
    )
