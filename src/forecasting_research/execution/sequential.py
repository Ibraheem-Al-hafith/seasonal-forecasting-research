from __future__ import annotations

from collections.abc import Callable, Sequence

from ..domain.contracts import EvaluationTask, TaskOutcome
from .base import ExecutionStrategy, TaskRunnerFn


class SequentialStrategy(ExecutionStrategy):
    """MVP default: one evaluation at a time (predictable memory, simple recovery)."""

    def execute(
        self, tasks: Sequence[EvaluationTask], runner: TaskRunnerFn, should_stop: Callable[[], bool]
    ) -> list[TaskOutcome]:
        outcomes: list[TaskOutcome] = []
        for task in tasks:
            if should_stop():
                break
            outcomes.append(runner(task))
        return outcomes


def create_strategy(name: str) -> ExecutionStrategy:
    if name == "sequential":
        return SequentialStrategy()
    raise ValueError(f"Unknown execution strategy: {name}")
