from __future__ import annotations

from collections.abc import Callable, Sequence

from ..domain.contracts import EvaluationTask, TaskOutcome
from .base import ExecutionStrategy, TaskExecutor


class SequentialStrategy(ExecutionStrategy):
    """One evaluation at a time (predictable memory, easiest debugging)."""

    def execute(
        self, tasks: Sequence[EvaluationTask], runner: TaskExecutor, should_stop: Callable[[], bool]
    ) -> list[TaskOutcome]:
        outcomes: list[TaskOutcome] = []
        for task in tasks:
            if should_stop():
                break
            outcomes.append(runner(task))
        return outcomes

    def describe(self) -> str:
        return "sequential"
