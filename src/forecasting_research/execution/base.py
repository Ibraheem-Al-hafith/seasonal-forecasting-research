"""Execution strategy contract. The experiment manager depends on this, never on a hard-coded loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from ..domain.contracts import EvaluationTask, TaskOutcome

TaskRunnerFn = Callable[[EvaluationTask], TaskOutcome]


class ExecutionStrategy(ABC):
    """Executes independent evaluation tasks.

    Contract: tasks are independent; `runner` converts per-task failures into FAILED outcomes and
    raises PersistenceError only when execution must stop; `should_stop` is polled between tasks.
    A future parallel strategy must honour the same contract (with explicit resource limits).
    """

    @abstractmethod
    def execute(
        self,
        tasks: Sequence[EvaluationTask],
        runner: TaskRunnerFn,
        should_stop: Callable[[], bool],
    ) -> list[TaskOutcome]: ...
