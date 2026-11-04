"""Execution strategy contract. The experiment manager depends on this, never on a hard-coded loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol

from ..domain.contracts import EvaluationTask, TaskOutcome
from ..domain.enums import Stage
from ..experiments.work import WorkItem, WorkResult


class TaskExecutor(Protocol):
    """What a strategy may call (implemented by experiments.task_runner.TaskRunner).

    `__call__` runs a whole task in-process. A parallel strategy instead uses the three parent-side
    steps around a pure computation it ships to a worker: begin -> work_item -> (compute) -> commit.
    All of these run in the PARENT process: it is the only writer to SQLite and the artifact store.
    """

    def __call__(self, task: EvaluationTask) -> TaskOutcome: ...
    def begin(self, task: EvaluationTask) -> int: ...
    def work_item(self, task: EvaluationTask) -> WorkItem | WorkResult: ...
    def commit(self, task: EvaluationTask, attempt: int, result: WorkResult) -> TaskOutcome: ...
    def stage(self, task: EvaluationTask, stage: Stage) -> None: ...


class ExecutionStrategy(ABC):
    """Executes independent evaluation tasks.

    Contract: tasks are independent; model/data failures become FAILED outcomes (never exceptions);
    PersistenceError is raised only when execution must stop; `should_stop` is polled between tasks
    and a stop lets already-started evaluations finish and be recorded.
    """

    @abstractmethod
    def execute(
        self,
        tasks: Sequence[EvaluationTask],
        runner: TaskExecutor,
        should_stop: Callable[[], bool],
    ) -> list[TaskOutcome]: ...

    def describe(self) -> str:
        return type(self).__name__
