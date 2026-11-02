"""Repositories: the only place SQL is written. Other components never issue SQL."""

from __future__ import annotations

from ..database import Database
from .datasets import DatasetRepository
from .evaluations import EvaluationRepository
from .experiments import EventRepository, ExperimentRepository


class Repository:
    """Facade bundling all repositories over one database."""

    def __init__(self, db: Database):
        self.db = db
        self.datasets = DatasetRepository(db)
        self.experiments = ExperimentRepository(db)
        self.evaluations = EvaluationRepository(db)
        self.events = EventRepository(db)


__all__ = ["Repository", "DatasetRepository", "ExperimentRepository", "EvaluationRepository", "EventRepository"]
