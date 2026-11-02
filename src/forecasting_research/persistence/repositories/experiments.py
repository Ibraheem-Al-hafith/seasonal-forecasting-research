from __future__ import annotations

import json

from ...domain.contracts import ProgressEvent, utcnow
from ...domain.exceptions import UnknownEntityError
from ..database import Database


class ExperimentRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_experiment(
        self, experiment_id: str, name: str, batch_id: str | None, config: dict, environment: dict, dataset_ids: list[str]
    ) -> None:
        with self.db.transaction() as c:
            c.execute(
                """INSERT INTO experiments(experiment_id,name,batch_id,config_json,environment_json,status,outcome,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (experiment_id, name, batch_id, json.dumps(config, sort_keys=True),
                 json.dumps(environment, sort_keys=True), "CREATED", "NONE", utcnow()),
            )
            c.executemany(
                "INSERT INTO experiment_datasets(experiment_id,dataset_id) VALUES (?,?)",
                [(experiment_id, d) for d in dataset_ids],
            )

    def set_status(
        self, experiment_id: str, status: str, outcome: str | None = None, *, mark_started: bool = False, mark_completed: bool = False
    ) -> None:
        sets, args = ["status=?"], [status]
        if outcome is not None:
            sets.append("outcome=?")
            args.append(outcome)
        if mark_started:
            sets.append("started_at=COALESCE(started_at, ?)")
            args.append(utcnow())
        if mark_completed:
            sets.append("completed_at=?")
            args.append(utcnow())
        elif status == "RUNNING":
            sets.append("completed_at=NULL")
        args.append(experiment_id)
        with self.db.transaction() as c:
            c.execute(f"UPDATE experiments SET {', '.join(sets)} WHERE experiment_id=?", args)

    def get(self, experiment_id: str) -> dict:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            raise UnknownEntityError(f"Unknown experiment: {experiment_id}")
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        d["environment"] = json.loads(d.pop("environment_json"))
        return d

    def list(self) -> list[dict]:
        with self.db.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT experiment_id,name,batch_id,status,outcome,created_at,started_at,completed_at "
                "FROM experiments ORDER BY created_at, experiment_id")]

    def dataset_ids(self, experiment_id: str) -> list[str]:
        with self.db.connect() as c:
            return [r[0] for r in c.execute(
                "SELECT dataset_id FROM experiment_datasets WHERE experiment_id=? ORDER BY dataset_id", (experiment_id,))]


class EventRepository:
    """Durable subset of progress events (meaningful state transitions only)."""

    def __init__(self, db: Database):
        self.db = db

    def add(self, ev: ProgressEvent) -> None:
        with self.db.transaction() as c:
            c.execute(
                "INSERT INTO progress_events(experiment_id,evaluation_id,event_type,message,created_at) VALUES (?,?,?,?,?)",
                (ev.experiment_id, ev.evaluation_id, ev.event_type, ev.message, ev.timestamp),
            )

    def recent(self, experiment_id: str, limit: int = 50) -> list[dict]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT * FROM progress_events WHERE experiment_id=? ORDER BY event_id DESC LIMIT ?",
                (experiment_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
