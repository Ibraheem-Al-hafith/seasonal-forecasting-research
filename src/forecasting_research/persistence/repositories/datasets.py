from __future__ import annotations

import json
import sqlite3

from ...domain.contracts import DatasetRecord, utcnow
from ...domain.exceptions import UnknownEntityError
from ..database import Database


def _record(row: sqlite3.Row) -> DatasetRecord:
    d = dict(row)
    d["metadata"] = json.loads(d.pop("metadata_json"))
    return DatasetRecord(**d)


class DatasetRepository:
    def __init__(self, db: Database):
        self.db = db

    # ---- batches -----------------------------------------------------------------------
    def create_batch(self, batch_id: str, name: str, config: dict) -> None:
        with self.db.transaction() as c:
            c.execute(
                "INSERT INTO dataset_batches(batch_id,name,config_json,status,created_at) VALUES (?,?,?,?,?)",
                (batch_id, name, json.dumps(config, sort_keys=True), "RUNNING", utcnow()),
            )

    def finish_batch(self, batch_id: str, status: str) -> None:
        with self.db.transaction() as c:
            n = c.execute("SELECT COUNT(*) FROM batch_datasets WHERE batch_id=?", (batch_id,)).fetchone()[0]
            c.execute(
                "UPDATE dataset_batches SET status=?, dataset_count=?, finished_at=? WHERE batch_id=?",
                (status, n, utcnow(), batch_id),
            )

    def get_batch(self, batch_id: str) -> dict:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM dataset_batches WHERE batch_id=?", (batch_id,)).fetchone()
        if row is None:
            raise UnknownEntityError(f"Unknown dataset batch: {batch_id}")
        return dict(row)

    def list_batches(self) -> list[dict]:
        with self.db.connect() as c:
            return [dict(r) for r in c.execute("SELECT * FROM dataset_batches ORDER BY created_at, batch_id")]

    # ---- datasets ----------------------------------------------------------------------
    def dataset_exists(self, dataset_id: str) -> bool:
        with self.db.connect() as c:
            return c.execute("SELECT 1 FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone() is not None

    def add_dataset(self, rec: DatasetRecord, batch_id: str) -> bool:
        """Insert a dataset (immutable) and link it to a batch. Returns True if newly created."""
        with self.db.transaction() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO datasets(dataset_id,batch_id,seasonality_type,seasonal_period,
                   trend_type,noise_level,length,replication,seed,metadata_json,observed_path,truth_path,
                   observed_sha256,truth_sha256,generator_version,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.dataset_id, batch_id, rec.seasonality_type, rec.seasonal_period, rec.trend_type,
                 rec.noise_level, rec.length, rec.replication, rec.seed, json.dumps(rec.metadata, sort_keys=True),
                 rec.observed_path, rec.truth_path, rec.observed_sha256, rec.truth_sha256,
                 rec.generator_version, rec.created_at),
            )
            c.execute("INSERT OR IGNORE INTO batch_datasets(batch_id,dataset_id) VALUES (?,?)", (batch_id, rec.dataset_id))
            return cur.rowcount == 1

    def link_to_batch(self, batch_id: str, dataset_id: str) -> None:
        with self.db.transaction() as c:
            c.execute("INSERT OR IGNORE INTO batch_datasets(batch_id,dataset_id) VALUES (?,?)", (batch_id, dataset_id))

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if row is None:
            raise UnknownEntityError(f"Unknown dataset: {dataset_id}")
        return _record(row)

    def list_datasets(
        self,
        batch_id: str | None = None,
        seasonality_type: str | None = None,
        seasonal_period: int | None = None,
        trend_type: str | None = None,
        noise_level: str | None = None,
        length: int | None = None,
        dataset_ids: list[str] | None = None,
    ) -> list[DatasetRecord]:
        where, args = [], []
        for col, val in [("d.seasonality_type", seasonality_type), ("d.seasonal_period", seasonal_period),
                         ("d.trend_type", trend_type), ("d.noise_level", noise_level), ("d.length", length)]:
            if val is not None:
                where.append(f"{col}=?")
                args.append(val)
        join = ""
        if batch_id is not None:
            join = "JOIN batch_datasets b ON b.dataset_id=d.dataset_id"
            where.append("b.batch_id=?")
            args.append(batch_id)
        if dataset_ids is not None:
            where.append(f"d.dataset_id IN ({','.join('?' * len(dataset_ids)) or 'NULL'})")
            args += list(dataset_ids)
        sql = f"SELECT d.* FROM datasets d {join}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY d.seasonality_type, d.seasonal_period, d.trend_type, d.noise_level, d.length, d.replication, d.dataset_id"
        with self.db.connect() as c:
            return [_record(r) for r in c.execute(sql, args)]

    def count_datasets(self) -> int:
        with self.db.connect() as c:
            return int(c.execute("SELECT COUNT(*) FROM datasets").fetchone()[0])
