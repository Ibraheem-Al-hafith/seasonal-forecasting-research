"""SQLite access: connections, transactions, versioned migrations and integrity checks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..domain.exceptions import PersistenceError

# Numbered migrations, tracked with PRAGMA user_version. Append only; never edit a shipped one.
MIGRATIONS: list[str] = [
    # ---- v1 ------------------------------------------------------------------------------
    """
    CREATE TABLE dataset_batches (
        batch_id      TEXT PRIMARY KEY,
        name          TEXT,
        config_json   TEXT NOT NULL,
        status        TEXT NOT NULL,
        dataset_count INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT NOT NULL,
        finished_at   TEXT
    );
    CREATE TABLE datasets (
        dataset_id        TEXT PRIMARY KEY,
        batch_id          TEXT NOT NULL REFERENCES dataset_batches(batch_id),
        seasonality_type  TEXT NOT NULL,
        seasonal_period   INTEGER NOT NULL,
        trend_type        TEXT NOT NULL,
        noise_level       TEXT NOT NULL,
        length            INTEGER NOT NULL,
        replication       INTEGER NOT NULL,
        seed              TEXT NOT NULL,
        metadata_json     TEXT NOT NULL,
        observed_path     TEXT NOT NULL,
        truth_path        TEXT NOT NULL,
        observed_sha256   TEXT NOT NULL,
        truth_sha256      TEXT NOT NULL,
        generator_version TEXT NOT NULL,
        created_at        TEXT NOT NULL
    );
    CREATE INDEX idx_datasets_batch ON datasets(batch_id);
    CREATE INDEX idx_datasets_factors
        ON datasets(seasonality_type, seasonal_period, trend_type, noise_level, length);
    CREATE TABLE batch_datasets (
        batch_id   TEXT NOT NULL REFERENCES dataset_batches(batch_id),
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        PRIMARY KEY (batch_id, dataset_id)
    );
    CREATE TABLE experiments (
        experiment_id    TEXT PRIMARY KEY,
        name             TEXT,
        batch_id         TEXT REFERENCES dataset_batches(batch_id),
        config_json      TEXT NOT NULL,
        environment_json TEXT NOT NULL,
        status           TEXT NOT NULL,
        outcome          TEXT NOT NULL DEFAULT 'NONE',
        created_at       TEXT NOT NULL,
        started_at       TEXT,
        completed_at     TEXT
    );
    CREATE TABLE experiment_datasets (
        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
        dataset_id    TEXT NOT NULL REFERENCES datasets(dataset_id),
        PRIMARY KEY (experiment_id, dataset_id)
    );
    CREATE TABLE model_evaluations (
        evaluation_id    TEXT PRIMARY KEY,
        experiment_id    TEXT NOT NULL REFERENCES experiments(experiment_id),
        dataset_id       TEXT NOT NULL REFERENCES datasets(dataset_id),
        model_name       TEXT NOT NULL,
        status           TEXT NOT NULL,
        current_stage    TEXT,
        failure_category TEXT,
        error_summary    TEXT,
        parameters_json  TEXT,
        warnings_json    TEXT,
        train_size       INTEGER,
        test_size        INTEGER,
        horizon          INTEGER,
        artifact_path    TEXT,
        artifact_sha256  TEXT,
        runtime_seconds  REAL,
        attempts         INTEGER NOT NULL DEFAULT 0,
        started_at       TEXT,
        finished_at      TEXT,
        UNIQUE (experiment_id, dataset_id, model_name)
    );
    CREATE INDEX idx_eval_experiment_status ON model_evaluations(experiment_id, status);
    CREATE INDEX idx_eval_dataset ON model_evaluations(dataset_id);
    CREATE TABLE evaluation_attempts (
        attempt_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_id    TEXT NOT NULL REFERENCES model_evaluations(evaluation_id),
        attempt_number   INTEGER NOT NULL,
        status           TEXT NOT NULL,
        failure_category TEXT,
        error_details    TEXT,
        started_at       TEXT NOT NULL,
        finished_at      TEXT,
        runtime_seconds  REAL,
        UNIQUE (evaluation_id, attempt_number)
    );
    CREATE TABLE evaluation_metrics (
        evaluation_id TEXT NOT NULL REFERENCES model_evaluations(evaluation_id),
        metric_name   TEXT NOT NULL,
        value         REAL,
        status        TEXT NOT NULL,
        PRIMARY KEY (evaluation_id, metric_name)
    );
    CREATE TABLE progress_events (
        event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT,
        evaluation_id TEXT,
        event_type    TEXT NOT NULL,
        message       TEXT,
        created_at    TEXT NOT NULL
    );
    CREATE INDEX idx_events_experiment ON progress_events(experiment_id, event_id);
    """,
]


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ---- connections -----------------------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Short-lived connection (safe across threads: one connection per use)."""
        conn = None
        try:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as exc:
            raise PersistenceError(f"Database error ({self.path.name}): {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE ... COMMIT; rolls back on any exception."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            else:
                conn.execute("COMMIT")

    # ---- lifecycle -------------------------------------------------------------------
    def initialize(self) -> None:
        """Create the database file and apply any pending migrations."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PersistenceError(f"Cannot create database directory: {exc}") from exc
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > len(MIGRATIONS):
                raise PersistenceError(
                    f"Database schema v{version} is newer than this application (v{len(MIGRATIONS)})"
                )
            for i in range(version, len(MIGRATIONS)):
                conn.executescript(f"BEGIN;\n{MIGRATIONS[i]}\nPRAGMA user_version = {i + 1};\nCOMMIT;")

    def schema_version(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def integrity_problems(self) -> list[str]:
        """Empty list means healthy (PRAGMA integrity_check + foreign_key_check)."""
        problems: list[str] = []
        with self.connect() as conn:
            for row in conn.execute("PRAGMA integrity_check"):
                if row[0] != "ok":
                    problems.append(f"integrity_check: {row[0]}")
            for row in conn.execute("PRAGMA foreign_key_check"):
                problems.append(f"foreign_key_check: table={row[0]} rowid={row[1]} parent={row[2]}")
        return problems
