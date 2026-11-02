import sqlite3

import numpy as np
import pytest

from forecasting_research.domain.exceptions import ArtifactIntegrityError, PersistenceError
from forecasting_research.persistence.artifacts import ArtifactStore
from forecasting_research.persistence.backup import backup_database
from forecasting_research.persistence.database import Database


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path)


def test_roundtrip_and_no_pickle(store):
    sha = store.write_npz("artifacts/evaluations/x.npz", {"a": np.arange(5.0)}, allow_overwrite=True)
    d = store.read_npz("artifacts/evaluations/x.npz", sha)
    assert np.array_equal(d["a"], np.arange(5.0)) and int(d["schema_version"]) == 1


def test_no_temp_files_left(store):
    store.write_npz("artifacts/evaluations/x.npz", {"a": np.ones(3)}, allow_overwrite=True)
    assert store.list_temp_files() == []


def test_tamper_and_missing_detected(store, tmp_path):
    sha = store.write_npz("artifacts/evaluations/x.npz", {"a": np.ones(3)}, allow_overwrite=True)
    p = store.abs_path("artifacts/evaluations/x.npz")
    p.write_bytes(p.read_bytes() + b"junk")
    with pytest.raises(ArtifactIntegrityError):
        store.read_npz("artifacts/evaluations/x.npz", sha)
    p.unlink()
    with pytest.raises(ArtifactIntegrityError):
        store.read_npz("artifacts/evaluations/x.npz")


def test_dataset_artifacts_never_silently_overwritten(store):
    rel = "artifacts/datasets/d.observed.npz"
    store.write_npz(rel, {"y": np.arange(3.0)}, allow_overwrite=False)
    store.write_npz(rel, {"y": np.arange(3.0)}, allow_overwrite=False)  # identical: idempotent
    with pytest.raises(ArtifactIntegrityError):
        store.write_npz(rel, {"y": np.arange(4.0)}, allow_overwrite=False)


def test_write_failure_is_persistence_error(store, monkeypatch):
    monkeypatch.setattr(np, "savez_compressed", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(PersistenceError):
        store.write_npz("artifacts/evaluations/x.npz", {"a": np.ones(2)}, allow_overwrite=True)
    assert store.list_temp_files() == []


def test_database_migrations_and_integrity(tmp_path):
    db = Database(tmp_path / "db" / "r.sqlite")
    db.initialize()
    db.initialize()  # idempotent
    assert db.schema_version() == 1 and db.integrity_problems() == []
    with db.connect() as c:
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_transaction_rolls_back_and_wraps_errors(tmp_path):
    db = Database(tmp_path / "r.sqlite")
    db.initialize()
    with pytest.raises(RuntimeError):
        with db.transaction() as c:
            c.execute("INSERT INTO dataset_batches(batch_id,config_json,status,created_at) VALUES ('b','{}','X','t')")
            raise RuntimeError("boom")
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM dataset_batches").fetchone()[0] == 0
    with pytest.raises(PersistenceError):  # FK violation surfaces as PersistenceError
        with db.transaction() as c:
            c.execute("INSERT INTO experiment_datasets VALUES ('nope','nope')")


def test_backup_uses_sqlite_api(tmp_path):
    db = Database(tmp_path / "r.sqlite")
    db.initialize()
    with db.transaction() as c:
        c.execute("INSERT INTO dataset_batches(batch_id,config_json,status,created_at) VALUES ('b','{}','X','t')")
    dest = backup_database(db.path, tmp_path / "bk")
    assert sqlite3.connect(dest).execute("SELECT COUNT(*) FROM dataset_batches").fetchone()[0] == 1
