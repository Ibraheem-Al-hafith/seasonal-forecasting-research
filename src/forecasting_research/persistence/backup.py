"""Safe SQLite backup (sqlite3 backup API; never a naive file copy) and workspace integrity checks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..domain.exceptions import PersistenceError
from .artifacts import ArtifactStore


def backup_database(db_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"
    src = dst = None
    try:
        src = sqlite3.connect(db_path)
        dst = sqlite3.connect(dest)
        src.backup(dst)
    except sqlite3.Error as exc:
        raise PersistenceError(f"Database backup failed: {exc}") from exc
    finally:
        for c in (dst, src):
            if c is not None:
                c.close()
    return dest


@dataclass
class IntegrityReport:
    database_problems: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    corrupt_artifacts: list[str] = field(default_factory=list)
    orphan_artifacts: list[str] = field(default_factory=list)
    temp_files: list[str] = field(default_factory=list)
    artifacts_checked: int = 0

    @property
    def ok(self) -> bool:
        return not (self.database_problems or self.missing_artifacts or self.corrupt_artifacts)

    def to_dict(self) -> dict:
        return {**self.__dict__, "ok": self.ok}


def check_workspace(db, repo, store: ArtifactStore) -> IntegrityReport:
    """Database integrity + referenced-artifact existence/checksum + orphan detection. Read-only."""
    rep = IntegrityReport(database_problems=db.integrity_problems())
    referenced: set[str] = set()
    refs: list[tuple[str, str]] = []
    for d in repo.datasets.list_datasets():
        refs += [(d.observed_path, d.observed_sha256), (d.truth_path, d.truth_sha256)]
    refs += repo.evaluations.completed_artifact_refs()
    for rel, sha in refs:
        referenced.add(rel)
        rep.artifacts_checked += 1
        if not store.abs_path(rel).is_file():
            rep.missing_artifacts.append(rel)
            continue
        try:
            store.verify(rel, sha)
        except PersistenceError:
            rep.corrupt_artifacts.append(rel)
    rep.orphan_artifacts = [a for a in store.list_artifacts() if a not in referenced]
    rep.temp_files = store.list_temp_files()
    return rep
