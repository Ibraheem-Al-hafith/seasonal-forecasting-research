"""NumPy NPZ artifact store with atomic, verified writes.

Safe sequence: write temp file -> re-read & validate -> atomic replace -> (caller commits DB reference).
Observed data and ground truth live in *separate* files so the forecasting path can never read truth.
No pickle is used anywhere (allow_pickle=False).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

from ..domain.contracts import GeneratedDataset, GroundTruth
from ..domain.exceptions import ArtifactIntegrityError, PersistenceError

SCHEMA_VERSION = 1
TMP_PREFIX = ".tmp-"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)

    # ---- path helpers (relative POSIX paths are what the database stores) --------------
    @staticmethod
    def observed_rel(dataset_id: str) -> str:
        return f"artifacts/datasets/{dataset_id}.observed.npz"

    @staticmethod
    def truth_rel(dataset_id: str) -> str:
        return f"artifacts/datasets/{dataset_id}.truth.npz"

    @staticmethod
    def evaluation_rel(evaluation_id: str) -> str:
        return f"artifacts/evaluations/{evaluation_id}.npz"

    def abs_path(self, rel: str) -> Path:
        return self.workspace.joinpath(*PurePosixPath(rel).parts)

    # ---- low level ---------------------------------------------------------------------
    def write_npz(self, rel: str, arrays: dict[str, np.ndarray], *, allow_overwrite: bool) -> str:
        """Atomically write an NPZ and return its SHA-256.

        If the target exists and overwriting is disallowed, identical content is accepted
        (idempotent re-generation); different content raises ArtifactIntegrityError.
        """
        arrays = {"schema_version": np.array(SCHEMA_VERSION), **{k: np.asarray(v) for k, v in arrays.items()}}
        target = self.abs_path(rel)
        if target.exists() and not allow_overwrite:
            existing = self.read_npz(rel)
            if _same_arrays(existing, arrays):
                return sha256_file(target)
            raise ArtifactIntegrityError(f"Refusing to overwrite existing artifact with different content: {rel}")
        tmp: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=TMP_PREFIX, suffix=".npz")
            os.close(fd)
            np.savez_compressed(tmp, **arrays)
            with np.load(tmp, allow_pickle=False) as check:  # validate it can be read back
                if set(check.files) != set(arrays) or not all(np.array_equal(check[k], arrays[k]) for k in arrays):
                    raise ArtifactIntegrityError(f"Written artifact failed read-back validation: {rel}")
            os.replace(tmp, target)  # atomic on POSIX and Windows
            tmp = None
            return sha256_file(target)
        except ArtifactIntegrityError:
            raise
        except (OSError, ValueError) as exc:
            raise PersistenceError(f"Could not write artifact {rel}: {exc}") from exc
        finally:
            if tmp is not None and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def read_npz(self, rel: str, expected_sha: str | None = None) -> dict[str, np.ndarray]:
        path = self.abs_path(rel)
        if not path.is_file():
            raise ArtifactIntegrityError(f"Artifact missing: {rel}")
        if expected_sha is not None and sha256_file(path) != expected_sha:
            raise ArtifactIntegrityError(f"Artifact checksum mismatch (corrupt or modified): {rel}")
        try:
            with np.load(path, allow_pickle=False) as data:
                out = {k: data[k] for k in data.files}
        except Exception as exc:
            raise ArtifactIntegrityError(f"Artifact unreadable: {rel}: {exc}") from exc
        if "schema_version" not in out or int(out["schema_version"]) > SCHEMA_VERSION:
            raise ArtifactIntegrityError(f"Artifact has missing/unsupported schema version: {rel}")
        return out

    def verify(self, rel: str, expected_sha: str) -> None:
        self.read_npz(rel, expected_sha)

    # ---- datasets ----------------------------------------------------------------------
    def save_dataset(self, ds: GeneratedDataset) -> tuple[str, str, str, str]:
        """Returns (observed_rel, truth_rel, observed_sha, truth_sha). Never overwrites silently."""
        o_rel, t_rel = self.observed_rel(ds.dataset_id), self.truth_rel(ds.dataset_id)
        o_sha = self.write_npz(o_rel, {"y": ds.observed}, allow_overwrite=False)
        t_sha = self.write_npz(
            t_rel,
            {
                "trend": ds.truth.trend,
                "seasonal": ds.truth.seasonal,
                "noise": ds.truth.noise,
                "noise_kind": np.array(ds.truth.noise_kind),
            },
            allow_overwrite=False,
        )
        return o_rel, t_rel, o_sha, t_sha

    def load_observed(self, rel: str, sha: str | None = None) -> np.ndarray:
        return self.read_npz(rel, sha)["y"].astype(float, copy=False)

    def load_truth(self, rel: str, sha: str | None = None) -> GroundTruth:
        d = self.read_npz(rel, sha)
        return GroundTruth(d["trend"], d["seasonal"], d["noise"], str(d["noise_kind"]))

    # ---- evaluations -------------------------------------------------------------------
    def save_evaluation(self, evaluation_id: str, actual: np.ndarray, forecast: np.ndarray) -> tuple[str, str]:
        rel = self.evaluation_rel(evaluation_id)
        return rel, self.write_npz(rel, {"actual": actual, "forecast": forecast}, allow_overwrite=True)

    def load_evaluation(self, rel: str, sha: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = self.read_npz(rel, sha)
        return d["actual"], d["forecast"]

    # ---- maintenance -------------------------------------------------------------------
    def list_artifacts(self) -> list[str]:
        root = self.workspace / "artifacts"
        if not root.is_dir():
            return []
        return sorted(
            p.relative_to(self.workspace).as_posix()
            for p in root.rglob("*.npz")
            if not p.name.startswith(TMP_PREFIX)
        )

    def list_temp_files(self) -> list[str]:
        root = self.workspace / "artifacts"
        if not root.is_dir():
            return []
        return sorted(p.relative_to(self.workspace).as_posix() for p in root.rglob(f"{TMP_PREFIX}*"))


def _same_arrays(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
