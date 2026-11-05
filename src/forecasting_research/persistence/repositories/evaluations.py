from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

import pandas as pd

from ...domain.contracts import EvaluationTask, MetricValue, utcnow
from ...domain.exceptions import UnknownEntityError
from ...experiments.lifecycle import assert_evaluation_transition
from ..database import Database


class EvaluationRepository:
    def __init__(self, db: Database):
        self.db = db

    # ---- creation ----------------------------------------------------------------------
    def create_pending(self, tasks: Iterable[EvaluationTask]) -> int:
        """Insert PENDING rows (idempotent). Returns number newly created."""
        n = 0
        with self.db.transaction() as c:
            for t in tasks:
                cur = c.execute(
                    """INSERT OR IGNORE INTO model_evaluations(evaluation_id,experiment_id,dataset_id,model_name,status)
                       VALUES (?,?,?,?, 'PENDING')""",
                    (t.evaluation_id, t.experiment_id, t.dataset_id, t.model_name),
                )
                n += cur.rowcount
        return n

    # ---- reads -------------------------------------------------------------------------
    def get(self, evaluation_id: str) -> dict:
        with self.db.connect() as c:
            row = c.execute("SELECT * FROM model_evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()
        if row is None:
            raise UnknownEntityError(f"Unknown evaluation: {evaluation_id}")
        d = dict(row)
        d["parameters"] = json.loads(d["parameters_json"]) if d.get("parameters_json") else None
        d["warnings"] = json.loads(d["warnings_json"]) if d.get("warnings_json") else []
        return d

    def list_for_experiment(self, experiment_id: str, statuses: Iterable[str] | None = None) -> list[dict]:
        sql = "SELECT * FROM model_evaluations WHERE experiment_id=?"
        args: list = [experiment_id]
        if statuses is not None:
            st = list(statuses)
            sql += f" AND status IN ({','.join('?' * len(st)) or 'NULL'})"
            args += st
        sql += " ORDER BY rowid"
        with self.db.connect() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def counts(self, experiment_id: str) -> dict[str, int]:
        out = {"PENDING": 0, "RUNNING": 0, "COMPLETED": 0, "FAILED": 0, "INTERRUPTED": 0}
        with self.db.connect() as c:
            for r in c.execute(
                "SELECT status, COUNT(*) n FROM model_evaluations WHERE experiment_id=? GROUP BY status", (experiment_id,)
            ):
                out[r["status"]] = r["n"]
        out["TOTAL"] = sum(out.values())
        return out

    def attempts(self, evaluation_id: str) -> list[dict]:
        with self.db.connect() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM evaluation_attempts WHERE evaluation_id=? ORDER BY attempt_number", (evaluation_id,))]

    def metrics(self, evaluation_id: str) -> dict[str, MetricValue]:
        from ...domain.enums import MetricStatus

        with self.db.connect() as c:
            return {
                r["metric_name"]: MetricValue(r["value"], MetricStatus(r["status"]))
                for r in c.execute("SELECT * FROM evaluation_metrics WHERE evaluation_id=?", (evaluation_id,))
            }

    def completed_artifact_refs(self) -> list[tuple[str, str]]:
        with self.db.connect() as c:
            return [(r[0], r[1]) for r in c.execute(
                "SELECT artifact_path, artifact_sha256 FROM model_evaluations "
                "WHERE status='COMPLETED' AND artifact_path IS NOT NULL")]

    # ---- state transitions ---------------------------------------------------------------
    def begin_attempt(self, evaluation_id: str) -> int:
        """-> RUNNING; records a new attempt row (history is never overwritten). Returns attempt number."""
        with self.db.transaction() as c:
            row = c.execute("SELECT status, attempts FROM model_evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()
            if row is None:
                raise UnknownEntityError(f"Unknown evaluation: {evaluation_id}")
            assert_evaluation_transition(row["status"], "RUNNING")
            attempt = row["attempts"] + 1
            now = utcnow()
            c.execute(
                """UPDATE model_evaluations SET status='RUNNING', attempts=?, started_at=?, finished_at=NULL,
                   current_stage=NULL, failure_category=NULL, error_summary=NULL WHERE evaluation_id=?""",
                (attempt, now, evaluation_id),
            )
            c.execute(
                "INSERT INTO evaluation_attempts(evaluation_id,attempt_number,status,started_at) VALUES (?,?,?,?)",
                (evaluation_id, attempt, "RUNNING", now),
            )
            return attempt

    def set_stage(self, evaluation_id: str, stage: str) -> None:
        with self.db.transaction() as c:
            c.execute("UPDATE model_evaluations SET current_stage=? WHERE evaluation_id=?", (stage, evaluation_id))

    def complete(
        self,
        evaluation_id: str,
        attempt: int,
        *,
        parameters: dict,
        metrics: Mapping[str, MetricValue],
        warnings: list[str],
        artifact_path: str,
        artifact_sha256: str,
        runtime_seconds: float,
        train_size: int,
        test_size: int,
        horizon: int,
    ) -> None:
        """Atomically mark COMPLETED together with metrics. Call only after the artifact is safely written."""
        with self.db.transaction() as c:
            row = c.execute("SELECT status FROM model_evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()
            assert_evaluation_transition(row["status"], "COMPLETED")
            now = utcnow()
            c.execute(
                """UPDATE model_evaluations SET status='COMPLETED', current_stage='complete', parameters_json=?,
                   warnings_json=?, train_size=?, test_size=?, horizon=?, artifact_path=?, artifact_sha256=?,
                   runtime_seconds=?, finished_at=?, failure_category=NULL, error_summary=NULL WHERE evaluation_id=?""",
                (json.dumps(parameters, sort_keys=True), json.dumps(warnings), train_size, test_size, horizon,
                 artifact_path, artifact_sha256, runtime_seconds, now, evaluation_id),
            )
            c.execute("DELETE FROM evaluation_metrics WHERE evaluation_id=?", (evaluation_id,))
            c.executemany(
                "INSERT INTO evaluation_metrics(evaluation_id,metric_name,value,status) VALUES (?,?,?,?)",
                [(evaluation_id, k, v.value, v.status.value) for k, v in metrics.items()],
            )
            c.execute(
                "UPDATE evaluation_attempts SET status='COMPLETED', finished_at=?, runtime_seconds=? "
                "WHERE evaluation_id=? AND attempt_number=?",
                (now, runtime_seconds, evaluation_id, attempt),
            )

    def fail(self, evaluation_id: str, attempt: int, category: str, details: str, runtime_seconds: float) -> None:
        with self.db.transaction() as c:
            row = c.execute("SELECT status FROM model_evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()
            assert_evaluation_transition(row["status"], "FAILED")
            now = utcnow()
            c.execute(
                """UPDATE model_evaluations SET status='FAILED', failure_category=?, error_summary=?, runtime_seconds=?,
                   finished_at=? WHERE evaluation_id=?""",
                (category, details.splitlines()[0][:300] if details else "", runtime_seconds, now, evaluation_id),
            )
            c.execute(
                "UPDATE evaluation_attempts SET status='FAILED', failure_category=?, error_details=?, finished_at=?, "
                "runtime_seconds=? WHERE evaluation_id=? AND attempt_number=?",
                (category, details[:8000], now, runtime_seconds, evaluation_id, attempt),
            )

    def mark_running_interrupted(self, experiment_id: str) -> int:
        """RUNNING -> INTERRUPTED (user interrupt, crash recovery). Returns rows changed."""
        with self.db.transaction() as c:
            ids = [r[0] for r in c.execute(
                "SELECT evaluation_id FROM model_evaluations WHERE experiment_id=? AND status='RUNNING'", (experiment_id,))]
            now = utcnow()
            for eid in ids:
                c.execute("UPDATE model_evaluations SET status='INTERRUPTED', finished_at=? WHERE evaluation_id=?", (now, eid))
                c.execute(
                    "UPDATE evaluation_attempts SET status='INTERRUPTED', finished_at=? WHERE evaluation_id=? AND status='RUNNING'",
                    (now, eid),
                )
            return len(ids)

    # ---- analysis queries ----------------------------------------------------------------
    def parameter_rows(self, experiment_ids: list[str] | None = None, model_name: str | None = None) -> pd.DataFrame:
        """One row per evaluation of a model (all statuses) with its parsed selected-parameters document."""
        where, args = [], []
        if experiment_ids:
            where.append(f"e.experiment_id IN ({','.join('?' * len(experiment_ids))})")
            args += list(experiment_ids)
        if model_name:
            where.append("e.model_name=?")
            args.append(model_name)
        sql = f"""SELECT e.evaluation_id, e.experiment_id, e.dataset_id, e.model_name, e.status, e.runtime_seconds,
                         e.parameters_json, d.seasonality_type, d.seasonal_period, d.trend_type, d.noise_level, d.length
                  FROM model_evaluations e JOIN datasets d ON d.dataset_id=e.dataset_id
                  {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY e.experiment_id, e.rowid"""
        with self.db.connect() as c:
            df = pd.read_sql_query(sql, c, params=args)
        df["parameters"] = [json.loads(x) if x else None for x in df.pop("parameters_json")]
        return df

    def results_table(self, experiment_ids: list[str] | None = None) -> pd.DataFrame:
        """One row per evaluation (all statuses) joined with dataset factors and wide metrics."""
        where, args = "", []
        if experiment_ids:
            where = f"WHERE e.experiment_id IN ({','.join('?' * len(experiment_ids))})"
            args = list(experiment_ids)
        sql = f"""SELECT e.evaluation_id, e.experiment_id, e.dataset_id, e.model_name, e.status, e.failure_category,
                         e.error_summary, e.runtime_seconds, e.attempts, e.horizon, e.train_size,
                         d.seasonality_type, d.seasonal_period, d.trend_type, d.noise_level, d.length, d.replication
                  FROM model_evaluations e JOIN datasets d ON d.dataset_id=e.dataset_id {where}
                  ORDER BY e.experiment_id, e.rowid"""
        msql = f"""SELECT m.evaluation_id, m.metric_name, m.value, m.status
                   FROM evaluation_metrics m JOIN model_evaluations e ON e.evaluation_id=m.evaluation_id {where}"""
        with self.db.connect() as c:
            base = pd.read_sql_query(sql, c, params=args)
            met = pd.read_sql_query(msql, c, params=args)
        if met.empty:
            for name in ("MAE", "MSE", "RMSE", "MAPE"):
                base[name] = float("nan")
            base["MAPE_status"] = None
            return base
        wide = met.pivot(index="evaluation_id", columns="metric_name", values="value")
        wide = wide.reindex(columns=["MAE", "MSE", "RMSE", "MAPE"])
        mape_status = met[met.metric_name == "MAPE"].set_index("evaluation_id")["status"].rename("MAPE_status")
        out = base.merge(wide, left_on="evaluation_id", right_index=True, how="left")
        out = out.merge(mape_status, left_on="evaluation_id", right_index=True, how="left")
        return out
