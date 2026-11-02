import sqlite3

import pytest
from typer.testing import CliRunner

from forecasting_research.cli.app import app as cli_app
from forecasting_research.domain.exceptions import DataError, PersistenceError, UnknownEntityError
from forecasting_research.monitoring import events as E
from tests.helpers import AlwaysFails, NanForecast, Registered

from ..conftest import make_cfg


def test_generation_only_mode(app):
    b = app.generate()
    assert app.repo.experiments.list() == [] and app.repo.datasets.get_batch(b)["dataset_count"] == 4


def test_full_experiment_events_and_progress(app):
    seen = []
    app.bus.subscribe(lambda ev: seen.append(ev.event_type))
    eid = app.run()
    snap = app.tracker.snapshot()
    assert snap["total"] == 12 and snap["completed"] == 12 and snap["percent"] == 100.0 and snap["pending"] == 0
    for t in (E.EXPERIMENT_STARTED, E.PLAN_BUILT, E.EVALUATION_STAGE, E.EVALUATION_COMPLETED, E.EXPERIMENT_FINISHED, E.GENERATION_FINISHED):
        assert t in seen
    assert len(app.repo.events.recent(eid)) > 12  # durable transitions persisted
    assert (app.workspace / "logs" / "experiments" / f"{eid}.log").exists()
    assert (app.workspace / "logs" / "application.log").exists()


def test_existing_datasets_mode_with_filters_and_subset_methods(app):
    app.generate()
    cfg = make_cfg(forecasting={"methods": ["holt_winters"]})
    eid = app.evaluate(cfg, filters={"seasonality_type": "additive"})
    c = app.repo.evaluations.counts(eid)
    assert c["TOTAL"] == 2 and c["COMPLETED"] == 2
    one = app.repo.datasets.list_datasets()[0].dataset_id
    eid2 = app.evaluate(make_cfg(forecasting={"methods": ["classical_decomposition"]}), dataset_ids=[one])
    assert app.repo.evaluations.counts(eid2)["TOTAL"] == 1


def test_evaluate_errors(app):
    with pytest.raises(DataError):
        app.evaluate()  # nothing generated
    with pytest.raises(UnknownEntityError):
        app.evaluate(dataset_ids=["ds_nope"])


def test_interrupt_then_resume_skips_completed(app):
    done = []

    def stopper(ev):
        if ev.event_type == E.EVALUATION_COMPLETED:
            done.append(ev.evaluation_id)
            if len(done) == 5:
                app.stop()

    app.bus.subscribe(stopper)
    eid = app.run()
    d = app.experiment_details(eid)
    assert d["status"] == "INTERRUPTED" and d["outcome"] == "INCOMPLETE" and d["counts"]["COMPLETED"] == 5
    before = {e["evaluation_id"]: e["attempts"] for e in app.repo.evaluations.list_for_experiment(eid)}
    app.resume(eid)
    d = app.experiment_details(eid)
    assert d["status"] == "COMPLETED" and d["counts"]["COMPLETED"] == 12
    after = {e["evaluation_id"]: e["attempts"] for e in app.repo.evaluations.list_for_experiment(eid)}
    assert all(after[i] == 1 for i in done), "completed evaluations must not be repeated"
    assert sum(1 for i in after if before[i] == 0 and after[i] == 1) == 7
    assert app.repo.experiments.get(eid)["completed_at"]


def test_crash_leaves_running_rows_that_resume_recovers(app):
    eid = app.run()
    victim = app.repo.evaluations.list_for_experiment(eid)[3]["evaluation_id"]
    with sqlite3.connect(app.db.path) as c:  # simulate kill -9 mid-evaluation
        c.execute("UPDATE model_evaluations SET status='RUNNING' WHERE evaluation_id=?", (victim,))
        c.execute("UPDATE experiments SET status='RUNNING', outcome='NONE' WHERE experiment_id=?", (eid,))
    app.resume(eid)
    ev = app.repo.evaluations.get(victim)
    assert ev["status"] == "COMPLETED" and ev["attempts"] == 2
    assert app.experiment_details(eid)["status"] == "COMPLETED"
    hist = app.repo.evaluations.attempts(victim)
    assert [h["status"] for h in hist] == ["COMPLETED", "COMPLETED"]


def test_model_failures_are_isolated_reported_and_retryable(app):
    with Registered(always_fails=AlwaysFails, nan_forecast=NanForecast):
        cfg = make_cfg(forecasting={"methods": ["holt_winters", "always_fails", "nan_forecast"]})
        app.generate()
        eid = app.evaluate(cfg)
        d = app.experiment_details(eid)
        assert d["status"] == "COMPLETED" and d["outcome"] == "PARTIAL" and "partial" in d["status_display"]
        assert d["counts"]["COMPLETED"] == 4 and d["counts"]["FAILED"] == 8
        t = app.evaluations_table(eid)
        cats = set(t[t.status == "FAILED"]["failure"])
        assert cats == {"MODEL_FIT_ERROR", "INVALID_FORECAST"}
        # failures are visible in analysis, never silently dropped
        agg = app.aggregate([eid], ["seasonality"]).set_index(["seasonality_type", "model_name"])
        assert agg.loc[("additive", "always_fails"), "n_failed"] == 2 and agg.loc[("additive", "always_fails"), "n_completed"] == 0
        best = app.best_methods([eid], ["seasonality"], ["RMSE"])
        assert not best["complete_coverage"].any() and (best["best_model"] == "holt_winters").all()
        # default resume must NOT retry permanent model failures
        app.resume(eid)
        assert app.repo.evaluations.counts(eid)["FAILED"] == 8
        failed_id = t[t.model == "always_fails"].iloc[0]["evaluation_id"]
        assert app.repo.evaluations.get(failed_id)["attempts"] == 1
    # the method is "fixed" (now a working one under a registered name) -> explicit retry, history preserved
    from tests.helpers import Naive
    with Registered(always_fails=Naive, nan_forecast=Naive):
        app.resume(eid, retry_failed=True)
        assert app.repo.evaluations.counts(eid)["COMPLETED"] == 12
        hist = app.repo.evaluations.attempts(failed_id)
        assert [h["status"] for h in hist] == ["FAILED", "COMPLETED"] and "synthetic model failure" in hist[0]["error_details"]
    assert app.experiment_details(eid)["outcome"] == "SUCCESS"


def test_database_failure_stops_safely_then_resumes(app, monkeypatch):
    orig = app.repo.evaluations.complete
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise PersistenceError("simulated database failure")
        return orig(*a, **k)

    monkeypatch.setattr(app.repo.evaluations, "complete", flaky)
    with pytest.raises(PersistenceError):
        app.run()
    eid = app.repo.experiments.list()[-1]["experiment_id"]
    d = app.experiment_details(eid)
    assert d["status"] == "FAILED" and d["outcome"] == "ABORTED"
    assert d["counts"]["COMPLETED"] == 2 and d["counts"]["TOTAL"] == 12  # stopped: nothing after the failure ran
    monkeypatch.undo()
    app.resume(eid)  # persistence-failed evaluation is retried automatically (infrastructure, not model failure)
    assert app.experiment_details(eid)["counts"]["COMPLETED"] == 12


def test_artifact_write_failure_never_marks_completed(app, monkeypatch):
    def boom(*a, **k):
        raise PersistenceError("disk full")

    monkeypatch.setattr(app.store, "save_evaluation", boom)
    with pytest.raises(PersistenceError):
        app.run()
    eid = app.repo.experiments.list()[-1]["experiment_id"]
    assert app.repo.evaluations.counts(eid)["COMPLETED"] == 0
    assert app.repo.evaluations.completed_artifact_refs() == []


def test_missing_artifact_not_trusted_on_resume(app):
    eid = app.run()
    ev = app.repo.evaluations.list_for_experiment(eid)[2]
    app.store.abs_path(ev["artifact_path"]).unlink()
    assert not app.check_workspace().ok and app.check_workspace().missing_artifacts == [ev["artifact_path"]]
    app.resume(eid)
    again = app.repo.evaluations.get(ev["evaluation_id"])
    assert again["attempts"] == 2 and app.store.abs_path(again["artifact_path"]).exists()
    assert app.check_workspace().ok


def test_corrupt_dataset_fails_only_its_evaluations(app):
    b = app.generate()
    rec = app.repo.datasets.list_datasets(batch_id=b)[0]
    app.store.abs_path(rec.observed_path).write_bytes(b"garbage")
    eid = app.evaluate()
    d = app.experiment_details(eid)
    assert d["counts"]["FAILED"] == 3 and d["counts"]["COMPLETED"] == 9 and d["outcome"] == "PARTIAL"
    t = app.evaluations_table(eid)
    assert set(t[t.dataset_id == rec.dataset_id]["failure"]) == {"DATA_ERROR"}


def test_export_matches_stored_results(app, tmp_path):
    import pandas as pd
    eid = app.run()
    p = app.export_results(eid, "evaluations")
    df = pd.read_csv(p)
    assert len(df) == 12 and set(df.status) == {"COMPLETED"}
    ev = app.repo.evaluations.list_for_experiment(eid)[0]
    row = df[df.evaluation_id == ev["evaluation_id"]].iloc[0]
    assert row["RMSE"] == pytest.approx(app.repo.evaluations.metrics(ev["evaluation_id"])["RMSE"].value)
    fc = pd.read_csv(app.export_results(eid, "forecasts"))
    assert len(fc) == 12 * 4
    for kind in ("aggregate", "best"):
        assert len(pd.read_csv(app.export_results(eid, kind, ["noise"]))) > 0


def test_backup_and_workspace_check(app):
    app.run()
    assert app.backup().exists() and app.check_workspace().ok


# ---------------------------------------------------------------- CLI
YAML = """
generation: {seasonality_types: [additive], seasonal_periods: [4], trends: [linear], noise_levels: [low], dataset_lengths: [100], replications: 2}
forecasting: {sarima: {max_pq_sum: 1, max_seasonal_pq_sum: 0}}
"""


def test_cli_end_to_end(tmp_path):
    r = CliRunner()
    cfg = tmp_path / "c.yaml"
    cfg.write_text(YAML)
    base = ["-c", str(cfg), "-w", str(tmp_path / "ws")]
    res = r.invoke(cli_app, base + ["run", "--no-progress"])
    assert res.exit_code == 0, res.output
    assert "COMPLETED" in res.output
    eid = next(w for w in res.output.split() if w.startswith("exp_"))
    for args, needle in [
        (["experiments", "list"], eid[:12]), (["experiments", "status", eid], "Progress"),
        (["experiments", "show", eid], "Evaluations"), (["datasets", "list"], "Datasets"),
        (["analyze", eid, "--group-by", "noise", "-m", "RMSE"], "Best method"),
        (["analyze", eid, "--view", "wins"], "won"),
        (["export", eid, "--kind", "aggregate"], "Wrote"), (["plot", eid, "--dimension", "noise"], "Wrote"),
        (["workspace", "check"], "OK"), (["workspace", "backup"], "Backup"), (["workspace", "init"], "ready"),
    ]:
        out = r.invoke(cli_app, base + args)
        assert out.exit_code == 0 and needle in out.output, (args, out.output)
    ds = r.invoke(cli_app, base + ["datasets", "list"]).output
    assert r.invoke(cli_app, base + ["evaluate", "--methods", "holt_winters", "--no-progress", "--noise", "low"]).exit_code == 0
    assert r.invoke(cli_app, base + ["resume", eid, "--no-progress"]).exit_code == 0
    assert ds


def test_cli_version_and_bare_help():
    r = CliRunner()
    out = r.invoke(cli_app, ["--version"])
    assert out.exit_code == 0 and "forecasting-research" in out.output
    out = r.invoke(cli_app, [])
    assert "Commands" in out.output  # bare invocation prints help


def test_cli_errors_are_clean(tmp_path):
    r = CliRunner()
    bad = tmp_path / "bad.yaml"
    bad.write_text("generation: {replications: 0}")
    out = r.invoke(cli_app, ["-c", str(bad), "-w", str(tmp_path / "w"), "generate"])
    assert out.exit_code == 1 and "Invalid configuration" in out.output and "Traceback" not in out.output
    out = r.invoke(cli_app, ["-w", str(tmp_path / "w"), "experiments", "show", "exp_nope"])
    assert out.exit_code == 1 and "Unknown experiment" in out.output
    out = r.invoke(cli_app, ["-w", str(tmp_path / "w"), "run", "--methods", "nope", "--no-progress"])
    assert out.exit_code == 1 and "Unknown forecasting method" in out.output


def test_gui_controller_uses_shared_services(app):
    pytest.importorskip("gradio")
    from forecasting_research.gui.app import GuiController, build_ui
    build_ui(app)
    c = GuiController(app)
    assert "Configuration applied" in c.apply_config("generation: {replications: 1}")
    assert "Invalid" in c.apply_config("generation: {replications: 0}")
    import time
    assert "Started" in c.start_run()
    assert "already running" in c.start_run() or not app.is_running  # second start is refused while busy
    while app.is_running:
        time.sleep(0.2)
    head, table, _ = c.poll()
    assert "COMPLETED" in head and len(table) > 0
    eid = c.experiment_ids()[0]
    assert not c.best(eid, ["noise"], ["RMSE"]).empty and c.forecast_plot(eid, c.dataset_choices(eid)[0]) is not None
