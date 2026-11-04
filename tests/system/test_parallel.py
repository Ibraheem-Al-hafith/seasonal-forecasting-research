import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from forecasting_research.application.services import ResearchApp
from forecasting_research.cli.app import app as cli_app
from forecasting_research.domain.exceptions import PersistenceError
from forecasting_research.monitoring import events as E

from ..conftest import make_cfg

PAR = {"strategy": "parallel", "workers": 2, "worker_modules": ["tests.parallel_methods"]}


def pcfg(**over):
    return make_cfg(execution=dict(PAR, **over.pop("execution", {})), **over)


@pytest.fixture
def papp(tmp_path):
    return ResearchApp(tmp_path / "ws", pcfg())


def _no_children_left():
    deadline = time.time() + 5
    while mp.active_children() and time.time() < deadline:
        time.sleep(0.1)
    return not mp.active_children()


def test_parallel_results_identical_to_sequential(tmp_path):
    seq = ResearchApp(tmp_path / "seq", make_cfg())
    par = ResearchApp(tmp_path / "par", pcfg())
    e1, e2 = seq.run(), par.run()
    cols = ["dataset_id", "model_name", "status", "MAE", "MSE", "RMSE", "MAPE", "horizon"]
    a = seq.results([e1]).sort_values(["dataset_id", "model_name"])[cols].reset_index(drop=True)
    b = par.results([e2]).sort_values(["dataset_id", "model_name"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # same data + same code => same numbers, whatever the execution mode
    d = par.experiment_details(e2)
    assert d["status"] == "COMPLETED" and d["outcome"] == "SUCCESS" and d["counts"]["COMPLETED"] == 12
    assert {e["attempts"] for e in par.repo.evaluations.list_for_experiment(e2)} == {1}
    assert par.check_workspace().ok and _no_children_left()


def test_parallel_progress_overlap_and_logs(papp):
    seen_running, stages = [], set()

    def watch(ev):
        seen_running.append(papp.tracker.snapshot()["running"])
        if ev.event_type == E.EVALUATION_STAGE:
            stages.add(ev.stage)

    papp.bus.subscribe(watch)
    eid = papp.run()
    assert max(seen_running) == 2  # two evaluations genuinely in flight at once
    assert "persist_results" in stages
    snap = papp.tracker.snapshot()
    assert snap["completed"] == 12 and snap["percent"] == 100.0 and snap["running"] == 0
    started = [r["message"] for r in papp.repo.events.recent(eid, 200) if r["event_type"] == E.EXPERIMENT_STARTED]
    assert "parallel (2 workers, spawn)" in started[0]


def test_parallel_failure_isolation(tmp_path):
    app = ResearchApp(tmp_path / "ws", pcfg(forecasting={"methods": ["holt_winters", "par_always_fails", "par_nan"]}))
    eid = app.run()
    d = app.experiment_details(eid)
    assert d["status"] == "COMPLETED" and d["outcome"] == "PARTIAL" and "partial" in d["status_display"]
    assert d["counts"]["COMPLETED"] == 4 and d["counts"]["FAILED"] == 8
    t = app.evaluations_table(eid)
    assert set(t[t.status == "FAILED"]["failure"]) == {"MODEL_FIT_ERROR", "INVALID_FORECAST"}
    assert "synthetic model failure" in app.repo.evaluations.attempts(t[t.model == "par_always_fails"].iloc[0]["evaluation_id"])[0]["error_details"]


def test_parallel_stop_then_resume(papp):
    done = []

    def stopper(ev):
        if ev.event_type == E.EVALUATION_COMPLETED:
            done.append(ev.evaluation_id)
            if len(done) == 3:
                papp.stop()

    papp.bus.subscribe(stopper)
    eid = papp.run()
    d = papp.experiment_details(eid)
    assert d["status"] == "INTERRUPTED" and d["outcome"] == "INCOMPLETE"
    assert d["counts"]["RUNNING"] == 0 and 3 <= d["counts"]["COMPLETED"] < 12  # in-flight work finished and was recorded
    finished = [e["evaluation_id"] for e in papp.repo.evaluations.list_for_experiment(eid, ["COMPLETED"])]
    papp.resume(eid)  # resumes in parallel (settings come from the loaded config)
    d = papp.experiment_details(eid)
    assert d["status"] == "COMPLETED" and d["counts"]["COMPLETED"] == 12
    assert all(papp.repo.evaluations.get(i)["attempts"] == 1 for i in finished), "completed work must not be repeated"
    assert _no_children_left()


def test_parallel_persistence_failure_stops_safely_and_kills_workers(papp, monkeypatch):
    orig, calls = papp.repo.evaluations.complete, {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise PersistenceError("simulated database failure")
        return orig(*a, **k)

    monkeypatch.setattr(papp.repo.evaluations, "complete", flaky)
    with pytest.raises(PersistenceError):
        papp.run()
    eid = papp.repo.experiments.list()[-1]["experiment_id"]
    d = papp.experiment_details(eid)
    assert d["status"] == "FAILED" and d["outcome"] == "ABORTED"
    assert d["counts"]["RUNNING"] == 0 and d["counts"]["COMPLETED"] == 2
    assert _no_children_left(), "worker processes must not outlive an aborted experiment"
    monkeypatch.undo()
    papp.resume(eid)
    assert papp.experiment_details(eid)["counts"]["COMPLETED"] == 12


def test_crashed_worker_fails_only_inflight_evaluations(tmp_path):
    app = ResearchApp(tmp_path / "ws", pcfg(forecasting={"methods": ["par_crash"]}, execution={"max_worker_restarts": 5}))
    eid = app.run()  # must not hang or raise even though every worker dies
    d = app.experiment_details(eid)
    assert d["counts"]["RUNNING"] == 0 and d["counts"]["FAILED"] == 4 and d["status"] == "FAILED"
    t = app.evaluations_table(eid)
    assert set(t["failure"]) == {"UNEXPECTED"} and "Worker process terminated" in t.iloc[0]["error"]
    assert _no_children_left()


def test_worker_recycling_option(tmp_path):
    import sys
    if sys.version_info < (3, 11):
        pytest.skip("max_tasks_per_worker needs Python 3.11+")
    app = ResearchApp(tmp_path / "ws", pcfg(execution={"max_tasks_per_worker": 3}))
    eid = app.run()
    assert app.experiment_details(eid)["counts"]["COMPLETED"] == 12 and _no_children_left()


YAML = """
generation: {seasonality_types: [additive], seasonal_periods: [4], trends: [linear], noise_levels: [low], dataset_lengths: [100], replications: 2}
forecasting: {methods: [classical_decomposition, holt_winters]}
"""


def test_cli_workers_option_run_and_resume(tmp_path):
    r, cfg = CliRunner(), tmp_path / "c.yaml"
    cfg.write_text(YAML)
    base = ["-c", str(cfg), "-w", str(tmp_path / "ws")]
    out = r.invoke(cli_app, base + ["run", "-j", "2", "--no-progress"])
    assert out.exit_code == 0 and "parallel (2 workers" in out.output, out.output
    eid = next(w for w in out.output.split() if w.startswith("exp_"))
    out = r.invoke(cli_app, base + ["resume", eid, "-j", "1", "--no-progress"])
    assert out.exit_code == 0 and "Execution: sequential" in out.output, out.output
    out = r.invoke(cli_app, base + ["run", "--workers", "1", "--no-progress"])
    assert "execution: sequential" in out.output


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inspects /proc")
def test_idle_workers_exit_when_parent_is_killed(tmp_path):
    """kill -9 on the main process must not leave orphaned, memory-holding worker processes behind.

    Regression test: *idle* workers used to block forever on the call queue after the parent died.
    """
    script = tmp_path / "idle_pool.py"
    script.write_text(
        "import multiprocessing as mp, time\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from forecasting_research.execution.parallel import _init_worker\n"
        "if __name__ == '__main__':\n"
        "    ctx = mp.get_context('spawn')\n"
        "    pool = ProcessPoolExecutor(2, mp_context=ctx, initializer=_init_worker, initargs=(ctx.Queue(), []))\n"
        "    list(pool.map(abs, [-1, -2, -3, -4]))  # start the workers, then leave them idle\n"
        "    print('ready', flush=True)\n"
        "    time.sleep(600)\n")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)

    def children(pid):
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
        return [int(x) for x in out]

    def alive(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"  # zombie == dead
        except OSError:
            return False

    try:
        assert proc.stdout.readline().strip() == "ready"
        kids = children(proc.pid)
        assert len(kids) >= 2, f"expected worker + resource tracker processes, got {kids}"
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        deadline = time.time() + 20
        while time.time() < deadline and any(alive(p) for p in kids):
            time.sleep(0.5)
        survivors = [p for p in kids if alive(p)]
        assert not survivors, f"orphaned processes survived the death of the parent: {survivors}"
    finally:
        proc.kill()


def test_unspawnable_main_falls_back_to_sequential(tmp_path, monkeypatch):
    """`python -` (stdin) mains cannot be re-imported by spawned workers: run sequentially with a clear warning."""
    monkeypatch.setattr(sys.modules["__main__"], "__file__", "<stdin>", raising=False)
    app = ResearchApp(tmp_path / "ws", pcfg())
    eid = app.run()
    d = app.experiment_details(eid)
    assert d["status"] == "COMPLETED" and d["counts"]["COMPLETED"] == 12
    assert "running sequentially instead" in (app.workspace / "logs" / "application.log").read_text()
    assert mp.active_children() == []
