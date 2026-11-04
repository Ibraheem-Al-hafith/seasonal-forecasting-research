import os

import pytest

from forecasting_research.config.loader import config_from_dict
from forecasting_research.domain.contracts import ProgressEvent
from forecasting_research.domain.exceptions import ConfigError
from forecasting_research.execution.factory import create_strategy, import_worker_modules
from forecasting_research.execution.parallel import ParallelStrategy, limit_blas_threads, resolve_workers
from forecasting_research.execution.sequential import SequentialStrategy
from forecasting_research.monitoring import events as E
from forecasting_research.monitoring.progress import ProgressTracker


def test_default_is_sequential_and_parallel_config_validates():
    assert config_from_dict({}).execution.strategy == "sequential"
    c = config_from_dict({"execution": {"strategy": "parallel", "workers": 4}})
    s = create_strategy(c.execution)
    assert isinstance(s, ParallelStrategy) and s.workers == 4 and s.start_method == "spawn"
    assert isinstance(create_strategy(config_from_dict({}).execution), SequentialStrategy)


@pytest.mark.parametrize("bad", [{"workers": 0}, {"workers": -2}, {"workers": "many"}, {"strategy": "gpu"},
                                 {"start_method": "magic"}, {"max_tasks_per_worker": 0}, {"unknown": 1}])
def test_invalid_execution_config_rejected(bad):
    with pytest.raises(ConfigError):
        config_from_dict({"execution": bad})


def test_resolve_workers(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert resolve_workers("auto") == 3 and resolve_workers(2) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert resolve_workers("auto") == 1
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert resolve_workers("auto") == 8  # capped: each worker holds ~150-250 MB


def test_blas_threads_limited_then_restored_without_overriding_user(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")  # explicit user choice must win
    with limit_blas_threads(True):
        assert os.environ["OMP_NUM_THREADS"] == "1" and os.environ["MKL_NUM_THREADS"] == "4"
    assert "OMP_NUM_THREADS" not in os.environ and os.environ["MKL_NUM_THREADS"] == "4"
    with limit_blas_threads(False):
        assert "OMP_NUM_THREADS" not in os.environ


def test_single_worker_parallel_describes_as_sequential():
    assert ParallelStrategy(1).describe() == "sequential"
    assert "3 workers" in ParallelStrategy(3).describe()


def test_bad_worker_module_is_config_error():
    with pytest.raises(ConfigError):
        import_worker_modules(["definitely.not.a.module"])


def test_eta_uses_throughput_not_per_task_duration(monkeypatch):
    """2 workers, each evaluation takes 10 s: 100 evaluations need ~500 s, not 1,000 s."""
    clock = [0.0]
    monkeypatch.setattr("forecasting_research.monitoring.progress.time.monotonic", lambda: clock[0])
    t = ProgressTracker()
    ev = lambda typ, eid=None, **kw: t.on_event(ProgressEvent(typ, experiment_id="x", evaluation_id=eid, **kw))  # noqa: E731
    ev(E.EXPERIMENT_STARTED)
    ev(E.PLAN_BUILT, payload={"total": 100})
    n = 0
    for _ in range(5):  # five rounds of two concurrent evaluations of 10 s each
        ids = [f"e{n}", f"e{n + 1}"]
        for i in ids:
            ev(E.EVALUATION_STARTED, i)
        clock[0] += 10
        for i in ids:
            ev(E.EVALUATION_COMPLETED, i)
        n += 2
    snap = t.snapshot()
    naive = 10 * (snap["pending"] + snap["running"])  # old formula: mean duration x remaining
    assert snap["eta_seconds"] is not None and snap["eta_seconds"] < 0.7 * naive
    assert snap["eta_seconds"] > 0.4 * naive  # and not absurdly optimistic (true: ~half of naive)
