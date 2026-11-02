"""Research-validity tests: leakage, fair comparison, parameter selection, metrics, reproducibility."""

import dataclasses
import hashlib

import numpy as np
import pytest

from forecasting_research.application.services import ResearchApp
from forecasting_research.domain.contracts import ForecastRequest
from forecasting_research.evaluation.evaluator import Evaluator
from forecasting_research.evaluation.splitter import split_series
from forecasting_research.forecasting.registry import create_method
from forecasting_research.persistence.artifacts import ArtifactStore
from tests.helpers import Naive, Registered

from ..conftest import make_cfg

BUILTIN = ["classical_decomposition", "holt_winters", "sarima"]


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_forecast_request_contract_has_no_room_for_leaks():
    assert {f.name for f in dataclasses.fields(ForecastRequest)} == {"train", "seasonal_period", "horizon", "seasonal_form"}


# ---- data leakage ---------------------------------------------------------------------------------
@pytest.mark.parametrize("method", BUILTIN)
def test_changing_future_observations_cannot_change_forecast_or_parameters(app, cfg, method):
    """Perturb every test observation: forecast AND selected parameters must be bit-identical."""
    app.generate()
    rec = app.repo.datasets.list_datasets()[0]
    y = app.store.load_observed(rec.observed_path)
    n_train = split_series(y, 0.8, rec.seasonal_period).train_size
    y2 = y.copy()
    y2[n_train:] = y2[n_train:] * 3 + 1000.0
    ev = Evaluator(cfg.evaluation)
    a = ev.evaluate(y, rec.seasonality_type, rec.seasonal_period, create_method(method, cfg.forecasting))
    b = ev.evaluate(y2, rec.seasonality_type, rec.seasonal_period, create_method(method, cfg.forecasting))
    assert np.array_equal(a.forecast, b.forecast)
    assert a.parameters == b.parameters  # incl. SARIMA order/AIC and Holt-Winters trend/AICc
    assert not np.array_equal(a.actual, b.actual) and a.metrics["RMSE"].value != b.metrics["RMSE"].value


def test_ground_truth_is_never_read_during_evaluation(app, monkeypatch):
    app.generate()
    reads = []
    orig = ArtifactStore.read_npz

    def spy(self, rel, expected_sha=None):
        reads.append(rel)
        return orig(self, rel, expected_sha)

    monkeypatch.setattr(ArtifactStore, "read_npz", spy)
    app.evaluate()
    assert reads and not any(".truth." in r for r in reads)
    assert all(".observed." in r or "/evaluations/" in r for r in reads)


def test_methods_only_receive_training_slice_and_readonly_data(app):
    app.generate()
    Naive.requests = []
    with Registered(spy_naive=Naive):
        app.evaluate(make_cfg(forecasting={"methods": ["spy_naive"]}))
    assert len(Naive.requests) == 4
    for req in Naive.requests:
        assert len(req.train) == 80 and req.horizon == req.seasonal_period
        assert not req.train.flags.writeable


# ---- fair comparison ------------------------------------------------------------------------------------
def test_identical_train_horizon_and_actuals_across_methods(app):
    app.generate()
    Naive.requests = []
    with Registered(spy_naive=Naive):
        cfg = make_cfg(forecasting={"methods": ["spy_naive", "holt_winters", "classical_decomposition"]})
        eid = app.evaluate(cfg)
    by_ds = {}
    for e in app.repo.evaluations.list_for_experiment(eid):
        actual, fc = app.store.load_evaluation(e["artifact_path"], e["artifact_sha256"])
        by_ds.setdefault(e["dataset_id"], []).append((e["train_size"], e["horizon"], _h(actual), len(fc)))
    for ds, rows in by_ds.items():
        assert len(set(rows)) == 1 and len(rows) == 3, f"methods saw different protocol on {ds}"
        rec = app.repo.datasets.get_dataset(ds)
        y = app.store.load_observed(rec.observed_path)
        assert rows[0][2] == _h(y[80:80 + rec.seasonal_period])  # actuals == the true next observations
    # the shared training slice handed to the (spied) method is exactly y[:80]
    recs = {d.dataset_id: app.store.load_observed(d.observed_path) for d in app.repo.datasets.list_datasets()}
    assert {_h(r.train) for r in Naive.requests} == {_h(y[:80]) for y in recs.values()}


def test_metrics_use_one_implementation_on_stored_values(app):
    eid = app.run()
    from forecasting_research.evaluation.metrics import compute_metrics
    for e in app.repo.evaluations.list_for_experiment(eid):
        actual, fc = app.store.load_evaluation(e["artifact_path"], e["artifact_sha256"])
        stored = app.repo.evaluations.metrics(e["evaluation_id"])
        for name, mv in compute_metrics(actual, fc).items():
            assert stored[name].value == pytest.approx(mv.value) and stored[name].status == mv.status


def test_horizon_override_applies_to_all_methods(tmp_path):
    cfg = make_cfg(evaluation={"forecast_horizon": 6})
    app = ResearchApp(tmp_path / "w", cfg)
    eid = app.run()
    assert {e["horizon"] for e in app.repo.evaluations.list_for_experiment(eid)} == {6}


def test_seasonal_mode_override_is_recorded_and_runs(tmp_path):
    cfg = make_cfg(evaluation={"seasonal_mode": "additive"}, forecasting={"methods": ["holt_winters"]})
    app = ResearchApp(tmp_path / "w", cfg)
    eid = app.run()
    forms = {app.repo.evaluations.get(e["evaluation_id"])["parameters"]["seasonal"] for e in app.repo.evaluations.list_for_experiment(eid)}
    assert forms == {"additive"}  # even on multiplicative data: deliberate mis-specification study


# ---- failures / reproducibility ---------------------------------------------------------------------------
def test_failed_evaluations_remain_distinguishable(app):
    from tests.helpers import AlwaysFails
    app.generate()
    with Registered(always_fails=AlwaysFails):
        eid = app.evaluate(make_cfg(forecasting={"methods": ["holt_winters", "always_fails"]}))
    res = app.results([eid])
    assert set(res.status) == {"COMPLETED", "FAILED"}
    failed = res[res.status == "FAILED"]
    assert failed["RMSE"].isna().all() and failed["failure_category"].notna().all()
    assert app.repo.evaluations.completed_artifact_refs() and len(app.repo.evaluations.completed_artifact_refs()) == 4


def test_same_seed_same_results_across_workspaces(tmp_path, cfg):
    """Identical data => identical forecasts for the deterministic methods."""
    out = []
    for name in ("a", "b"):
        a = ResearchApp(tmp_path / name, make_cfg(forecasting={"methods": ["classical_decomposition", "holt_winters"]}))
        eid = a.run()
        out.append(a.results([eid]).sort_values(["dataset_id", "model_name"])[["dataset_id", "model_name", "RMSE"]].reset_index(drop=True))
    assert out[0].equals(out[1])


def test_generated_ground_truth_matches_observed_after_roundtrip(app):
    app.generate()
    for r in app.repo.datasets.list_datasets():
        y = app.store.load_observed(r.observed_path, r.observed_sha256)
        t = app.store.load_truth(r.truth_path, r.truth_sha256)
        recon = t.trend + t.seasonal + t.noise if r.seasonality_type == "additive" else t.trend * t.seasonal * t.noise
        assert np.allclose(y, recon)
