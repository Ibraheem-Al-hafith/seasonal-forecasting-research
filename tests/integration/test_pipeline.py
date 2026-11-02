import numpy as np

from forecasting_research.application.services import ResearchApp
from forecasting_research.evaluation.evaluator import Evaluator
from forecasting_research.forecasting.registry import create_method


def test_generation_persists_datasets_and_separate_truth(app, cfg):
    batch = app.generate()
    recs = app.repo.datasets.list_datasets(batch_id=batch)
    assert len(recs) == cfg.generation.dataset_count() == 4
    assert app.repo.datasets.get_batch(batch)["status"] == "COMPLETED"
    r = recs[0]
    assert r.observed_path != r.truth_path and ".observed." in r.observed_path and ".truth." in r.truth_path
    y = app.store.load_observed(r.observed_path, r.observed_sha256)
    truth = app.store.load_truth(r.truth_path, r.truth_sha256)
    assert len(y) == r.length == len(truth.trend)
    assert r.metadata["generator_params"]["base_level"] == 100.0 and r.metadata["seed"] == r.seed


def test_generation_is_idempotent_and_reuses_datasets(app):
    b1, b2 = app.generate(), app.generate()
    assert b1 != b2
    assert app.repo.datasets.count_datasets() == 4  # same ids: verified & linked, not duplicated
    assert {d.dataset_id for d in app.repo.datasets.list_datasets(batch_id=b1)} == \
           {d.dataset_id for d in app.repo.datasets.list_datasets(batch_id=b2)}


def test_two_workspaces_reproduce_identical_data(tmp_path, cfg):
    a, b = ResearchApp(tmp_path / "a", cfg), ResearchApp(tmp_path / "b", cfg)
    a.generate(), b.generate()
    for da, db_ in zip(a.repo.datasets.list_datasets(), b.repo.datasets.list_datasets(), strict=True):
        assert da.dataset_id == db_.dataset_id
        assert np.array_equal(a.store.load_observed(da.observed_path), b.store.load_observed(db_.observed_path))


def test_dataset_split_forecast_evaluation(app, cfg):
    app.generate()
    rec = app.repo.datasets.list_datasets()[0]
    y = app.store.load_observed(rec.observed_path)
    for name in cfg.forecasting.methods:
        out = Evaluator(cfg.evaluation).evaluate(y, rec.seasonality_type, rec.seasonal_period, create_method(name, cfg.forecasting))
        assert out.train_size == 80 and out.test_size == 20 and out.horizon == rec.seasonal_period
        assert np.array_equal(out.actual, y[80:80 + out.horizon])
        assert set(out.metrics) == {"MAE", "MSE", "RMSE", "MAPE"}


def test_results_survive_restart_and_analysis_reads_storage(tmp_path, cfg):
    a = ResearchApp(tmp_path / "ws", cfg)
    eid = a.run()
    del a
    b = ResearchApp(tmp_path / "ws", cfg)  # fresh process state
    det = b.experiment_details(eid)
    assert det["status"] == "COMPLETED" and det["outcome"] == "SUCCESS" and det["counts"]["COMPLETED"] == 12
    ev = b.repo.evaluations.list_for_experiment(eid)[0]
    actual, fc = b.store.load_evaluation(ev["artifact_path"], ev["artifact_sha256"])
    assert len(actual) == len(fc) == ev["horizon"]
    m = b.repo.evaluations.metrics(ev["evaluation_id"])
    assert np.isclose(m["MAE"].value, np.mean(np.abs(actual - fc)))
    assert len(b.aggregate([eid], ["seasonality"])) == 6  # 2 seasonalities x 3 models
    assert b.check_workspace().ok
