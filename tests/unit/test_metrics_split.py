import numpy as np
import pytest

from forecasting_research.domain.enums import MetricStatus
from forecasting_research.domain.exceptions import DataError
from forecasting_research.evaluation import metrics as M
from forecasting_research.evaluation.splitter import resolve_horizon, split_series


def test_known_values():
    a, f = np.array([10.0, 20, 30, 40]), np.array([12.0, 18, 33, 40])
    assert M.mae(a, f) == pytest.approx(1.75)
    assert M.mse(a, f) == pytest.approx(4.25)
    assert M.rmse(a, f) == pytest.approx(np.sqrt(4.25))
    mp = M.mape(a, f)
    assert mp.status == MetricStatus.OK and mp.value == pytest.approx((0.2 + 0.1 + 0.1 + 0) / 4 * 100)


def test_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a = rng.uniform(50, 150, 30)
    f = a + rng.normal(0, 5, 30)
    assert M.mae(a, f) == pytest.approx(sk.mean_absolute_error(a, f))
    assert M.mse(a, f) == pytest.approx(sk.mean_squared_error(a, f))
    assert M.mape(a, f).value == pytest.approx(sk.mean_absolute_percentage_error(a, f) * 100)


@pytest.mark.parametrize("actual", [[0.0, 1, 2, 3], [100, 100, 1e-6, 100], [0.0, 0, 0, 0]])
def test_mape_undefined_never_valid_looking(actual):
    r = M.mape(np.array(actual), np.array(actual) + 1)
    assert r.value is None and r.status == MetricStatus.UNDEFINED_NEAR_ZERO


def test_negative_actuals_defined():
    assert M.mape(np.array([-10.0, -20]), np.array([-11.0, -18])).status == MetricStatus.OK


def test_metric_input_validation():
    with pytest.raises(DataError):
        M.mae(np.array([1.0, 2]), np.array([1.0]))
    with pytest.raises(DataError):
        M.mae(np.array([1.0, np.nan]), np.array([1.0, 2]))


def test_split_80_20_example_from_blueprint():
    s = split_series(np.arange(100.0), 0.8, 12)
    assert (s.train_size, s.test_size, s.horizon) == (80, 20, 12)
    assert len(s.test_eval) == 12 and np.array_equal(s.test_eval, np.arange(80.0, 92))
    assert np.array_equal(s.train, np.arange(80.0))


def test_horizon_override_and_errors():
    assert resolve_horizon(None, 7) == 7 and resolve_horizon(5, 7) == 5
    with pytest.raises(DataError):
        split_series(np.arange(100.0), 0.8, 4, horizon_override=25)
    with pytest.raises(DataError):
        split_series(np.arange(20.0), 0.8, 12)
