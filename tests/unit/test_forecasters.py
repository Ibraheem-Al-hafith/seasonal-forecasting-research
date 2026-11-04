import numpy as np
import pytest

from forecasting_research.config.loader import load_config
from forecasting_research.domain.contracts import ForecastRequest
from forecasting_research.domain.exceptions import ConfigError, DataError
from forecasting_research.forecasting import classical_decomposition as cd
from forecasting_research.forecasting import registry
from forecasting_research.forecasting.sarima import candidate_grid, choose_differencing

CFG = load_config(None).forecasting
METHODS = ["classical_decomposition", "holt_winters", "sarima"]


def series(m=4, n=80, form="additive", trend=0.3, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    s = np.sin(2 * np.pi * t / m)
    tr = 100 + trend * t
    y = tr + 10 * s + rng.normal(0, noise, n) if form == "additive" else tr * (1 + 0.1 * s) * (1 + rng.normal(0, 0.005, n))
    return y


def test_registry():
    assert set(METHODS) <= set(registry.available_methods())  # tests may register extra methods
    with pytest.raises(ConfigError):
        registry.create_method("nope", CFG)
    with pytest.raises(ConfigError):
        registry.validate_method_names(["sarima", "nope"])


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("form", ["additive", "multiplicative"])
def test_shape_finite_and_params_recorded(name, form):
    y = series(form=form)
    res = registry.create_method(name, CFG).forecast(ForecastRequest(y, 4, 4, form))
    assert res.forecast.shape == (4,) and np.isfinite(res.forecast).all()
    assert res.parameters and res.runtime_seconds >= 0


@pytest.mark.parametrize("name", METHODS)
def test_input_validation(name):
    m = registry.create_method(name, CFG)
    for bad in [ForecastRequest(np.array([1.0, np.nan] * 8), 4, 4, "additive"),
                ForecastRequest(np.arange(5.0), 4, 4, "additive"),          # < 2 cycles
                ForecastRequest(series(), 4, 0, "additive"),
                ForecastRequest(series(), 4, 4, "bogus"),
                ForecastRequest(-series(), 4, 4, "multiplicative")]:
        with pytest.raises(DataError):
            m.forecast(bad)


def test_does_not_mutate_input():
    y = series()
    before = y.copy()
    for name in METHODS:
        registry.create_method(name, CFG).forecast(ForecastRequest(y, 4, 4, "additive"))
    assert np.array_equal(y, before)


def test_classical_exact_on_noiseless_additive():
    m, n, h = 4, 40, 8
    t = np.arange(n + h)
    full = 50 + 0.5 * t + np.array([3, -1, -4, 2.0])[t % m]
    res = registry.create_method("classical_decomposition", CFG).forecast(ForecastRequest(full[:n], m, h, "additive"))
    assert np.allclose(res.forecast, full[n:], atol=1e-6)
    assert sum(res.parameters["seasonal_indices"]) == pytest.approx(0, abs=1e-9)


def test_classical_exact_on_noiseless_multiplicative():
    m, n, h = 4, 40, 8
    t = np.arange(n + h)
    idx = np.array([1.1, 0.9, 0.8, 1.2])
    full = (50 + 0.5 * t) * idx[t % m]
    res = registry.create_method("classical_decomposition", CFG).forecast(ForecastRequest(full[:n], m, h, "multiplicative"))
    # moving-average trend of a trending multiplicative series is approximate: tight but not exact
    assert np.allclose(res.forecast, full[n:], rtol=0.01)
    assert np.mean(res.parameters["seasonal_indices"]) == pytest.approx(1.0)


def test_centered_ma_even_odd():
    y = np.arange(20.0)
    for m in (4, 5):
        tr = cd.centered_moving_average(y, m)
        h = m // 2
        assert np.isnan(tr[:h]).all() and np.isnan(tr[-h:]).all()
        assert np.allclose(tr[h:-h], y[h:-h])  # linear data is reproduced by a centred MA


def test_sarima_grid_bounded():
    g = candidate_grid(CFG.sarima)
    assert len(g) == 18 == CFG.sarima.candidate_count()
    assert all(sum(a) <= 2 and sum(b) <= 1 for a, b in g)
    assert sum(g[0][0]) + sum(g[0][1]) == 0  # simplest first


def test_sarima_differencing_detects_trend_and_season():
    d, D, diag = choose_differencing(series(m=12, n=120, trend=0.5), 12, CFG.sarima)
    assert D == 1 and diag["seasonal_strength"] > 0.64


def test_sarima_search_records_diagnostics():
    p = registry.create_method("sarima", CFG).forecast(ForecastRequest(series(), 4, 4, "additive")).parameters
    assert p["n_candidates_considered"] <= p["n_candidates_planned"] == 18
    assert p["search_limits"]["max_pq_sum"] == 2 and p["selection_criterion"] == "aic"
    assert len(p["order"]) == 3 and len(p["seasonal_order"]) == 4


def test_holt_winters_trend_selection_recorded():
    p = registry.create_method("holt_winters", CFG).forecast(ForecastRequest(series(trend=1.0), 4, 4, "additive")).parameters
    assert p["trend"] in ("none", "add") and p["selection_criterion"] == "aicc"
