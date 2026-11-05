"""SARIMA component tests: candidates, criterion mapping, convergence, search stages, differencing, diagnostics."""

import itertools
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from forecasting_research.config.loader import config_from_dict
from forecasting_research.config.schemas import SarimaCfg
from forecasting_research.domain.contracts import ForecastRequest
from forecasting_research.domain.exceptions import (
    ConfigError,
    DataError,
    EvaluationTimeout,
    InvalidForecastError,
    ModelFitError,
)
from forecasting_research.forecasting import classical_decomposition as cd
from forecasting_research.forecasting.registry import create_method
from forecasting_research.forecasting.sarima import SarimaForecaster
from forecasting_research.forecasting.sarima_components import fitting
from forecasting_research.forecasting.sarima_components.candidates import (
    Candidate,
    full_grid,
    stage1_grid,
    stage2_grid,
)
from forecasting_research.forecasting.sarima_components.differencing import (
    choose_regular_differencing,
    choose_seasonal_differencing,
    describe_trend,
    seasonal_strength,
)
from forecasting_research.forecasting.sarima_components.fitting import (
    CandidateRecord,
    criterion_value,
    fit_candidate,
)
from forecasting_research.forecasting.sarima_components.search import run_search

FAST = SarimaCfg(max_pq_sum=1, max_seasonal_pq_sum=1, stage1_max_pq_sum=1, stage1_max_seasonal_pq_sum=1)


def wave(n=120, m=12, form="additive", growth=0.0, noise=0.3, seed=0, amp=10.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    s = np.sin(2 * np.pi * t / m)
    if form == "additive":
        return 100 + 0.2 * t + amp * s + rng.normal(0, noise, n)
    return 100 * np.exp(growth * t) * (1 + 0.2 * s) * (1 + rng.normal(0, 0.01, n))


# ================================================================== moving average / indices
@pytest.mark.parametrize("m", [3, 7, 24])
def test_moving_average_odd_even_exact_on_linear_data(m):
    y = np.arange(60.0)
    tr = cd.centered_moving_average(y, m)
    h = m // 2
    assert np.isnan(tr[:h]).all() and np.isnan(tr[len(y) - h:]).all()
    assert np.allclose(tr[h:len(y) - h], y[h:len(y) - h])


def test_even_period_uses_half_weight_end_points():
    y = np.zeros(20)
    y[8] = 1.0  # impulse: weights of the 2xm kernel are 1/m inside, 1/(2m) at both ends
    tr = cd.centered_moving_average(y, 4)
    assert tr[6] == pytest.approx(0.125) and tr[7] == pytest.approx(0.25) and tr[8] == pytest.approx(0.25)
    assert tr[9] == pytest.approx(0.25) and tr[10] == pytest.approx(0.125)


@pytest.mark.parametrize("bad_y,m", [
    (np.array([1.0, np.nan, 3, 4, 5, 6, 7, 8]), 3), (np.array([1.0, np.inf, 3, 4, 5, 6]), 3),
    (np.arange(10.0), 1), (np.arange(10.0), 2.5), (np.arange(10.0), True), (np.arange(4.0), 12), (np.ones((4, 4)), 2)])
def test_moving_average_validates_input(bad_y, m):
    with pytest.raises(DataError):
        cd.centered_moving_average(bad_y, m)


def test_seasonal_indices_normalisation_and_guards():
    y = wave(60, 4)
    tr = cd.centered_moving_average(y, 4)
    assert cd.seasonal_indices(y, tr, 4, "additive").sum() == pytest.approx(0, abs=1e-9)
    ym = wave(60, 4, "multiplicative")
    assert cd.seasonal_indices(ym, cd.centered_moving_average(ym, 4), 4, "multiplicative").mean() == pytest.approx(1.0)
    with pytest.raises(ModelFitError):  # zero trend -> multiplicative detrending undefined
        cd.seasonal_indices(np.ones(20), np.zeros(20), 4, "multiplicative")
    with pytest.raises(DataError):
        cd.seasonal_indices(np.ones(20), np.ones(20), 4, "bogus")
    for n, m in ((120, 30), (120, 24)):  # the report's cases: 3 and 4 observations per phase
        yy = wave(n, m)
        with pytest.warns(UserWarning, match="per seasonal phase"):  # unstable indices are reported, not hidden
            cd.seasonal_indices(yy, cd.centered_moving_average(yy, m), m, "additive")
    with pytest.raises(ModelFitError, match="No detrended observations"):
        yy = wave(40, 30)
        cd.seasonal_indices(yy, cd.centered_moving_average(yy, 30), 30, "additive")


# ================================================================== seasonal strength (diagnostic)
def test_multiplicative_strength_uses_log_domain_and_beats_additive_on_growing_series():
    y = wave(120, 12, "multiplicative", growth=0.03)
    log_s, add_s = seasonal_strength(y, 12, "multiplicative"), seasonal_strength(y, 12, "additive")
    assert log_s.domain == "log" and add_s.domain == "additive"
    assert log_s.value > 0.9 and log_s.value > add_s.value  # the old additive-always code understated it


def test_strength_constant_series_and_invalid_input():
    s = seasonal_strength(np.full(60, 5.0), 4, "additive")
    assert s.value == 0.0
    for bad in (np.r_[np.ones(30), 0.0, np.ones(29)], np.r_[np.ones(30), -1.0, np.ones(29)]):
        with pytest.raises(DataError):
            seasonal_strength(bad, 4, "multiplicative")  # log needs strictly positive data
    with pytest.raises(DataError):
        seasonal_strength(np.r_[np.ones(30), np.nan, np.ones(29)], 4, "additive")
    with pytest.raises(DataError):
        seasonal_strength(np.ones(60), 4, "weird")


def test_strength_reliability_flag_reflects_observations_per_phase():
    assert seasonal_strength(wave(120, 4), 4).reliable
    s30 = seasonal_strength(wave(120, 30), 30)
    assert not s30.reliable and s30.obs_per_phase == pytest.approx(3.0)  # the report's m=30 / n=120 case


# ================================================================== seasonal differencing strategies
@pytest.mark.parametrize("strategy,expected", [("fixed_zero", 0), ("fixed_one", 1)])
def test_fixed_seasonal_differencing(strategy, expected):
    D, info = choose_seasonal_differencing(wave(), 12, "additive", SarimaCfg(seasonal_differencing_strategy=strategy))
    assert D == expected and info["seasonal_strength"]["value"] > 0.9  # strength is stored as a diagnostic either way


def test_default_strategy_is_fixed_and_independent_of_seasonal_strength():
    assert SarimaCfg().seasonal_differencing_strategy == "fixed_one"
    for y in (wave(), np.random.default_rng(1).normal(100, 1, 120)):  # strong and absent seasonality: same D
        D, info = choose_seasonal_differencing(y, 12, "additive", SarimaCfg())
        assert D == 1 and info["strategy"] == "fixed_one"


def test_legacy_strength_rule_still_available_for_comparison():
    cfg = SarimaCfg(seasonal_differencing_strategy="seasonal_strength")
    assert choose_seasonal_differencing(wave(), 12, "additive", cfg)[0] == 1
    assert choose_seasonal_differencing(np.random.default_rng(1).normal(100, 1, 120), 12, "additive", cfg)[0] == 0


def test_unit_root_test_strategies_need_pmdarima(monkeypatch):
    monkeypatch.setitem(sys.modules, "pmdarima", None)
    monkeypatch.setitem(sys.modules, "pmdarima.arima", None)
    with pytest.raises(ConfigError, match="pmdarima"):
        SarimaForecaster(SarimaCfg(seasonal_differencing_strategy="ocsb"))  # fails at construction, not mid-run


def test_ocsb_strategy_with_pmdarima():
    pytest.importorskip("pmdarima")
    D, info = choose_seasonal_differencing(wave(), 12, "additive", SarimaCfg(seasonal_differencing_strategy="ocsb"))
    assert D in (0, 1) and info.get("test") == "ocsb"


# ================================================================== KPSS / deterministic trend
def test_kpss_decisions_on_known_series():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 160)
    walk = np.cumsum(rng.normal(0, 1, 160))
    trended = 0.5 * np.arange(160) + noise
    auto, level, ct = SarimaCfg(kpss_regression="auto"), SarimaCfg(kpss_regression="c"), SarimaCfg(kpss_regression="ct")
    assert choose_regular_differencing(noise, 4, 0, auto)[:2] == (0, "c")
    assert choose_regular_differencing(walk, 4, 0, auto)[:2] == (1, "c")           # stochastic trend -> differencing
    d, trend, info = choose_regular_differencing(trended, 4, 0, auto)
    assert (d, trend) == (0, "ct") and info["deterministic_trend"] == "linear_time_trend"  # deterministic trend: no differencing
    assert choose_regular_differencing(trended, 4, 0, level)[0] == 1               # legacy level-only test differences it
    assert choose_regular_differencing(trended, 4, 0, ct)[:2] == (0, "ct")
    assert choose_regular_differencing(walk, 4, 0, ct)[:2] == (1, "c")
    assert all("pvalue" in t for t in info["kpss_tests"]) and info["kpss_tests"][0]["regression"] == "c"


def test_after_seasonal_differencing_a_linear_trend_is_just_a_constant():
    y = 0.5 * np.arange(160) + np.random.default_rng(2).normal(0, 1, 160)
    d, trend, _ = choose_regular_differencing(y, 4, 1, SarimaCfg(kpss_regression="auto"))
    assert (d, trend) == (0, "c")  # never 'ct' when D = 1 (it would imply a quadratic trend)


def test_trend_description_matches_differencing():
    assert describe_trend(0, 0, "c") == "intercept" and describe_trend(1, 0, "c") == "drift"
    assert describe_trend(0, 1, "c") == "seasonal_drift" and describe_trend(1, 1, "n") == "none"
    assert describe_trend(0, 0, "ct") == "linear_time_trend"


# ================================================================== candidate generation (independent of fitting)
def test_candidate_counts_default_and_report_configuration():
    d = SarimaCfg()
    assert (len(full_grid(d)), len(stage1_grid(d)), len(stage2_grid(d))) == (18, 9, 9)
    r = SarimaCfg(max_seasonal_pq_sum=2)  # the report's search space
    assert (len(full_grid(r)), len(stage1_grid(r)), len(stage2_grid(r))) == (36, 9, 27)
    for cfg in (d, r):
        f, s1, s2 = full_grid(cfg), stage1_grid(cfg), stage2_grid(cfg)
        assert len(set(f)) == len(f) and not set(s1) & set(s2) and set(s1) | set(s2) == set(f)
        assert all(c.p + c.q <= cfg.max_pq_sum and c.P + c.Q <= cfg.max_seasonal_pq_sum for c in f)
        assert all(c.p + c.q <= 1 and c.P + c.Q <= 1 for c in s1)  # first-order screening only
    assert f[0] == Candidate(0, 0, 0, 0)  # simplest first


def test_full_grid_strategy_has_no_stage2():
    cfg = SarimaCfg(search_strategy="full_grid")
    assert stage1_grid(cfg) == full_grid(cfg) and stage2_grid(cfg) == []


def test_stage1_limits_default_to_auto_and_follow_lower_full_limits():
    c = config_from_dict({"forecasting": {"sarima": {"max_pq_sum": 1, "max_seasonal_pq_sum": 0}}}).forecasting.sarima
    assert (c.stage1_pq(), c.stage1_spq()) == (1, 0)  # no error: auto-clamped
    assert (len(full_grid(c)), len(stage1_grid(c)), len(stage2_grid(c))) == (3, 3, 0)
    assert (SarimaCfg().stage1_pq(), SarimaCfg().stage1_spq()) == (1, 1)
    assert SarimaCfg(max_pq_sum=0, max_seasonal_pq_sum=0).stage1_pq() == 0


def test_stage_limits_must_not_exceed_full_limits():
    with pytest.raises(ConfigError):
        config_from_dict({"forecasting": {"sarima": {"max_pq_sum": 1, "stage1_max_pq_sum": 2}}})
    with pytest.raises(ConfigError):
        config_from_dict({"forecasting": {"sarima": {"selection_criterion": "hqic"}}})
    with pytest.raises(ConfigError):
        config_from_dict({"forecasting": {"sarima": {"method_version": 1}}})


# ================================================================== information criterion mapping
def test_criterion_mapping_reads_the_requested_attribute():
    res = SimpleNamespace(aic=1.0, aicc=2.0, bic=3.0)
    assert (criterion_value(res, "aic"), criterion_value(res, "aicc"), criterion_value(res, "bic")) == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="Unsupported"):
        criterion_value(res, "hqic")


@pytest.mark.parametrize("crit", ["aic", "aicc", "bic"])
def test_forecaster_selects_by_the_configured_criterion(crit):
    cfg = SarimaCfg(max_pq_sum=1, max_seasonal_pq_sum=0, stage1_max_pq_sum=1, stage1_max_seasonal_pq_sum=0, selection_criterion=crit)
    p = SarimaForecaster(cfg).forecast(ForecastRequest(wave(80, 4), 4, 4, "additive")).parameters
    assert p["criterion_value"] == p[crit]
    others = {p[c] for c in ("aic", "aicc", "bic") if c != crit}
    assert p[crit] not in others  # the three criteria really differ: nothing is silently mapped to another


# ================================================================== convergence handling
class _FakeModel:
    def __init__(self, retvals):
        self.retvals = retvals

    def __call__(self, *a, **k):
        return self

    def fit(self, **k):
        return SimpleNamespace(aic=10.0, aicc=11.0, bic=12.0, mle_retvals=self.retvals)


@pytest.mark.parametrize("retvals,status", [(None, "no_convergence_info"), ({}, "no_convergence_info"),
                                            ({"converged": False}, "non_converged"), ({"converged": True}, "ok")])
def test_missing_convergence_metadata_is_never_treated_as_converged(monkeypatch, retvals, status):
    monkeypatch.setattr(fitting, "SARIMAX", _FakeModel(retvals))
    rec, res = fit_candidate(np.arange(50.0), Candidate(0, 0, 0, 0), 0, 0, 4, "c", FAST, 1)
    assert rec.status == status and rec.selectable == (status == "ok") and (res is not None) == (status == "ok")


def test_non_finite_criterion_and_fit_errors_are_recorded_not_selected(monkeypatch):
    class Nan(_FakeModel):
        def fit(self, **k):
            return SimpleNamespace(aic=np.nan, aicc=np.nan, bic=np.nan, mle_retvals={"converged": True})

    monkeypatch.setattr(fitting, "SARIMAX", Nan(None))
    assert fit_candidate(np.arange(50.0), Candidate(0, 0, 0, 0), 0, 0, 4, "c", FAST, 1)[0].status == "non_finite_criterion"

    def boom(*a, **k):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(fitting, "SARIMAX", boom)
    rec, res = fit_candidate(np.arange(50.0), Candidate(0, 0, 0, 0), 0, 0, 4, "c", FAST, 1)
    assert rec.status == "error" and "LinAlgError" in rec.error and res is None


# ================================================================== search engine (stubbed fits)
def stub_fit(table=None, base=100.0):
    table = table or {}

    def fit(y, cand, d, D, m, trend, cfg, stage):
        status, score = table.get(cand, ("ok", base + cand.complexity))
        rec = CandidateRecord(cand, (cand.p, d, cand.q), (cand.P, D, cand.Q, m), stage, status=status,
                              converged=(status == "ok"), score=score if status in ("ok", "non_converged") else None)
        return rec, (object() if rec.selectable else None)

    return fit


def diag(significant=False):
    return lambda res, cand, d, D, m, alpha: {"significant": significant}


Y = np.arange(60.0)
SIMPLE_BEST = {Candidate(0, 0, 0, 0): ("ok", 90.0)}


def test_stage2_not_run_when_simple_model_is_adequate():
    out = run_search(Y, 0, 0, 4, "c", SarimaCfg(), fit_fn=stub_fit(SIMPLE_BEST), diag_fn=diag(False))
    assert len(out.records) == 9 and [s["stage"] for s in out.stages] == [1]
    assert out.expansion["triggered"] is False and out.best.candidate == Candidate(0, 0, 0, 0) and out.stopped_reason == "completed"


def test_stage2_runs_when_best_model_is_at_the_stage1_boundary_and_best_overall_is_selected():
    table = {Candidate(1, 0, 0, 0): ("ok", 80.0), Candidate(2, 0, 0, 0): ("ok", 70.0)}  # better model only exists in stage 2
    out = run_search(Y, 0, 0, 4, "c", SarimaCfg(), fit_fn=stub_fit(table), diag_fn=diag(False))
    assert out.expansion["triggered"] and "best_stage1_model_at_boundary" in out.expansion["reasons"]
    assert len(out.records) == 18 and out.best.candidate == Candidate(2, 0, 0, 0) and out.best.stage == 2
    assert len({r.candidate for r in out.records}) == 18  # no candidate fitted twice


def test_stage2_runs_on_residual_autocorrelation():
    out = run_search(Y, 0, 0, 4, "c", SarimaCfg(), fit_fn=stub_fit(SIMPLE_BEST), diag_fn=diag(True))
    assert out.expansion["reasons"] == ["residual_autocorrelation"] and len(out.records) == 18


@pytest.mark.parametrize("trigger,lb,boundary_best,expected", [
    ("boundary", True, False, False), ("boundary", False, True, True), ("ljung_box", False, True, False),
    ("ljung_box", True, False, True), ("always", False, False, True), ("boundary_or_ljung_box", False, False, False)])
def test_trigger_policies(trigger, lb, boundary_best, expected):
    table = {Candidate(1, 0, 0, 0): ("ok", 80.0)} if boundary_best else SIMPLE_BEST
    out = run_search(Y, 0, 0, 4, "c", SarimaCfg(stage2_trigger=trigger), fit_fn=stub_fit(table), diag_fn=diag(lb))
    assert out.expansion["triggered"] is expected


def test_full_grid_strategy_fits_everything_in_one_stage():
    out = run_search(Y, 0, 0, 4, "c", SarimaCfg(search_strategy="full_grid"), fit_fn=stub_fit(), diag_fn=diag())
    assert len(out.records) == 18 and len(out.stages) == 1 and out.expansion["triggered"] is False


def test_report_search_space_fits_9_then_27():
    cfg = SarimaCfg(max_seasonal_pq_sum=2)
    out = run_search(Y, 0, 0, 4, "c", cfg, fit_fn=stub_fit({Candidate(1, 0, 0, 0): ("ok", 50.0)}), diag_fn=diag())
    assert [s["n_fitted"] for s in out.stages] == [9, 27] and len(out.records) == 36


def test_non_converged_candidates_are_never_selected_even_with_the_best_score():
    table = {Candidate(0, 0, 0, 0): ("non_converged", 1.0), Candidate(1, 0, 0, 0): ("no_convergence_info", 2.0),
             Candidate(0, 1, 0, 0): ("error", None), Candidate(0, 0, 1, 0): ("ok", 95.0)}
    out = run_search(Y, 0, 0, 4, "c", SarimaCfg(stage2_trigger="boundary"), fit_fn=stub_fit(table), diag_fn=diag())
    assert out.best.candidate == Candidate(0, 0, 1, 0)
    c = out.counts()
    assert c["n_non_converged"] == 1 and c["n_no_convergence_info"] == 1 and c["n_errors"] == 1


def test_search_fails_loudly_when_nothing_is_selectable():
    table = {c: ("non_converged", 1.0) for c in full_grid(SarimaCfg())}
    with pytest.raises(ModelFitError, match="none of the"):
        run_search(Y, 0, 0, 4, "c", SarimaCfg(), fit_fn=stub_fit(table), diag_fn=diag())


def test_time_budget_truncation_is_recorded_and_skips_stage2():
    ticks = itertools.count(0, 100)
    cfg = SarimaCfg(time_budget_seconds=450)
    out = run_search(Y, 0, 0, 4, "c", cfg, fit_fn=stub_fit({Candidate(1, 0, 0, 0): ("ok", 80.0)}), diag_fn=diag(True), clock=lambda: next(ticks))
    assert out.truncated and out.stopped_reason == "budget_exhausted"
    assert len(out.records) < 9 and len(out.stages) == 1 and "skipped" in out.expansion["note"]


def test_time_budget_without_any_usable_candidate_is_a_timeout():
    ticks = itertools.count(0, 1000)
    with pytest.raises(EvaluationTimeout):
        run_search(Y, 0, 0, 4, "c", SarimaCfg(time_budget_seconds=500), fit_fn=stub_fit(), diag_fn=diag(), clock=lambda: next(ticks))


# ================================================================== forecast generation + log transform validation
@pytest.mark.parametrize("form", ["additive", "multiplicative"])
@pytest.mark.parametrize("h", [1, 4, 9])
def test_exactly_h_finite_forecasts_and_stored_contract(form, h):
    y = wave(80, 4, form, growth=0.0)
    p_res = SarimaForecaster(FAST).forecast(ForecastRequest(y, 4, h, form))
    assert p_res.forecast.shape == (h,) and np.isfinite(p_res.forecast).all()
    p = p_res.parameters
    for key in ("method_version", "order", "seasonal_order", "trend", "deterministic_trend", "criterion_value", "differencing",
                "stages", "expansion", "search_truncated", "stopped_reason", "n_selectable", "n_non_converged", "candidates"):
        assert key in p, key
    expected_domain = "log" if form == "multiplicative" else "additive"
    assert p["differencing"]["seasonal"]["seasonal_strength"]["domain"] == expected_domain
    assert p["differencing"]["D"] == 1 and p["seasonal_order"][1] == 1  # default strategy fixed_one


def test_log_transform_validation_both_directions():
    with pytest.raises(DataError):
        SarimaForecaster._transform(np.array([1.0, 0.0, 2.0]), True)
    with pytest.raises(InvalidForecastError):
        SarimaForecaster._inverse(np.array([1e6, 1e6]), True, np.array([100.0, 110.0]))  # exp overflow
    with pytest.raises(InvalidForecastError):
        SarimaForecaster._inverse(np.array([np.nan, 1.0]), True, np.array([100.0, 110.0]))
    assert np.allclose(SarimaForecaster._inverse(np.log(np.array([100.0, 120.0])), True, np.array([100.0, 110.0])), [100.0, 120.0])
    assert SarimaForecaster._transform(np.array([1.0, 2.0]), False).tolist() == [1.0, 2.0]


def test_log_transform_option_end_to_end():
    cfg = SarimaCfg(max_pq_sum=1, max_seasonal_pq_sum=0, stage1_max_pq_sum=1, stage1_max_seasonal_pq_sum=0, log_transform_multiplicative=True)
    r = create_method("sarima", config_from_dict({"forecasting": {"sarima": cfg.model_dump()}}).forecasting).forecast(
        ForecastRequest(wave(80, 4, "multiplicative"), 4, 4, "multiplicative"))
    assert r.parameters["log_transform"] and (r.forecast > 0).all()


def test_test_data_cannot_influence_stage2_decision_or_selection():
    """The expansion decision uses training residuals only: changing the future leaves every diagnostic identical."""
    from forecasting_research.evaluation.evaluator import Evaluator
    y = wave(100, 4)
    y2 = y.copy()
    y2[80:] += 500.0
    ev = Evaluator(config_from_dict({}).evaluation)
    m = SarimaForecaster(FAST)
    a = ev.evaluate(y, "additive", 4, m)
    b = ev.evaluate(y2, "additive", 4, m)
    for k in ("order", "seasonal_order", "expansion", "stages", "criterion_value", "differencing"):
        a.parameters[k] = {**a.parameters[k]} if isinstance(a.parameters[k], dict) else a.parameters[k]
    strip = lambda p: {k: v for k, v in p.items() if k not in ("search_seconds", "candidates")}  # noqa: E731
    assert strip(a.parameters).keys() == strip(b.parameters).keys()
    for k in ("order", "seasonal_order", "expansion", "criterion_value", "differencing", "n_candidates_considered"):
        assert a.parameters[k] == b.parameters[k], k
