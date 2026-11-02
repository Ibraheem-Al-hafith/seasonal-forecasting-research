import numpy as np
import pytest

from forecasting_research.config.loader import load_config
from forecasting_research.generation.generator import DatasetGenerator, make_dataset_id
from forecasting_research.generation.seeds import derive_seed


@pytest.fixture(scope="module")
def gen():
    return DatasetGenerator(load_config(None).generation)


def test_full_design_unique_ids_and_count(gen):
    specs = list(gen.iter_specs())
    assert len(specs) == 540
    assert len({make_dataset_id(s, gen.cfg) for s in specs}) == 540
    assert len({s.seed for s in specs}) == 540


def test_reproducible(gen):
    s = list(gen.iter_specs())[123]
    a, b = gen.generate(s), gen.generate(s)
    assert a.dataset_id == b.dataset_id
    for x, y in [(a.observed, b.observed), (a.truth.trend, b.truth.trend), (a.truth.seasonal, b.truth.seasonal), (a.truth.noise, b.truth.noise)]:
        assert np.array_equal(x, y)


def test_seed_depends_on_identity_not_position():
    assert derive_seed(42, "additive|12|linear|low|100", 0) == derive_seed(42, "additive|12|linear|low|100", 0)
    assert derive_seed(42, "additive|12|linear|low|100", 0) != derive_seed(42, "additive|12|linear|low|100", 1)
    assert derive_seed(42, "additive|12|linear|low|100", 0) != derive_seed(43, "additive|12|linear|low|100", 0)


def test_seeds_stable_when_design_lists_change():
    full = load_config(None).generation
    sub = full.model_copy(update={"seasonal_periods": [12]})
    ids_full = {make_dataset_id(s, full) for s in DatasetGenerator(full).iter_specs()}
    ids_sub = {make_dataset_id(s, sub) for s in DatasetGenerator(sub).iter_specs()}
    assert ids_sub <= ids_full


def test_components_reconstruct_observed(gen):
    for s in list(gen.iter_specs())[::37]:
        d = gen.generate(s)
        t = d.truth
        if s.seasonality_type == "additive":
            assert np.allclose(d.observed, t.trend + t.seasonal + t.noise) and t.noise_kind == "additive"
        else:
            assert np.allclose(d.observed, t.trend * t.seasonal * t.noise) and t.noise_kind == "multiplicative_factor"
            assert (t.noise > 0).all() and (t.seasonal > 0).all() and (d.observed > 0).all()


def test_seasonal_component_periodic_and_zero_mean(gen):
    for s in list(gen.iter_specs())[::53]:
        m = s.seasonal_period
        seas = gen.generate(s).truth.seasonal
        seas = seas if s.seasonality_type == "additive" else seas - 1
        assert np.allclose(seas[:m], seas[m:2 * m])
        assert abs(seas[:m].mean()) < 1e-9


def test_no_trend_is_constant_and_linear_slope(gen):
    for s in list(gen.iter_specs()):
        if s.length == 100 and s.noise_level == "low" and s.seasonality_type == "additive":
            tr = gen.generate(s).truth.trend
            if s.trend_type == "none":
                assert np.all(tr == 100.0)
            else:
                assert np.allclose(np.diff(tr), 0.3)
            break


def test_noise_levels_ordered(gen):
    sd = {}
    for s in gen.iter_specs():
        if (s.seasonality_type, s.seasonal_period, s.trend_type, s.length, s.replication) == ("additive", 12, "none", 500, 0):
            sd[s.noise_level] = gen.generate(s).truth.noise.std()
    assert sd["low"] < sd["medium"] < sd["high"]
