import pytest

from forecasting_research.config.loader import config_from_dict, config_from_yaml_text, load_config
from forecasting_research.domain.exceptions import ConfigError


def test_defaults_match_blueprint():
    c = load_config(None)
    assert c.generation.combination_count() == 108
    assert c.generation.dataset_count() == 540
    assert c.generation.dataset_count() * len(c.forecasting.methods) == 1620
    assert c.forecasting.sarima.candidate_count() == 18


def test_default_yaml_file_valid():
    assert load_config("configs/default.yaml").generation.dataset_count() == 540


@pytest.mark.parametrize("bad", [
    {"generation": {"replications": 0}},
    {"generation": {"seasonal_periods": []}},
    {"generation": {"seasonal_periods": [4, 4]}},
    {"generation": {"seasonal_periods": [1]}},
    {"evaluation": {"train_fraction": 1.2}},
    {"evaluation": {"forecast_horizon": 50}},          # exceeds test segment
    {"generation": {"dataset_lengths": [20], "seasonal_periods": [12]}},  # < 2 training cycles
    {"unknown_section": {}},
    {"generation": {"typo_field": 1}},
])
def test_invalid_rejected(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_yaml_errors():
    with pytest.raises(ConfigError):
        config_from_yaml_text("a: [unclosed")
    with pytest.raises(ConfigError):
        config_from_yaml_text("- a list")
    with pytest.raises(ConfigError):
        load_config("/nonexistent.yaml")
