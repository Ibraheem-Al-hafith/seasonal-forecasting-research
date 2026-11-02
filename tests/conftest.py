import pytest

from forecasting_research.application.services import ResearchApp
from forecasting_research.config.loader import config_from_dict


def make_cfg(**over):
    data = {
        "generation": {
            "seasonality_types": ["additive", "multiplicative"], "seasonal_periods": [4], "trends": ["linear"],
            "noise_levels": ["low"], "dataset_lengths": [100], "replications": 2,
        },
        "forecasting": {"sarima": {"max_pq_sum": 1, "max_seasonal_pq_sum": 1}},
    }
    for k, v in over.items():
        data.setdefault(k, {}).update(v)
    return config_from_dict(data)


@pytest.fixture
def cfg():
    return make_cfg()


@pytest.fixture
def app(tmp_path, cfg):
    return ResearchApp(tmp_path / "ws", cfg)
