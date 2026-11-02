import numpy as np
import pandas as pd
import pytest

from forecasting_research.analysis.aggregation import aggregate_metrics, normalize_dims
from forecasting_research.analysis.comparison import best_method_per_group, win_counts
from forecasting_research.domain.exceptions import ConfigError
from forecasting_research.experiments.lifecycle import (
    IllegalTransition,
    assert_evaluation_transition,
    derive_experiment_status,
    describe_status,
)

C = lambda **k: {"PENDING": 0, "RUNNING": 0, "COMPLETED": 0, "FAILED": 0, "INTERRUPTED": 0, **k}  # noqa: E731


@pytest.mark.parametrize("counts,kw,expected", [
    (C(COMPLETED=6), {}, ("COMPLETED", "SUCCESS")),
    (C(COMPLETED=5, FAILED=1), {}, ("COMPLETED", "PARTIAL")),
    (C(FAILED=6), {}, ("FAILED", "ALL_FAILED")),
    (C(COMPLETED=3, PENDING=3), {}, ("INTERRUPTED", "INCOMPLETE")),
    (C(COMPLETED=3, INTERRUPTED=1), {}, ("INTERRUPTED", "INCOMPLETE")),
    (C(COMPLETED=6), {"interrupted": True}, ("INTERRUPTED", "INCOMPLETE")),
    (C(COMPLETED=3, PENDING=3), {"aborted": True}, ("FAILED", "ABORTED")),
])
def test_final_status_mapping(counts, kw, expected):
    s, o = derive_experiment_status(counts, **kw)
    assert (s.value, o.value) == expected


def test_partial_never_displayed_as_plain_success():
    assert "partial" in describe_status("COMPLETED", "PARTIAL", 2)
    assert describe_status("COMPLETED", "SUCCESS") == "COMPLETED"


def test_illegal_transitions():
    assert_evaluation_transition("PENDING", "RUNNING")
    assert_evaluation_transition("INTERRUPTED", "RUNNING")
    for old, new in [("PENDING", "COMPLETED"), ("COMPLETED", "FAILED"), ("PENDING", "FAILED")]:
        with pytest.raises(IllegalTransition):
            assert_evaluation_transition(old, new)


def _df():
    rows = []
    for i, (model, rmse, status) in enumerate([("a", 1.0, "COMPLETED"), ("a", 3.0, "COMPLETED"), ("b", 2.5, "COMPLETED"),
                                               ("b", np.nan, "FAILED"), ("c", 5.0, "COMPLETED"), ("c", 6.0, "COMPLETED")]):
        rows.append(dict(evaluation_id=f"e{i}", model_name=model, status=status, seasonality_type="additive", seasonal_period=4,
                         trend_type="none", noise_level="low", length=100, MAE=rmse, MSE=rmse ** 2, RMSE=rmse,
                         MAPE=np.nan if i == 0 else rmse))
    return pd.DataFrame(rows)


def test_aggregation_counts_and_stats():
    agg = aggregate_metrics(_df(), ["seasonality"], ["RMSE", "MAPE"]).set_index("model_name")
    assert agg.loc["a", "RMSE_mean"] == 2.0 and agg.loc["a", "RMSE_median"] == 2.0
    assert agg.loc["b", "n_total"] == 2 and agg.loc["b", "n_completed"] == 1 and agg.loc["b", "n_failed"] == 1
    assert agg.loc["b", "RMSE_n_valid"] == 1
    assert agg.loc["a", "MAPE_n_valid"] == 1 and agg.loc["a", "MAPE_n_undefined"] == 1  # reported, not hidden


def test_best_method_and_coverage_flag():
    agg = aggregate_metrics(_df(), ["seasonality"])
    best = best_method_per_group(agg, ["seasonality"], "RMSE")
    assert best.iloc[0]["best_model"] == "a" and best.iloc[0]["runner_up_model"] == "b"
    assert bool(best.iloc[0]["complete_coverage"]) is False and best.iloc[0]["n_failed_total"] == 1
    assert win_counts(best).iloc[0]["groups_won"] == 1


def test_dimension_aliases_and_errors():
    assert normalize_dims("noise,period") == ["noise_level", "seasonal_period"]
    with pytest.raises(ConfigError):
        normalize_dims(["bogus"])
