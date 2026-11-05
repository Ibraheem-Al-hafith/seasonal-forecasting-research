import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from forecasting_research.analysis.reliability import is_extreme_forecast
from forecasting_research.cli.app import app as cli_app
from forecasting_research.domain.exceptions import ConfigError

from ..conftest import make_cfg


def test_extreme_forecast_definition():
    train = np.linspace(90, 110, 50)  # range 20
    assert not is_extreme_forecast(train, np.array([85.0, 115.0, 100.0]))
    assert is_extreme_forecast(train, np.array([100.0, 131.0]))  # more than one range above the maximum
    assert is_extreme_forecast(train, np.array([69.0]))


def test_reliability_and_search_tables_from_stored_results(app):
    eid = app.run()
    rel = app.reliability([eid]).set_index("model_name")
    assert set(rel.index) == {"classical_decomposition", "holt_winters", "sarima"}
    assert (rel["n_total"] == 4).all() and (rel["success_pct"] == 100).all() and (rel["n_failed"] == 0).all()
    assert (rel["extreme_forecasts"] == 0).all() and (rel["total_runtime_s"] > 0).all()
    by_season = app.reliability([eid], ["seasonality"])
    assert len(by_season) == 6  # 2 seasonalities x 3 models
    s = app.sarima_search([eid]).iloc[0]
    assert s["n_searches"] == 4 and s["n_failed"] == 0 and s["mean_fits"] >= 4 and s["method_versions"] == "2"
    assert s["pct_truncated"] == 0 and s["mean_selectable_pct"] > 0 and s["pct_D1"] == 100  # default D fixed at 1
    assert 0 <= s["pct_stage2"] <= 100 and s["mean_seasonal_strength"] > 0
    grouped = app.sarima_search([eid], ["seasonality"])
    assert set(grouped["seasonality_type"]) == {"additive", "multiplicative"}


def test_failed_evaluations_lower_success_rate_and_are_never_dropped(app):
    from tests.helpers import AlwaysFails, Registered
    app.generate()
    with Registered(always_fails=AlwaysFails):
        eid = app.evaluate(make_cfg(forecasting={"methods": ["holt_winters", "always_fails"]}))
    rel = app.reliability([eid]).set_index("model_name")
    assert rel.loc["always_fails", "success_pct"] == 0 and rel.loc["always_fails", "n_failed"] == 4
    assert rel.loc["holt_winters", "success_pct"] == 100


def test_export_and_cli_views(app, tmp_path):
    eid = app.run()
    for kind in ("reliability", "search"):
        assert len(pd.read_csv(app.export_results(eid, kind))) >= 1
    r = CliRunner()
    base = ["-w", str(app.workspace)]
    wide = {"COLUMNS": "220"}  # the tables are wide; a narrow terminal folds headers
    out = r.invoke(cli_app, base + ["analyze", eid, "--view", "reliability"], env=wide)
    assert out.exit_code == 0 and "success_pct" in out.output and "sarima" in out.output, out.output
    out = r.invoke(cli_app, base + ["analyze", eid, "--view", "search", "--group-by", "seasonality"], env=wide)
    assert out.exit_code == 0 and "pct_stage2" in out.output, out.output


def _make_legacy(app, eid):
    """Rewrite an experiment so it looks like one created by SARIMA method v1 (no method_version in its config)."""
    with sqlite3.connect(app.db.path) as c:
        cfg = json.loads(c.execute("SELECT config_json FROM experiments WHERE experiment_id=?", (eid,)).fetchone()[0])
        cfg["forecasting"]["sarima"] = {k: v for k, v in cfg["forecasting"]["sarima"].items() if k in ("max_pq_sum", "max_seasonal_pq_sum")}
        c.execute("UPDATE experiments SET config_json=? WHERE experiment_id=?", (json.dumps(cfg), eid))


def test_resume_refuses_to_mix_sarima_method_versions(app):
    eid = app.run()
    ev = next(e for e in app.repo.evaluations.list_for_experiment(eid) if e["model_name"] == "sarima")
    with sqlite3.connect(app.db.path) as c:  # one SARIMA evaluation still to do
        c.execute("UPDATE model_evaluations SET status='INTERRUPTED' WHERE evaluation_id=?", (ev["evaluation_id"],))
        c.execute("UPDATE experiments SET status='INTERRUPTED' WHERE experiment_id=?", (eid,))
    _make_legacy(app, eid)
    with pytest.raises(ConfigError, match="SARIMA method v1"):
        app.resume(eid)
    assert app.repo.evaluations.get(ev["evaluation_id"])["status"] == "INTERRUPTED"  # nothing was run


def test_legacy_experiment_without_pending_sarima_work_can_still_be_resumed(app):
    cfg = make_cfg(forecasting={"methods": ["classical_decomposition", "sarima"]})
    eid = app.run(cfg)
    ev = next(e for e in app.repo.evaluations.list_for_experiment(eid) if e["model_name"] == "classical_decomposition")
    with sqlite3.connect(app.db.path) as c:
        c.execute("UPDATE model_evaluations SET status='INTERRUPTED' WHERE evaluation_id=?", (ev["evaluation_id"],))
        c.execute("UPDATE experiments SET status='INTERRUPTED' WHERE experiment_id=?", (eid,))
    _make_legacy(app, eid)
    app.resume(eid)  # only a non-SARIMA evaluation remains: no mixing, so it is allowed
    assert app.repo.evaluations.get(ev["evaluation_id"])["status"] == "COMPLETED"
