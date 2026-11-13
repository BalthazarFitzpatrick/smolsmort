"""the new families run through a real search, finish and ensemble; slow, so a few minutes each"""

from __future__ import annotations

import pytest
from test_forecast_coordinator import assert_final_artifacts, make_trending_panel, wait_run

from smolsmort.forecast.families import get_family
from smolsmort.forecast.runs import start_run


def run_families(tmp_path, names, budget_s):
    spec, prepared = make_trending_panel(tmp_path)
    root = tmp_path / "runs"
    run_id = start_run(
        root,
        spec,
        prepared,
        families=[{"name": name} for name in names],
        time_budget_s=budget_s,
        nthread=max(2, len(names)),
        ensemble={"enabled": True, "top": 3},
    )
    status = wait_run(root, run_id, timeout=budget_s + 10)
    assert status["state"] == "done", status
    return assert_final_artifacts(root / run_id, names)


@pytest.mark.skipif(
    not all(get_family(name).available()[0] for name in ("extratrees", "mlp", "lstm")),
    reason="needs scikit-learn and torch",
)
def test_extratrees_mlp_and_lstm_share_one_run_with_xgboost(tmp_path):
    names = ["xgboost", "extratrees", "mlp", "lstm", "snaive"]
    result = run_families(tmp_path, names, 240)
    for name in ("extratrees", "mlp", "lstm"):
        state = result["families"][name]
        assert state["fitness"] is not None, (name, state)
        assert state["state"] != "failed", (name, state)


@pytest.mark.skipif(not get_family("catboost").available()[0], reason="needs catboost")
def test_catboost_runs_alone_with_the_baseline(tmp_path):
    # one catboost candidate is several times slower than xgboost, so it gets a roomy slice
    result = run_families(tmp_path, ["catboost", "snaive"], 150)
    state = result["families"]["catboost"]
    assert state["fitness"] is not None, state
    assert state["state"] != "failed", state
