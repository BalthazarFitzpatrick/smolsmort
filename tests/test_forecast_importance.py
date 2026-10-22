"""synthetic fitted-model contributions and durable finish-time family output"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast.families import get_family
from smolsmort.forecast.features import Features
from smolsmort.forecast.importance import combine_importance, measure_importance
from smolsmort.forecast.pipeline import make_genome
from smolsmort.forecast.spec import Column, PrepSpec


@pytest.mark.parametrize("family,objective", [("xgboost", "squared"), ("lightgbm", "regression")])
def test_tree_gain_uses_pipeline_names_and_normalizes_fitted_scores(family, objective):
    pytest.importorskip(family)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 3))
    y = 12 + 5 * x[:, 0] - 0.5 * x[:, 1]
    names = ["lag_1", "roll_mean_4", "calendar_month"]
    fitted = get_family(family).fit(
        x, objective=objective, y=y, max_rounds=30, patience=0, nthread=1
    )
    scores = measure_importance(family, fitted, names, x)
    assert set(scores) == set(names)
    assert sum(scores.values()) == pytest.approx(1)
    result = combine_importance(family, [scores])
    assert result["method"] == "gain"
    assert result["features"][0]["feature"] == "lag_1"
    if family == "xgboost":
        raw = fitted.fitted.booster.get_score(importance_type="gain")
        expected = [raw.get(f"f{i}", 0) for i in range(3)]
    else:
        expected = fitted.booster.feature_importance(importance_type="gain")
    np.testing.assert_allclose(list(scores.values()), np.asarray(expected) / np.sum(expected))


def test_ridge_importance_uses_absolute_standardized_coefficients():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(240, 2)) * [100, 0.01]
    standardized = (x - x.mean(axis=0)) / x.std(axis=0)
    fitted = get_family("ridge").fit(
        x, y=4 * standardized[:, 0] - standardized[:, 1] + 8, params={"alpha": 1e-8}
    )
    scores = measure_importance("ridge", fitted, ["lag_1", "lag_2"], x)
    np.testing.assert_allclose(list(scores.values()), [0.8, 0.2], atol=1e-9)
    assert combine_importance("ridge", [scores])["method"] == "absolute_standardized_coefficient"


def test_ridge_reunites_missing_indicators_and_categories_under_pipeline_names():
    rng = np.random.default_rng(18)
    numeric = rng.normal(size=180)
    numeric[::3] = np.nan
    categories = np.tile([0, 0, 1, 2, 2, np.nan], 30)
    x = np.column_stack([numeric, categories])
    y = np.nan_to_num(numeric) + 3 * np.isnan(numeric) + 4 * (categories == 2)
    fitted = get_family("ridge").fit(x, y=y, feature_types=["q", "c"])
    scores = measure_importance("ridge", fitted, ["lag_1", "dimension_group"], x)
    scaled = np.abs(fitted.coefficients) * fitted.transform(x).std(axis=0)
    expected = np.array([scaled[0] + scaled[1], scaled[2:].sum()])
    np.testing.assert_allclose(list(scores.values()), expected / expected.sum())
    assert set(scores) == {"lag_1", "dimension_group"}


def test_bucket_aggregation_keeps_top_twenty_and_renormalizes():
    scores = {f"lag_{i:02}": float(i + 1) / 325 for i in range(25)}
    result = combine_importance("xgboost", [scores, {"lag_00": 1.0}])
    features = result["features"]
    assert len(features) == 20
    assert features[0]["feature"] == "lag_00"
    assert {entry["feature"] for entry in features[1:]} == {f"lag_{i:02}" for i in range(6, 25)}
    assert sum(entry["importance"] for entry in features) == pytest.approx(1)
    assert features[0]["importance"] / features[1]["importance"] == pytest.approx(326 / 25)


@pytest.mark.parametrize("family", ["snaive", "ets", "theta", "arima"])
def test_raw_series_families_have_no_importance(family):
    assert measure_importance(family, None, []) is None
    assert combine_importance(family, []) is None


def test_constant_model_has_no_fabricated_importance():
    booster = SimpleNamespace(get_score=lambda **kwargs: {})
    scores = measure_importance("xgboost", SimpleNamespace(booster=booster), ["lag_1"])
    assert scores == {"lag_1": 0}
    assert combine_importance("xgboost", [scores]) == {"method": "gain", "features": []}


@pytest.mark.parametrize("ensemble_enabled", [False, True])
def test_finish_persists_all_family_winners_and_reuses_final_fits(
    tmp_path, monkeypatch, ensemble_enabled
):
    pytest.importorskip("duckdb")
    pytest.importorskip("xgboost")
    from smolsmort.forecast import ensemble, pipeline

    rng = np.random.default_rng(51)
    x = rng.uniform(10, 20, size=(121, 2)).astype(np.float32)
    y = 2 * x[:, 0] + x[:, 1]
    y[-1] = np.nan
    position = np.arange(121)
    masks = {
        "train": position < 80,
        "val": (position >= 80) & (position < 100),
        "test": (position >= 100) & (position < 120),
        "future": position == 120,
    }
    frame = SimpleNamespace(
        y=y,
        step=position,
        series=np.zeros(len(y), dtype=int),
        features=Features(
            x, ["lag_1", "lag_2"], ["q", "q"], ["value@step:lag+0", "value@step:lag+1"]
        ),
    )
    spec = PrepSpec(
        "synthetic.csv",
        "timeseries",
        "regression",
        (Column("step", "time"), Column("value", "target", aggregation="sum")),
    )
    ws = SimpleNamespace(
        mode="timeseries",
        spec=spec,
        profile={"genes": {}},
        frames={(1, 1): frame},
        series_masks={(1, 1): masks},
        test_steps=20,
        excluded_series=[],
    )
    ranked = [
        {
            "family": name,
            "fitness": fitness,
            "genome": make_genome(
                ["value@step:lag+0", "value@step:lag+1"], "squared", {}, name
            ).to_dict(),
            "prediction": (y[masks["val"]] + fitness).tolist(),
        }
        for name, fitness in (("ridge", 0.1), ("xgboost", 0.2), ("snaive", 0.3))
    ]
    states = {name: {"state": "done"} for name in ("ridge", "xgboost", "snaive")}
    request = {
        "spec": asdict(spec),
        "prepared": str(tmp_path),
        "summary": {},
        "ensemble": {"enabled": ensemble_enabled, "top": 2},
    }
    for name in states:
        (tmp_path / "families" / name).mkdir(parents=True)
    for filename, data in (
        ("request.json", request),
        ("finalists.json", ranked),
        ("coordinator.json", {"families": states}),
    ):
        (tmp_path / filename).write_text(json.dumps(data))
    calls = []

    def fit_model(ws, genome, frame, rows, nthread):
        calls.append((genome.family, int(rows.sum())))
        fitted = get_family(genome.family).fit(
            x[rows], y=y[rows], objective="squared", max_rounds=15, patience=0
        )
        return fitted, x

    monkeypatch.setattr(pipeline, "_timeseries_fit", fit_model)
    monkeypatch.setattr(ensemble, "load_workspace", lambda *args, **kwargs: ws)
    monkeypatch.setattr(ensemble, "watch_parent", lambda pid: None)
    ensemble.run(tmp_path, 1)
    result = json.loads((tmp_path / "result.json").read_text())
    for name, method in (("ridge", "absolute_standardized_coefficient"), ("xgboost", "gain")):
        importance = json.loads((tmp_path / "families" / name / "importance.json").read_text())
        assert importance == result["families"][name]["importance"]
        assert importance["method"] == method
        assert {entry["feature"] for entry in importance["features"]} <= {"lag_1", "lag_2"}
        assert sum(entry["importance"] for entry in importance["features"]) == pytest.approx(1)
        assert calls.count((name, 120)) == 1
        assert "verdict" not in result["families"][name]
    assert result["families"]["snaive"]["importance"] is None
    assert not (tmp_path / "families" / "snaive" / "importance.json").exists()
    assert calls.count(("ridge", 100)) == 1
    assert calls.count(("xgboost", 100)) == int(ensemble_enabled)
    assert result["winner"]["importance"] == result["families"]["ridge"]["importance"]
    if ensemble_enabled:
        assert result["importance"] is None
        assert not (tmp_path / "importance.json").exists()
    else:
        assert json.loads((tmp_path / "importance.json").read_text()) == result["importance"]


def test_v1_result_keeps_importance_without_changing_recipe_family(tmp_path):
    from smolsmort.forecast.worker import _write_result

    spec = PrepSpec(
        "synthetic.csv",
        "entity",
        "regression",
        (Column("step", "anchor"), Column("value", "target")),
    )
    importance = {"method": "gain", "features": [{"feature": "value_lag", "importance": 1.0}]}
    genome = make_genome(["lag"], "squared", {}).to_dict()
    _write_result(
        tmp_path,
        spec,
        {"best": genome, "stopped": "complete"},
        {"verdict": {}, "model_error": 0.1, "baseline_error": 0.2, "importance": importance},
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["importance"] == importance
    assert json.loads((tmp_path / "importance.json").read_text()) == importance
    assert "family" not in json.loads((tmp_path / "recipe.json").read_text())["genome"]
