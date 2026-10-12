"""ridge transforms, closed-form coefficients and production family seams"""

import json

import numpy as np
import pytest
from forecast_fixtures import write_panel

from smolsmort.forecast.families import get_family
from smolsmort.forecast.model import ModelError
from smolsmort.forecast.pipeline import load_workspace, make_genome, score
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.runs import start_run, wait_run
from smolsmort.forecast.spec import Column, PrepSpec


@pytest.fixture
def family():
    return get_family("ridge")


@pytest.fixture
def panel(tmp_path):
    pytest.importorskip("duckdb")
    truth = write_panel(tmp_path)
    spec = PrepSpec(
        str(truth.path),
        "series",
        "regression",
        (
            Column("day", "time"),
            Column("project", "dimension"),
            Column("product", "dimension"),
            Column("orders", "measure", aggregation="sum"),
            Column("units", "target", aggregation="sum"),
        ),
    )
    return spec, prepare(spec, tmp_path / "cache")


def test_family_contract(family):
    assert family.available() == (True, "")
    assert family.pip_extra == "" and family.needs == "lag_features"
    assert family.objectives("regression") == ("squared",)
    assert family.objectives("classification") == ()
    assert family.objective_aliases == {"squared": "squared"}
    (param,) = family.space()
    assert (param.name, param.kind, param.lo, param.hi, param.default) == (
        "alpha",
        "log",
        1e-4,
        1e4,
        1.0,
    )
    assert family.defaults() == {"alpha": 1.0}


def test_recovers_coefficients_and_unpenalised_intercept(family):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    coefficients = np.array([1.2, -3.0, 0.5, 2.0])
    y = x.astype(float) @ coefficients + 12.0
    fitted = family.fit(x, y=y, params={"alpha": 1e-12})
    recovered = fitted.coefficients / fitted.scales
    intercept = fitted.intercept - fitted.means @ recovered
    np.testing.assert_allclose(recovered, coefficients, rtol=0, atol=1e-6)
    assert intercept == pytest.approx(12.0, abs=1e-6)
    np.testing.assert_allclose(fitted.predict(x), y, rtol=0, atol=1e-6)
    shrunk = family.fit(x, y=np.full(200, 12.0), params={"alpha": 1e4})
    np.testing.assert_array_equal(shrunk.predict(x), 12.0)


def test_deterministic_with_missing_categories_and_row_order(family):
    x = np.column_stack([np.tile([0, 1, 2, np.nan], 50), np.arange(200)]).astype("float32")
    x[::7, 1] = np.nan
    y = np.tile([10, 30, 20, 15], 50).astype(float)
    options = {"y": y, "feature_types": ["c", "q"], "seed": 17}
    first, second = family.fit(x, **options), family.fit(x, **options)
    values = first.predict(x)
    np.testing.assert_array_equal(values, second.predict(x))
    order = np.random.default_rng(4).permutation(len(x))
    np.testing.assert_array_equal(first.predict(x[order]), values[order])
    assert first.eval_history is None
    predicted = first.predict(np.array([[99, np.nan], [np.nan, np.nan]], dtype="float32"))
    assert np.isfinite(predicted).all()
    encoded = first.transform([[99, np.nan], [np.nan, np.nan]])[:, -4:]
    np.testing.assert_array_equal(encoded[0], encoded[1])
    assert np.abs(values - y).mean() < 0.5


def test_transforms_use_only_usable_training_rows(family):
    x = np.array([[1, 0], [np.nan, 1], [5, np.nan], [999, 99]], dtype="float32")
    fitted = family.fit(x, y=[1, 2, 3, np.nan], feature_types=["q", "c"])
    np.testing.assert_array_equal(fitted.medians, [3])
    np.testing.assert_array_equal(fitted.means, [3])
    np.testing.assert_array_equal(fitted.missing, [0, 1])
    np.testing.assert_array_equal(fitted.categories[0][1], [0, 1])
    predicted = fitted.transform([[np.nan, 99], [9, np.nan]])
    np.testing.assert_array_equal(predicted[:, 1:3], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(predicted[:, 3:], [[0, 0, 1], [0, 0, 1]])
    assert predicted[0, 0] == 0
    assert predicted[1, 0] == pytest.approx(6 / np.std([1, 3, 5]))
    np.testing.assert_array_equal(fitted.medians, [3])


def test_unseen_category_has_reserved_column_without_training_nans(family):
    fitted = family.fit([[0], [1], [0], [1]], y=[1, 2, 1, 2], feature_types=["c"])
    np.testing.assert_array_equal(fitted.transform([[99], [np.nan]]), [[0, 0, 1], [0, 0, 1]])
    assert np.isfinite(fitted.predict([[99], [np.nan]])).all()


def test_all_missing_and_constant_columns(family):
    x = np.array([[np.nan, 4, np.nan]] * 30, dtype="float32")
    fitted = family.fit(x, y=np.arange(30), feature_types=["q", "q", "c"])
    assert np.isfinite(fitted.transform(x)).all()
    np.testing.assert_array_equal(fitted.predict(x), 14.5)
    assert np.isfinite(fitted.predict([[3, 4, 99]])).all()


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf])
def test_invalid_alpha_is_candidate_failure(family, alpha):
    with pytest.raises(ModelError, match="alpha"):
        family.fit([[1], [2]], y=[1, 2], params={"alpha": alpha})


def test_invalid_inputs_are_candidate_failures(family):
    with pytest.raises(ModelError, match="objective"):
        family.fit([[1]], y=[1], objective="absolute")
    with pytest.raises(ModelError, match="usable target"):
        family.fit([[1]], y=[np.nan])
    with pytest.raises(ModelError, match="type marker"):
        family.fit([[1]], y=[1], feature_types=[])
    with pytest.raises(ModelError, match="finite or nan"):
        family.fit([[np.inf]], y=[1])
    fitted = family.fit([[1]], y=[1])
    with pytest.raises(ModelError, match="column count"):
        fitted.predict([[1, 2]])


def test_genome_scores_through_pipeline(family, panel):
    spec, prepared = panel
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    genome = make_genome(ws.families, "squared", family.defaults(), family="ridge")
    result = score(ws, genome)
    assert np.isfinite(result.fitness)
    assert result.fitness == pytest.approx(result.contrib.sum())


def test_v2_xgboost_ridge_snaive_finish_and_merge(panel, tmp_path):
    pytest.importorskip("xgboost")
    spec, prepared = panel
    selections = [
        {"name": "xgboost", "method": "grid", "space": {"max_depth": [3], "eta": [0.1]}},
        {"name": "ridge", "method": "grid", "space": {"alpha": [0.1, 1.0, 10.0]}},
        {"name": "snaive"},
    ]
    runs = tmp_path / "runs"
    run_id = start_run(runs, spec, prepared, families=selections, time_budget_s=30, nthread=3)
    state = wait_run(runs, run_id, timeout=120)
    assert state["state"] == "done", state
    assert set(state["families"]) == {"xgboost", "ridge", "snaive"}
    assert all(status["state"] == "done" for status in state["families"].values()), state
    board = json.loads((runs / run_id / "leaderboard.json").read_text())
    assert {entry["family"] for entry in board} == {"xgboost", "ridge", "snaive"}
    ridge = [entry for entry in board if entry["family"] == "ridge"]
    assert ridge and all(entry["fitness"] is not None for entry in ridge), ridge
    assert all(entry["genome"]["objective"] == "squared" for entry in ridge)
    assert {entry["genome"]["params"]["alpha"] for entry in ridge} == {0.1, 1.0, 10.0}
