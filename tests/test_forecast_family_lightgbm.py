"""lightgbm uses the production feature matrices and search score path"""

import subprocess
import sys

import numpy as np
import pytest
from forecast_fixtures import write_bike_daily, write_panel

from smolsmort.forecast.families import base, get_family
from smolsmort.forecast.model import ModelError
from smolsmort.forecast.pipeline import _series_fit, load_workspace, make_genome, score
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.spec import Column, PrepSpec


def test_registration_and_lazy_import():
    family = get_family("lightgbm")
    assert family.needs == "lag_features"
    assert family.pip_extra == "lightgbm"
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import smolsmort.forecast.families; assert 'lightgbm' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_missing_library_names_install_command(monkeypatch):
    def refuse_import(name):
        assert name == "lightgbm"
        raise ImportError("missing library")

    monkeypatch.setattr(base, "import_module", refuse_import)
    available, command = get_family("lightgbm").available()
    assert not available
    assert command.startswith("uv sync --extra lightgbm")


def test_typed_space_and_defaults():
    family = get_family("lightgbm")
    specs = {param.name: param for param in family.space()}
    assert specs["num_leaves"].kind == "log"
    assert isinstance(specs["num_leaves"].default, int)
    assert specs["learning_rate"].kind == "log"
    assert specs["lambda_l2"].kind == "log"
    assert specs["min_child_samples"].kind == "int"
    assert specs["max_depth"].lo == -1
    assert specs["objective"].choices == ("regression_l1", "regression", "poisson", "tweedie")
    assert family.defaults()["objective"] == "regression_l1"


@pytest.fixture
def family():
    pytest.importorskip("lightgbm")
    return get_family("lightgbm")


def test_daily_nan_features_fit_and_predict(family, tmp_path):
    pytest.importorskip("duckdb")
    source = write_bike_daily(tmp_path)
    spec = PrepSpec(
        str(source),
        "series",
        "regression",
        (Column("dteday", "time"), Column("cnt", "target", aggregation="sum")),
        step="day",
        horizon=30,
    )
    prepared = prepare(spec, tmp_path / "cache")
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    bucket = next(iter(ws.frames))
    frame = ws.frames[bucket]
    genome = make_genome(ws.families, "regression_l1", family.defaults(), family="lightgbm")
    fitted, x = _series_fit(ws, genome, frame, ws.series_masks[bucket]["train"], 1)
    assert x.dtype == np.float32
    assert np.isnan(x[ws.series_masks[bucket]["train"]]).any()
    values = fitted.predict(x[ws.series_masks[bucket]["val"]])
    assert values.shape == frame.y[ws.series_masks[bucket]["val"]].shape
    assert np.isfinite(values).all()
    history = fitted.eval_history
    assert set(history) == {"metric", "training", "validation"}
    assert len(history["training"]) == len(history["validation"])
    assert 1 <= fitted.rounds <= len(history["validation"])


def test_categories_determinism_and_row_order(family):
    x = np.column_stack([np.tile([0, 1, 2, np.nan], 50), np.arange(200)]).astype("float32")
    y = np.tile([10, 30, 20, 15], 50).astype(float)
    options = {
        "objective": "regression",
        "y": y,
        "feature_types": ["c", "q"],
        "seed": 17,
        "nthread": 1,
        "params": {"min_child_samples": 5, "feature_fraction": 0.8, "bagging_fraction": 0.8},
        "max_rounds": 150,
        "patience": 15,
    }
    first = family.fit(x, **options)
    second = family.fit(x, **options)
    values = first.predict(x)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values, second.predict(x))
    order = np.random.default_rng(4).permutation(len(x))
    np.testing.assert_array_equal(first.predict(x[order]), values[order])
    assert np.abs(values - y).mean() < 1
    assert first.booster.params["deterministic"] is True
    assert first.booster.params["num_threads"] == 1
    assert first.booster.params["seed"] == 17


def test_multithread_determinism_in_separate_family_process(family):
    code = """
import numpy as np
from smolsmort.forecast.families import get_family
x = np.column_stack([np.tile([0, 1, 2, np.nan], 50), np.arange(200)]).astype('float32')
y = np.tile([10, 30, 20, 15], 50).astype(float)
family = get_family('lightgbm')
options = {'y': y, 'objective': 'regression', 'feature_types': ['c', 'q'],
           'seed': 17, 'nthread': 2, 'max_rounds': 100, 'patience': 10,
           'params': {'min_child_samples': 5, 'bagging_fraction': 0.8}}
first = family.fit(x, **options)
second = family.fit(x, **options)
assert first.booster.params['num_threads'] == 2
np.testing.assert_array_equal(first.predict(x), second.predict(x))
"""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr


def test_newest_rows_select_rounds_then_all_usable_rows_refit(family, monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    train = lgb.train
    calls = []

    def record_train(config, data, **options):
        calls.append((data.data.copy(), options))
        return train(config, data, **options)

    monkeypatch.setattr(lgb, "train", record_train)
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    y = np.sin(np.arange(100)) + 2
    y[3] = np.nan
    fitted = family.fit(x, y=y, max_rounds=30, patience=5)
    usable = x[np.isfinite(y)]
    cut = int(len(usable) * 0.85)
    np.testing.assert_array_equal(calls[0][0], usable[:cut])
    np.testing.assert_array_equal(calls[1][0], usable)
    assert calls[1][1]["num_boost_round"] == fitted.rounds
    assert len(fitted.eval_history["validation"]) <= 30


@pytest.mark.parametrize("objective", ["regression_l1", "regression", "poisson", "tweedie"])
def test_thin_training_skips_probe_and_refits_all_rows(family, objective):
    x = np.arange(60, dtype=np.float32).reshape(30, 2)
    fitted = family.fit(x, y=np.arange(30) + 1, objective=objective, max_rounds=7)
    assert fitted.rounds == 7
    assert fitted.eval_history is None
    assert np.isfinite(fitted.predict(x)).all()
    with pytest.raises(ModelError, match="too few to fit"):
        family.fit(x[:19], y=np.arange(19), objective=objective)


@pytest.mark.parametrize("objective", ["poisson", "tweedie"])
def test_nonnegative_objectives_refuse_negative_targets(family, objective):
    with pytest.raises(ModelError, match="non-negative target"):
        family.fit(np.ones((30, 2)), y=np.arange(30) - 1, objective=objective)


def test_genome_runs_through_production_score(family, tmp_path):
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
    prepared = prepare(spec, tmp_path / "cache")
    ws = load_workspace(spec, prepared.folder, prepared.summary)
    genome = make_genome(ws.families, "regression_l1", family.defaults(), family="lightgbm")
    result = score(ws, genome)
    assert np.isfinite(result.fitness)
    assert result.fitness == pytest.approx(result.contrib.sum())
