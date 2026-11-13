"""catboost family on tiny synthetic data, instantiated directly"""

import os

import numpy as np
import pytest

from smolsmort.forecast.families import base
from smolsmort.forecast.families.catboost import CatboostFamily
from smolsmort.forecast.model import ModelError

FAST = {"depth": 3.0, "learning_rate": 0.2}


def make_data(rows=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 4))
    y = 5 + 2 * x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=rows)
    return x, y


def fit(family, x, y, **kwargs):
    options = {"params": FAST, "max_rounds": 40, "patience": 10, "objective": "RMSE"}
    return family.fit(x, y=y, **{**options, **kwargs})


@pytest.fixture
def family():
    return CatboostFamily()


def test_space_and_defaults(family):
    specs = {param.name: param for param in family.space()}
    assert set(specs) == {"depth", "learning_rate", "l2_leaf_reg"}
    assert specs["depth"].kind == "int"
    assert specs["learning_rate"].kind == "log"
    assert family.defaults() == {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0}
    assert family.objectives("regression") == ("RMSE", "MAE", "Poisson", "Tweedie")
    assert family.objectives("classification") == ()
    assert family.cost((100, 10), {"depth": 4}) == 100 * 10 * 16


def test_fit_predict_shape(family):
    x, y = make_data()
    fitted = fit(family, x, y)
    assert fitted.predict(x).shape == (len(x),)
    assert fitted.rounds >= 1


def test_nan_features(family):
    x, y = make_data()
    x[::7, 2] = np.nan
    fitted = fit(family, x, y)
    assert np.isfinite(fitted.predict(x)).all()


def test_categorical_column(family):
    x, y = make_data()
    x[:, 3] = np.arange(len(x)) % 3
    x[::11, 3] = np.nan
    fitted = fit(family, x, y, feature_types=["n", "n", "n", "c"])
    assert fitted.predict(x).shape == (len(x),)


def test_nonfinite_target_rows_are_skipped(family):
    x, y = make_data()
    y[::10] = np.nan
    assert np.isfinite(fit(family, x, y).predict(x)).all()


def test_fixed_seed_is_repeatable(family):
    x, y = make_data()
    first = fit(family, x, y, seed=3).predict(x)
    second = fit(family, x, y, seed=3).predict(x)
    assert np.array_equal(first, second)


def test_missing_library_names_install_command(monkeypatch, family):
    def refuse_import(name):
        raise ImportError("missing library")

    monkeypatch.setattr(base, "import_module", refuse_import)
    available, hint = family.available()
    assert not available
    assert "uv sync --extra catboost" in hint


def test_poisson_rejects_negative_target(family):
    x, y = make_data()
    with pytest.raises(ModelError, match="non-negative"):
        fit(family, x, y, objective="Poisson")


def test_poisson_predicts_positive_means(family):
    x, y = make_data()
    fitted = fit(family, x, np.abs(y), objective="Poisson")
    assert (fitted.predict(x) > 0).all()


def test_unknown_objective_and_too_few_rows(family):
    x, y = make_data()
    with pytest.raises(ModelError):
        fit(family, x, y, objective="nope")
    with pytest.raises(ModelError, match="too few"):
        fit(family, x[:10], y[:10])


def test_eval_history_lengths_match(family):
    x, y = make_data()
    history = fit(family, x, y).eval_history
    assert history["metric"] == "RMSE"
    size = len(history["validation"])
    assert size > 0
    assert len(history["training"]) == size
    assert all(len(values) == size for values in history["curves"].values())


def test_round_errors_can_be_off(family):
    x, y = make_data()
    assert "curves" not in fit(family, x, y, round_errors=False).eval_history


def test_no_files_written_to_cwd(family, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x, y = make_data()
    fit(family, x, y)
    assert os.listdir(tmp_path) == []
