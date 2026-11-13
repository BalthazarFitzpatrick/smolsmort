"""extra trees family: fit, missing values, categories, determinism"""

from __future__ import annotations

import os

import numpy as np
import pytest

from smolsmort.forecast.families import base
from smolsmort.forecast.families.extratrees import ExtratreesFamily
from smolsmort.forecast.model import ModelError

PARAMS = {"n_estimators": 20.0, "max_depth": 5.4, "min_samples_leaf": 2.0, "max_features": 0.8}


def make_data(rows: int = 60):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(rows, 3))
    y = x[:, 0] * 2 + rng.normal(scale=0.1, size=rows)
    return x, y


def test_fit_predict_shape_and_history():
    x, y = make_data()
    fitted = ExtratreesFamily().fit(x, y=y, params=PARAMS)
    assert fitted.predict(x).shape == (60,)
    assert len(fitted.eval_history["training"]) == 1


def test_nan_features():
    x, y = make_data()
    x[::5, 1] = np.nan
    x[:, 2] = np.nan
    fitted = ExtratreesFamily().fit(x, y=y, params=PARAMS)
    test = x.copy()
    test[0, 0] = np.nan
    assert np.isfinite(fitted.predict(test)).all()


def test_unseen_category_level():
    x, y = make_data()
    x[:, 2] = np.arange(60) % 3
    fitted = ExtratreesFamily().fit(x, y=y, feature_types=["q", "q", "c"], params=PARAMS)
    test = x[:4].copy()
    test[:, 2] = 99.0
    assert np.isfinite(fitted.predict(test)).all()


def test_seed_is_deterministic():
    x, y = make_data()
    family = ExtratreesFamily()
    a = family.fit(x, y=y, params=PARAMS, seed=3).predict(x)
    b = family.fit(x, y=y, params=PARAMS, seed=3).predict(x)
    assert np.array_equal(a, b)


def test_absolute_error_objective():
    x, y = make_data(30)
    fitted = ExtratreesFamily().fit(x, y=y, objective="absolute_error", params=PARAMS)
    assert np.isfinite(fitted.predict(x)).all()
    with pytest.raises(ModelError):
        ExtratreesFamily().fit(x, y=y, objective="poisson", params=PARAMS)


def test_unavailable_hint(monkeypatch):
    def broken(name):
        raise ImportError("no sklearn")

    monkeypatch.setattr(base, "import_module", broken)
    ok, hint = ExtratreesFamily().available()
    assert not ok
    assert "uv sync --extra trees" in hint


def test_space_and_objectives():
    family = ExtratreesFamily()
    assert [p.name for p in family.space()] == [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    ]
    assert family.objectives("classification") == ()


def test_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x, y = make_data()
    ExtratreesFamily().fit(x, y=y, params=PARAMS).predict(x)
    assert os.listdir(tmp_path) == []
