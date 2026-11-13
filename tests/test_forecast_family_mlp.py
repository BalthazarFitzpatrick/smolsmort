"""mlp family: shapes, missing values, unseen levels, determinism, history, safety"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smolsmort.forecast.families import base  # noqa: E402
from smolsmort.forecast.families.mlp import MlpFamily  # noqa: E402
from smolsmort.forecast.model import ModelError  # noqa: E402

PARAMS = {"n_layers": 1.0, "width": 8.4, "batch_size": 32.0}
FAST = {"max_rounds": 6, "patience": 3}


def make_data(rows=120, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 4))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=rows)
    return x, y


def fit(x, y, **kwargs):
    options = {"params": PARAMS, **FAST, **kwargs}
    return MlpFamily().fit(x, objective=options.pop("objective", "squared"), y=y, **options)


def test_fit_and_predict_shape():
    x, y = make_data()
    fitted = fit(x, y)
    prediction = fitted.predict(x[:10])
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()


def test_nan_features():
    x, y = make_data()
    x[::7, 2] = np.nan
    fitted = fit(x, y)
    assert np.isfinite(fitted.predict(x)).all()


def test_unseen_categorical_level():
    x, y = make_data()
    x[:, 3] = np.arange(len(x)) % 3
    fitted = fit(x, y, feature_types=["q", "q", "q", "c"])
    probe = x[:5].copy()
    probe[:, 3] = 99
    assert np.isfinite(fitted.predict(probe)).all()


def test_fixed_seed_is_identical():
    x, y = make_data()
    first = fit(x, y, seed=3).predict(x)
    second = fit(x, y, seed=3).predict(x)
    assert np.array_equal(first, second)


def test_absolute_objective():
    x, y = make_data()
    fitted = fit(x, y, objective="absolute")
    assert fitted.eval_history["metric"] == "absolute_error"
    assert np.isfinite(fitted.predict(x)).all()


def test_curves_match_epochs_run():
    x, y = make_data()
    history = fit(x, y, patience=100, max_rounds=5).eval_history
    epochs = len(history["validation"])
    assert epochs == len(history["training"])
    assert epochs <= 5
    assert len(history["curves"]["wape"]) == epochs


def test_unusable_target_raises():
    x, y = make_data(rows=15)
    with pytest.raises(ModelError):
        fit(x, y)


def test_thread_count_restored():
    x, y = make_data()
    before = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    fit(x, y, nthread=2)
    assert torch.get_num_threads() == before
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_unavailable_library_hint(monkeypatch):
    def refuse(name):
        raise ImportError("no torch here")

    monkeypatch.setattr(base, "import_module", refuse)
    ok, hint = MlpFamily().available()
    assert not ok
    assert "uv sync --extra neural" in hint


def test_no_files_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x, y = make_data()
    fit(x, y).predict(x)
    assert list(tmp_path.iterdir()) == []


def test_space_names_and_cost():
    family = MlpFamily()
    names = [p.name for p in family.space()]
    assert names == [
        "n_layers",
        "width",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
    ]
    assert family.cost((100, 5), family.defaults()) > 0
    assert family.objectives("regression") == ("squared", "absolute")
