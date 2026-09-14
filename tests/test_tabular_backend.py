"""the xgboost tabular backend, driven through smolsmort.backends like any other named backend.

frames are one-row json feature files rather than images. the rule is trivially separable
(feature "x" above 0.5 is "called") so this is a mechanics check - it proves train/predict/save/load
round-trip through the real seam, not a claim about accuracy on a real forecast."""

from __future__ import annotations

import json

import pytest

from smolsmort.backends import get_backend

xgboost = pytest.importorskip("xgboost")


def _row(tmp_path, name, x, y):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"features": {"x": x, "y": y}}))
    return path


def _dataset(tmp_path, n=40):
    """a deterministic, linearly separable set of rows labelled by whether x crosses 0.5"""
    rows = []
    for i in range(n):
        x = (i % 10) / 10.0
        y = (i % 3) / 3.0
        label = "called" if x >= 0.5 else "not_called"
        rows.append({"path": _row(tmp_path, f"row{i}", x, y), "label": label})
    return rows


def test_train_and_predict_round_trip_through_the_named_backend(tmp_path):
    examples = _dataset(tmp_path)
    classes = {"not_called": 0, "called": 1}
    backend = get_backend("xgboost", rounds=20, seed=0)

    progress = []
    weights = backend.train(
        examples, classes=classes, on_progress=lambda e, t: progress.append((e, t))
    )
    assert progress == [(20, 20)]

    held_out = [
        _row(tmp_path, "held_low", 0.1, 0.5),
        _row(tmp_path, "held_high", 0.9, 0.5),
    ]
    predicted = backend.predict(weights, held_out, classes=classes)
    assert [p["matched_template"] for p in predicted] == ["not_called", "called"]
    assert all(0.0 <= p["score"] <= 1.0 for p in predicted)


def test_a_saved_backend_predicts_the_same_after_loading(tmp_path):
    examples = _dataset(tmp_path)
    classes = {"not_called": 0, "called": 1}
    backend = get_backend("xgboost", rounds=20, seed=0)
    weights = backend.train(examples, classes=classes)

    checkpoint = tmp_path / "weights" / "forecast.xgb"
    backend.save(weights, checkpoint)
    reloaded = backend.load(checkpoint)

    frame = [_row(tmp_path, "check", 0.9, 0.2)]
    before = backend.predict(weights, frame, classes=classes)
    after = backend.predict(reloaded, frame, classes=classes)
    assert before[0]["matched_template"] == after[0]["matched_template"]
    assert before[0]["score"] == pytest.approx(after[0]["score"], abs=1e-4)


def test_training_with_no_labelled_example_is_refused(tmp_path):
    backend = get_backend("xgboost")
    unlabelled = [{"path": _row(tmp_path, "row0", 0.1, 0.2), "label": None}]
    with pytest.raises(Exception, match="no example carries a label"):
        backend.train(unlabelled, classes={})


def test_xgboost_is_registered_by_name():
    from smolsmort.backends import names

    assert "xgboost" in names()
