"""evaluate() on the two real backends: finite, deterministic, and lower once trained"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from smolsmort import backends
from smolsmort.review.train import TrainState

pytest.importorskip("torch")

CLASSES = {"friendly": 0, "hostile": 1}


def make_examples(tmp_path, count=6, size=(320, 256)):
    rng = np.random.default_rng(3)
    examples = []
    for i in range(count):
        pixels = rng.integers(20, 60, size=(size[1], size[0], 3)).astype(np.uint8)
        cx, cy = 100 + i * 15, 100 + (i % 3) * 20
        pixels[cy - 12 : cy + 12, cx - 24 : cx + 24] = 230
        path = tmp_path / f"frame_{i}.png"
        Image.fromarray(pixels).save(path)
        examples.append(
            {
                "path": str(path),
                "centres": [(cx, cy)],
                "labels": ["hostile" if i % 2 else "friendly"],
                "width": 48,
                "height": 24,
                "exhaustive": True,
            }
        )
    return examples


@pytest.fixture(params=["heatmap", "box"])
def fitted(request, tmp_path):
    examples = make_examples(tmp_path)
    backend = backends.get_backend(request.param, epochs=15, device="cpu")
    weights = backend.train(examples, classes=CLASSES)
    return request.param, backend, weights, examples


def test_evaluate_is_finite_deterministic_and_lower_when_trained(fitted):
    name, backend, weights, examples = fitted
    first = backend.evaluate(weights, examples, classes=CLASSES)
    assert isinstance(first, float)
    assert math.isfinite(first)
    assert backend.evaluate(weights, examples, classes=CLASSES) == first

    untrained = type(weights)(**{**weights.__dict__, "model": fresh_model(name, weights)})
    assert first < backend.evaluate(untrained, examples, classes=CLASSES)


def fresh_model(name, weights):
    if name == "heatmap":
        from smolsmort.detect.model import build_model

        return build_model(channels=24, classes=2)
    from smolsmort.boxes.model import build_model

    return build_model(classes=2)


def test_evaluate_handles_an_empty_list(fitted):
    _, backend, weights, _ = fitted
    assert backend.evaluate(weights, [], classes=CLASSES) == 0.0


def test_status_reports_val_and_test_loss(fitted, tmp_path):
    name, _, _, examples = fitted
    trainer = TrainState(name, epochs=2, device="cpu")
    trainer.bind(
        examples[:4],
        CLASSES,
        training_set="synthetic",
        val_examples=examples[4:5],
        test_examples=examples[5:],
    )
    trainer.start()
    trainer._worker.join()
    status = trainer.status()
    assert status["state"] == "finished", status
    assert math.isfinite(status["val_loss"])
    assert math.isfinite(status["test_loss"])
