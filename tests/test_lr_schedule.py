"""learning rates at optimiser steps, with deterministic cpu training on tiny frames"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from smolsmort.detect import train as detector
from smolsmort.detect.backend import HeatmapBackend
from smolsmort.detect.dataset import Example
from smolsmort.detect.model import build_model

torch = pytest.importorskip("torch")


@pytest.fixture
def training_examples(tmp_path):
    examples = []
    rng = np.random.default_rng(7)
    for index in range(4):
        path = tmp_path / f"frame_{index}.png"
        pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        pixels[24:32, 24:40] = 255
        Image.fromarray(pixels).save(path)
        examples.append(
            Example(path=path, centres=[(32.0, 28.0)], sizes=[(16, 8)], exhaustive=True)
        )
    return examples


@pytest.fixture
def trace_training(training_examples, monkeypatch):
    build_optimizer = detector.build_optimizer
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def run_training(**options):
        rates = []

        def record_optimizer(*args, **kwargs):
            optimizer = build_optimizer(*args, **kwargs)
            optimizer.register_step_pre_hook(
                lambda optimizer, args, kwargs: rates.append(
                    [group["lr"] for group in optimizer.param_groups]
                )
            )
            return optimizer

        settings = {
            "epochs": 3,
            "batch": 2,
            "learning_rate": 3e-4,
            "seed": 11,
            "device": "cpu",
            "crop": 16,
            "channels": 4,
            "feed_workers": 1,
        }
        with monkeypatch.context() as patch:
            patch.setattr(detector, "build_optimizer", record_optimizer)
            model, history = detector.train(training_examples, **(settings | options))
        return model, history, rates

    try:
        yield run_training
    finally:
        torch.set_num_threads(threads)


def hash_weights(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_constant_preserves_unscheduled_steps_and_weights(trace_training, optimizer):
    original, original_loss, original_rates = trace_training(optimizer=optimizer)
    constant, constant_loss, constant_rates = trace_training(
        optimizer=optimizer, lr_schedule="constant", lr_floor=2e-4
    )
    assert original_rates == constant_rates == [[3e-4]] * 6
    assert original_loss == constant_loss
    assert hash_weights(original) == hash_weights(constant)
    if optimizer == "adamw":
        # captured on the unedited pin 1ae1286 before adding schedule handling
        assert constant_loss == pytest.approx(
            [4.640725135803223, 3.661468505859375, 6.806231498718262], rel=1e-5
        )


@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
@pytest.mark.parametrize("floor", [0.0, 1e-5, 3e-4])
def test_cosine_anneals_each_step_and_finishes_at_the_floor(trace_training, optimizer, floor):
    _, history, rates = trace_training(optimizer=optimizer, lr_schedule="cosine", lr_floor=floor)
    values = [rate[0] for rate in rates]
    assert len(values) == 6 and len(history) == 3
    assert values[0] == 3e-4
    assert values[-1] == floor
    assert all(left >= right >= floor for left, right in zip(values, values[1:], strict=False))
    assert values[1] == pytest.approx(floor + (3e-4 - floor) * 0.9045084971874737)


@pytest.mark.parametrize("epochs,batch", [(1, 2), (3, 8), (1, 8)])
def test_cosine_covers_the_planned_steps_including_a_single_step(trace_training, epochs, batch):
    _, _, rates = trace_training(epochs=epochs, batch=batch, lr_schedule="cosine")
    assert len(rates) == epochs * max(1, 4 // batch)
    assert rates[-1] == [1e-5]
    if len(rates) > 1:
        assert rates[0] == [3e-4]


def test_cosine_early_stop_keeps_the_planned_schedule(trace_training):
    _, full_loss, full_rates = trace_training(lr_schedule="cosine", cache_bytes=0)
    _, stopped_loss, stopped_rates = trace_training(
        lr_schedule="cosine", cache_bytes=0, on_progress=lambda progress: progress.epoch == 2
    )
    assert stopped_loss == full_loss[:2]
    assert stopped_rates == full_rates[:4]
    assert stopped_rates[-1][0] > 1e-5


@pytest.mark.parametrize("schedule", ["linear", "Cosine", "", None])
def test_an_unknown_schedule_is_refused_before_training(schedule):
    with pytest.raises(ValueError, match="allowed values: constant, cosine"):
        detector.train([], lr_schedule=schedule)


@pytest.mark.parametrize("floor", [-1e-5, 1e-3, float("nan"), float("inf")])
def test_a_cosine_floor_outside_the_learning_rate_range_is_refused(floor):
    with pytest.raises(ValueError, match="lr_floor must be between 0 and learning_rate"):
        detector.train([], lr_schedule="cosine", lr_floor=floor)


@pytest.mark.parametrize("options", [{}, {"lr_schedule": "cosine", "lr_floor": 2e-5}])
def test_the_backend_forwards_the_schedule_and_floor(training_examples, monkeypatch, options):
    calls = []
    model = build_model(channels=4, capture_width=64)

    def record_train(examples, **kwargs):
        calls.append(kwargs)
        return model, []

    monkeypatch.setattr(detector, "train", record_train)
    weights = HeatmapBackend(**options).train(training_examples, classes={"object": 0})
    assert weights.model is model
    assert calls[0]["lr_schedule"] == options.get("lr_schedule", "constant")
    assert calls[0]["lr_floor"] == options.get("lr_floor", 1e-5)
