"""smolsmort/review/train.py: the reference driver of the model-backend seam - bind or load, train,
sweep, inspect a channel, save under a name.

the FAKE half proves the seam contract end to end with no torch at all, same as test_loop.py; the
heatmap half proves the same driver against the real reference backend (smolsmort/detect), gated by
importorskip so this file collects and mostly runs even where torch is not installed.
"""

from __future__ import annotations

import json

import pytest

from smolsmort import backends
from smolsmort.review.train import DEFAULT_LEARNING_RATE, DEFAULT_SEED, TrainState, TrainStateError

# ---------------------------------------------------------------- fake backend, no torch at all

FAKE_EXAMPLES = [
    {"path": "a.jpg", "centres": [(10, 4)], "labels": ["hostile"], "width": 20, "height": 8},
    {"path": "b.jpg", "centres": [(30, 12)], "labels": ["friendly"], "width": 20, "height": 8},
]
FAKE_CLASSES = {"friendly": 0, "hostile": 1}


@pytest.fixture(autouse=True)
def _fake_backend():
    """the same FakeBackend the loop test drives, registered under a name for this file only"""
    backends.register("fake", "test_loop", "FakeBackend")
    try:
        yield
    finally:
        backends._REGISTRY.pop("fake", None)


def test_binding_reports_the_bound_classes_and_drops_any_loaded_weights():
    trainer = TrainState("fake")
    trainer.weights = object()
    trainer.loaded_weights = "old.pt"
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES, training_set="set_a")
    assert trainer.weights is None
    assert trainer.loaded_weights is None
    assert trainer.class_names() == {
        "training_set": "set_a",
        "weights": None,
        "classes": ["friendly", "hostile"],
    }


def test_find_train_predict_closes_the_loop_against_a_fake_backend():
    """the loop test's own shape (test_loop.py), driven through TrainState instead of by hand -
    train must actually reach the backend's train(), and sweep its predict()."""
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    seen = []
    result = trainer.train(on_progress=lambda epoch, epochs: seen.append((epoch, epochs)))
    assert result["classes"] == ["friendly", "hostile"]
    assert seen, "on_progress must be reached at least once"

    proposed = trainer.sweep(["c.jpg", "d.jpg"])
    assert len(proposed) == 2
    for candidate in proposed:
        assert set(candidate) >= {"path", "left", "top", "width", "height", "score"}
        assert candidate["matched_template"] in FAKE_CLASSES


def test_training_on_nothing_bound_refuses():
    trainer = TrainState("fake")
    with pytest.raises(TrainStateError, match="bind a training set first"):
        trainer.train()


def test_sweeping_before_training_or_loading_refuses():
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    with pytest.raises(TrainStateError, match="train or load first"):
        trainer.sweep(["c.jpg"])


def test_channel_inspection_refuses_a_backend_that_is_not_heatmap_shaped():
    """a fake (or a box) backend's weights are not a stack of per-class heatmaps - this must say so
    rather than hand back whatever `.model` happens to hold"""
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    trainer.train()
    with pytest.raises(TrainStateError, match="does not expose one heatmap per class"):
        trainer.channel_heatmap("c.jpg")


def test_a_named_save_round_trips_with_provenance(tmp_path):
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES, training_set="set_a")
    trainer.train()
    weights_path = tmp_path / "weights" / "run.pt"
    saved = trainer.save(weights_path)

    assert saved["training_set"] == "set_a"
    assert saved["classes"] == ["friendly", "hostile"]
    assert weights_path.is_file()

    sidecar = json.loads(weights_path.with_name(weights_path.name + ".provenance.json").read_text())
    assert sidecar["training_set"] == "set_a"
    # defaults recorded even though this backend never asked for them - see DEFAULT_LEARNING_RATE
    assert sidecar["options"]["learning_rate"] == DEFAULT_LEARNING_RATE
    assert sidecar["options"]["seed"] == DEFAULT_SEED
    assert "saved" in sidecar

    reloaded = TrainState("fake")
    info = reloaded.load_weights(weights_path)
    assert info == {
        "training_set": "set_a",
        "weights": "run.pt",
        "classes": ["friendly", "hostile"],
    }
    # loading pulled the weights back too, not just the class map - a sweep works without training
    assert reloaded.sweep(["e.jpg"])


def test_saving_before_training_or_loading_refuses(tmp_path):
    trainer = TrainState("fake")
    with pytest.raises(TrainStateError, match="train a model first"):
        trainer.save(tmp_path / "w.pt")


def test_loading_reads_provenance_missing_a_training_set(tmp_path):
    """a checkpoint saved before this existed (or by a bare `backend.save`, no provenance file at
    all) must still load - the class map comes from the backend's own sidecar either way."""
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    trainer.train()
    path = tmp_path / "bare.pt"
    trainer._backend.save(trainer.weights, path)  # no review-level provenance written

    reloaded = TrainState("fake")
    info = reloaded.load_weights(path)
    assert info["training_set"] is None
    assert info["classes"] == ["friendly", "hostile"]


# ---------------------------------------------------------------- the reference backend, gated

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _synthetic_frame(path, rng, size=(320, 240)) -> None:
    pixels = rng.integers(20, 60, size=(size[1], size[0], 3)).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def _synthetic_examples(tmp_path, count=4):
    rng = np.random.default_rng(3)
    labels = ["hostile", "friendly"]
    frames, examples = [], []
    for i in range(count):
        path = tmp_path / f"frame_{i}.jpg"
        _synthetic_frame(path, rng)
        frames.append(path)
        examples.append(
            {
                "path": str(path),
                "centres": [(60 + i * 5, 40)],
                "labels": [labels[i % 2]],
                "width": 24,
                "height": 12,
            }
        )
    return frames, examples


def test_bind_train_inspect_sweep_save_load_against_the_heatmap_backend(tmp_path):
    """the reference backend, end to end, entirely through TrainState - never through
    smolsmort.detect.train directly except for channel_heatmap, which is documented as reaching
    past the seam on purpose."""
    frames, examples = _synthetic_examples(tmp_path)
    classes = {"friendly": 0, "hostile": 1}

    trainer = TrainState("heatmap", epochs=2, device="cpu")
    trainer.bind(examples, classes, training_set="synthetic")
    seen = []
    trainer.train(on_progress=lambda epoch, epochs: seen.append((epoch, epochs)))
    assert seen == [(1, 2), (2, 2)]

    heat = trainer.channel_heatmap(frames[0], channel=0)
    assert heat.ndim == 2
    loudest = trainer.channel_heatmap(frames[0], channel=99)  # out of range -> the loudest of all
    assert loudest.shape == heat.shape

    proposed = trainer.sweep([str(f) for f in frames])
    for candidate in proposed:
        assert candidate["matched_template"] in classes

    weights_path = tmp_path / "checkpoints" / "run-a.pt"
    saved = trainer.save(weights_path)
    assert saved["classes"] == ["friendly", "hostile"]
    assert saved["training_set"] == "synthetic"

    reloaded = TrainState("heatmap", device="cpu")
    info = reloaded.load_weights(weights_path)
    assert info == {
        "training_set": "synthetic",
        "weights": "run-a.pt",
        "classes": ["friendly", "hostile"],
    }

    # a checkpoint loaded back sweeps without ever calling train() again
    assert reloaded.sweep([str(frames[0])])


def test_channel_inspection_refuses_a_box_backend():
    """the box backend's model answers heat/size/offset/centre heads, not a stack of per-class
    heatmaps - this must refuse by name rather than hand back the wrong tensor shape"""
    trainer = TrainState("box", epochs=1, device="cpu")
    trainer.weights = object()  # inspection checks the backend name before touching weights
    with pytest.raises(TrainStateError, match="does not expose one heatmap per class"):
        trainer.channel_heatmap("f.jpg")
