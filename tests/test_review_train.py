"""smolsmort/review/train.py: the reference driver of the model-backend seam - bind or load, train,
sweep, inspect a channel, save under a name.

the FAKE half proves the seam contract end to end with no torch at all, same as test_loop.py; the
heatmap half proves the same driver against the real reference backend (smolsmort/detect), gated by
importorskip so this file collects and mostly runs even where torch is not installed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from smolsmort import backends
from smolsmort.backends import sidecar
from smolsmort.review.train import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    TrainState,
    TrainStateError,
    saved_checkpoints,
)

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


# ---------------------------------------------------------------- background training + abort
#
# ported from the parent project's snapshot/tests/test_train_abort.py: a training run can be
# stopped from the page, at the next epoch, without saving anything. the fixture there paused a
# real detect.train() epoch loop on a threading.Event; this pauses a fake backend's train() the
# same way, since `start`/`abort` now sit above the seam rather than above smolsmort.detect directly.


class PausableBackend:
    """a fake backend whose train() blocks per epoch on a release event, so an abort can be
    exercised deterministically with no timing luck beyond the event itself."""

    def __init__(self, epochs=50, release=None, seen=None):
        self.epochs = epochs
        self.release = release if release is not None else threading.Event()
        self.seen = seen if seen is not None else []

    def train(self, examples, *, classes, on_progress=None):
        for epoch in range(1, self.epochs + 1):
            self.release.wait(5)
            self.seen.append(epoch)
            if on_progress:
                on_progress(epoch, self.epochs)
        return {"classes": dict(classes)}

    def predict(self, weights, frames, *, classes):
        return []

    def save(self, weights, path):
        Path(path).write_text(json.dumps(weights))
        return Path(path)

    def load(self, path):
        data = json.loads(Path(path).read_text())
        return SimpleNamespace(**data)


@pytest.fixture
def _pausable():
    backends.register("pausable", "test_review_train", "PausableBackend")
    try:
        yield
    finally:
        backends._REGISTRY.pop("pausable", None)


def _abortable_trainer(_pausable, epochs=50):
    release = threading.Event()
    seen: list[int] = []
    trainer = TrainState("pausable", epochs=epochs, release=release, seen=seen)
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    return trainer, release, seen


def test_an_abort_stops_at_the_next_epoch_and_saves_nothing(_pausable):
    trainer, release, seen = _abortable_trainer(_pausable)
    assert trainer.start() == {"ok": True}
    assert trainer.abort() == {"ok": True}
    release.set()
    trainer._worker.join(5)

    job = trainer.status()
    assert job["running"] is False and job["finished"] is False
    assert job["aborted"] == 1
    assert seen == [1]
    assert trainer.weights is None


def test_an_abort_with_nothing_running_is_refused(_pausable):
    trainer, *_ = _abortable_trainer(_pausable)
    assert "error" in trainer.abort()


def test_starting_twice_is_refused_while_one_is_already_running(_pausable):
    trainer, release, _ = _abortable_trainer(_pausable)
    trainer.start()
    assert "error" in trainer.start()
    trainer.abort()
    release.set()
    trainer._worker.join(5)


def test_the_run_after_an_abort_is_not_aborted_too(_pausable):
    """the flag is cleared by start, or one abort would kill every later run at epoch one"""
    trainer, release, _ = _abortable_trainer(_pausable, epochs=3)
    trainer.start()
    trainer.abort()
    release.set()
    trainer._worker.join(5)

    trainer.start()
    trainer._worker.join(5)
    job = trainer.status()
    assert job["finished"] is True and job["aborted"] is None
    assert trainer.weights is not None


def test_a_training_error_surfaces_through_status_rather_than_hanging_the_poller(_pausable):
    trainer = TrainState("pausable", epochs=1)  # nothing bound - train() raises TrainStateError
    trainer.start()
    trainer._worker.join(5)
    job = trainer.status()
    assert job["running"] is False and job["finished"] is False
    assert "bind a training set first" in job["error"]


# ---------------------------------------------------------------- checkpoint-folder browsing
#
# ported from the parent project's `checkpoint_folders` / `saved_checkpoints`, which read the
# checkpoints root off `review.paths.CHECKPOINTS_DIR`. that module does not exist on this branch
# yet, so `root` is an explicit argument here - the routes card wires the real one in later.


def test_checkpoint_folders_lists_subfolders_and_suggests_a_name(tmp_path):
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES, training_set="set_a")
    (tmp_path / "run-1").mkdir()
    (tmp_path / "run-2").mkdir()

    listing = trainer.checkpoint_folders(tmp_path)
    assert listing["here"] == ""
    assert listing["parent"] is None
    assert listing["folders"] == ["run-1", "run-2"]
    assert listing["name"].startswith("set_a-")


def test_checkpoint_folders_descends_into_a_subfolder(tmp_path):
    trainer = TrainState("fake")
    (tmp_path / "run-1" / "nested").mkdir(parents=True)

    listing = trainer.checkpoint_folders(tmp_path, "run-1")
    assert listing["here"] == "run-1"
    assert listing["parent"] == ""
    assert listing["folders"] == ["nested"]


def test_checkpoint_folders_falls_back_to_root_rather_than_escaping_it(tmp_path):
    trainer = TrainState("fake")
    outside = trainer.checkpoint_folders(tmp_path, "../../etc")
    assert outside["here"] == ""


def test_resolve_checkpoint_refuses_a_name_that_would_leave_root(tmp_path):
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    trainer.train()
    saved = trainer.save(tmp_path / "run.pt")
    assert saved

    assert trainer.resolve_checkpoint(tmp_path, "run.pt") == tmp_path / "run.pt"
    assert trainer.resolve_checkpoint(tmp_path, "../elsewhere/run.pt") is None
    assert trainer.resolve_checkpoint(tmp_path, "missing.pt") is None


def test_saved_checkpoints_lists_named_saves_newest_first_with_their_classes(tmp_path):
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES, training_set="set_a")
    trainer.train()
    trainer.save(tmp_path / "a.pt")
    trainer.save(tmp_path / "sub" / "b.pt")

    listing = saved_checkpoints(tmp_path)
    names = [w["name"] for w in listing["weights"]]
    assert set(names) == {"a.pt", "sub/b.pt"}
    for weights in listing["weights"]:
        assert weights["classes"] == ["friendly", "hostile"]
        assert weights["training_set"] == "set_a"


def test_saved_checkpoints_falls_back_to_the_backend_sidecar_with_no_provenance(tmp_path):
    """a checkpoint saved by a bare backend.save() - no review-level provenance beside it at all -
    must still be listable, with the classes the backend's OWN sidecar knows"""
    path = tmp_path / "bare.pt"
    path.write_text("stub")
    sidecar(path).write_text(
        json.dumps({"backend": "fake", "classes": {"friendly": 0, "hostile": 1}})
    )

    listing = saved_checkpoints(tmp_path)
    assert listing["weights"] == [
        {
            "name": "bare.pt",
            "training_set": None,
            "classes": ["friendly", "hostile"],
            "kb": 0,
            "final": False,
        }
    ]


def test_saved_checkpoints_over_an_empty_or_missing_root(tmp_path):
    assert saved_checkpoints(tmp_path / "does-not-exist") == {"weights": []}
    assert saved_checkpoints(tmp_path) == {"weights": []}


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


def test_heatmap_backend_trains_with_a_custom_optimizer_and_size(tmp_path):
    """the hyperparams menu's options (optimizer, momentum, weight_decay, channels) reach the real
    trainer through backend_options - proven with sgd/nesterov and a smaller model, not just the
    adamw default the other end-to-end test above already covers"""
    from smolsmort.detect.model import build_model, count_parameters

    frames, examples = _synthetic_examples(tmp_path)
    classes = {"friendly": 0, "hostile": 1}

    trainer = TrainState(
        "heatmap",
        epochs=1,
        device="cpu",
        optimizer="sgd",
        momentum=0.9,
        weight_decay=1e-4,
        channels=16,
        learning_rate=1e-2,
    )
    trainer.bind(examples, classes, training_set="synthetic")
    trainer.train()

    assert count_parameters(trainer.weights.model) == count_parameters(
        build_model(channels=16, classes=2)
    )
    # one sgd epoch on synthetic noise proves nothing about detection quality - the point here is
    # that sweep() still runs end to end against a model built with the custom options
    trainer.sweep([str(frames[0])])


# ---------------------------------------------------------------- losses, named runs, final flag


class LossBackend:
    """a fake that reports a loss per epoch, hands its weights to the callback, and can evaluate"""

    def __init__(self, epochs=4):
        self.epochs = epochs

    def train(self, examples, *, classes, on_progress=None, window=None):
        for epoch in range(1, self.epochs + 1):
            if on_progress:
                on_progress(epoch, self.epochs, 1.0 / epoch, {"epoch": epoch})
        return {"classes": dict(classes), "window": window}

    def evaluate(self, weights, examples, *, classes):
        return 0.1 * len(examples)

    def predict(self, weights, frames, *, classes):
        return []

    def save(self, weights, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(weights))
        return Path(path)

    def load(self, path):
        data = json.loads(Path(path).read_text())
        return SimpleNamespace(**data)


@pytest.fixture
def _lossy():
    backends.register("lossy", "test_review_train", "LossBackend")
    try:
        yield
    finally:
        backends._REGISTRY.pop("lossy", None)


STATUS_KEYS = {
    "running",
    "state",
    "train_loss",
    "val_loss",
    "test_loss",
    "checkpoint_count",
    "epoch",
    "epochs",
    "error",
    "finished",
    "aborted",
}


def _run_lossy(tmp_path, **start):
    trainer = TrainState("lossy", epochs=4)
    trainer.bind(
        FAKE_EXAMPLES,
        FAKE_CLASSES,
        val_examples=FAKE_EXAMPLES[:1],
        test_examples=FAKE_EXAMPLES * 2,
    )
    assert trainer.start(root=tmp_path, **start) == {"ok": True}
    trainer._worker.join(5)
    return trainer


def test_status_has_its_keys_before_during_and_after_a_run(_lossy, tmp_path):
    idle = TrainState("lossy")
    assert set(idle.status()) >= STATUS_KEYS
    assert idle.status()["state"] == "idle" and idle.status()["test_loss"] is None

    trainer = _run_lossy(tmp_path, name="run")
    done = trainer.status()
    assert set(done) >= STATUS_KEYS
    assert done["state"] == "finished" and done["running"] is False
    assert done["train_loss"] == pytest.approx(0.25)
    assert done["val_loss"] == pytest.approx(0.1)
    assert done["test_loss"] == pytest.approx(0.4)
    assert done["checkpoint_count"] == 1


def test_losses_are_none_for_a_backend_without_evaluate(_pausable):
    trainer, release, _ = _abortable_trainer(_pausable, epochs=2)
    release.set()
    trainer.start()
    trainer._worker.join(5)
    done = trainer.status()
    assert done["val_loss"] is None and done["test_loss"] is None
    assert done["checkpoint_count"] == 0


def test_interim_checkpoints_count_and_the_final_one_lists_first(_lossy, tmp_path):
    trainer = _run_lossy(tmp_path, name="run", checkpoint_every=2)
    assert trainer.status()["checkpoint_count"] == 2  # epoch 2 interim, then the final
    entries = saved_checkpoints(tmp_path)["weights"]
    assert [e["name"] for e in entries][0] == "run.pt"
    assert [e["final"] for e in entries] == [True, False]
    assert {e["name"] for e in entries} == {"run.pt", "run-e2.pt"}


def test_a_manual_save_is_not_final(tmp_path):
    trainer = TrainState("fake")
    trainer.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    trainer.train()
    trainer.save(tmp_path / "a.pt")
    trainer.save(tmp_path / "b.pt", final=True)
    flags = {e["name"]: e["final"] for e in saved_checkpoints(tmp_path)["weights"]}
    assert flags == {"a.pt": False, "b.pt": True}


def test_a_named_run_round_trips_through_load_weights(_lossy, tmp_path):
    trainer = _run_lossy(tmp_path, name="my run!", window=320)
    assert trainer.status()["weights"] == "myrun.pt" and trainer.status()["window"] == 320
    provenance = json.loads((tmp_path / "myrun.pt.provenance.json").read_text())
    assert provenance["final"] is True and provenance["test_loss"] == pytest.approx(0.4)

    reloaded = TrainState("lossy")
    reloaded.load_weights(tmp_path / "myrun.pt")
    assert reloaded.weights.window == 320
    assert reloaded.classes == FAKE_CLASSES


def test_a_taken_or_escaping_name_is_refused_up_front(_lossy, tmp_path):
    _run_lossy(tmp_path, name="run")
    again = TrainState("lossy")
    again.bind(FAKE_EXAMPLES, FAKE_CLASSES)
    assert "already exists" in again.start(root=tmp_path, name="run")["error"]
    assert "outside" in again.start(root=tmp_path, name="x", folder="../..")["error"]
    assert "checkpoints folder" in again.start(name="x")["error"]


def test_an_aborted_named_run_writes_no_final_weights(_pausable, tmp_path):
    trainer, release, _ = _abortable_trainer(_pausable)
    trainer.start(root=tmp_path, name="cut")
    trainer.abort()
    release.set()
    trainer._worker.join(5)
    assert trainer.status()["state"] == "aborted"
    assert saved_checkpoints(tmp_path)["weights"] == []


def test_window_floor_needs_the_box_and_snaps_to_cells():
    pytest.importorskip("torch")
    from smolsmort.detect.model import STRIDE
    from smolsmort.review.train import window_floor

    floor = window_floor(226, 100)
    assert floor % STRIDE == 0 and floor >= 226 / 4 / 0.2
    assert window_floor(226, 100, downscale=2) > floor
