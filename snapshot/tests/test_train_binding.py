"""picking a training set has to actually reach the trainer.

this exists because it did not. the chosen set's name lived in the page and was never sent, so
the trainer went on building examples from whatever candidates queue happened to be bound -
unreviewed boxes, which is the exact thing the picker was added to prevent. the bug was invisible
from the ui: the name appeared next to the button, so the pick looked like it had taken.

synthetic frames on purpose, so this runs on a fresh clone. the assertions are about WIRING - that
the bound set is what gets loaded, that its labels become channels, and that the overlay can be
asked for one of them - not about whether the model learns anything from three noise images.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image

import snapshot.tools.review_templates as rt
from snapshot.review import paths

LABELS = ("hostile_normal", "neutral_dim")


@pytest.fixture(autouse=True)
def _train_on_the_cpu(monkeypatch):
    """keep these off the GPU: a binding test has no use for it and the driver aborts the process.

    smolsmort's train() picks mps whenever it is available. On 2026-09-06 that took the whole suite
    down with "Fatal Python error: Aborted" inside AGXMetalG16G_B0, Apple's Metal shader compiler,
    reached through torch's MPS backend - a hard abort rather than a failure, so pytest reported
    nothing and every pre-push hook blocked. Intermittent, so bisecting it finds a false answer.

    What these tests actually assert is that a channel map survives binding, which is arithmetic.
    Nothing here needs a GPU, so the crash surface is removed rather than tolerated.
    """
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture
def bound(tmp_path, monkeypatch):
    """a promoted training set of two classes over synthetic frames, and a trainer bound to it"""
    sessions = tmp_path / "sessions"
    frames = sessions / "rec" / "frames"
    frames.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(3):
        pixels = rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(frames / f"{index:07d}.jpg")

    (tmp_path / "labels").mkdir()
    (tmp_path / "sets").mkdir()
    rows = []
    for index in range(3):
        for slot, label in enumerate(LABELS):
            rows.append(
                {
                    "recording": "rec",
                    "frame": f"{index:07d}.jpg",
                    "left": 40 + slot * 200,
                    "top": 60,
                    "width": 120,
                    "height": 20,
                    "label": label,
                }
            )
        # one hard negative per frame, so the negatives path is exercised too
        rows.append(
            {
                "recording": "rec",
                "frame": f"{index:07d}.jpg",
                "left": 10,
                "top": 200,
                "width": 120,
                "height": 20,
                "negative": True,
            }
        )
    (tmp_path / "sets" / "set_a.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(paths, "SESSIONS_DIR", sessions)
    # the promoted sets are named in paths now, not derived from the labels base - a test that
    # leaves this pointing at the real directory both reads and REWRITES the real sets
    monkeypatch.setattr(paths, "DATASETS_DIR", tmp_path / "sets")
    state = rt.ReviewState(
        frames, [], tmp_path / "lib", tmp_path / "decisions.json", "rec", pool=tmp_path / "tiles"
    )
    state.bases = dict(state.bases)
    state.bases["labels"] = str(tmp_path / "labels")
    trainer = rt.TrainState(state)
    trainer.MODEL_PATH = tmp_path / "model.pt"
    return trainer


def test_bind_reports_the_sets_own_classes(bound):
    # weights come back None: binding a set drops any loaded checkpoint, because a checkpoint
    # carries its own channel map and keeping one alive would name this set's channels wrongly
    assert bound.bind("set_a") == {"set": "set_a", "weights": None, "classes": list(LABELS)}


def test_bound_set_is_what_gets_loaded(bound):
    """the regression itself: examples must come from the SET, not from state.candidates.

    state.candidates is empty here, so a trainer still reading it would return nothing at all.
    """
    bound.bind("set_a")
    examples = bound.dataset()
    assert [e.path.name for e in examples] == ["0000000.jpg", "0000001.jpg", "0000002.jpg"]
    assert sum(len(e.centres) for e in examples) == 6
    assert sum(len(e.negatives) for e in examples) == 3
    assert bound.classes == {"hostile_normal": 0, "neutral_dim": 1}


def test_unbinding_falls_back_to_the_bound_queue(bound):
    """clearing the pick must not leave the old set's classes behind, or the channel picker
    goes on offering channels the next model does not have"""
    bound.bind("set_a")
    bound.dataset()
    bound.bind("")
    assert bound.training_set is None
    assert bound.classes == {}
    assert bound.class_names() == {"set": None, "weights": None, "classes": []}


def test_channel_map_survives_into_training(bound):
    """one output channel per label, which is what makes the picker mean anything"""
    bound.bind("set_a")
    bound.start(epochs=2, batch=2)
    _wait(bound)
    assert bound.job["error"] is None
    from smolsmort.detect.train import heatmaps_for

    stack = heatmaps_for(bound.model, bound.examples[0].path)
    assert stack.shape[0] == len(LABELS)


def test_overlay_can_be_asked_for_one_channel(bound):
    """a channel out of range falls back to the loudest rather than raising - the page can ask
    for a channel a freshly rebound set no longer has"""
    bound.bind("set_a")
    bound.start(epochs=2, batch=2)
    _wait(bound)
    for channel in (-1, 0, 1, 99):
        assert bound.overlay_bytes(0, channel=channel)


def test_proposal_size_comes_from_the_training_set(bound):
    """a sweep's boxes must be the size the set was drawn at, not the review state's default.

    THE REGRESSION THIS CATCHES: state.uniform_* defaults to a size cut for the old template
    library, and a sweep that took it emitted proposals a quarter the size of a real plate. A
    heatmap peak says where a plate is and never how big, so the size can only come from the data
    the model was trained on.
    """
    bound.state.uniform_width, bound.state.uniform_height = 64, 10
    bound.bind("set_a")
    assert bound._trained_box_size() == (120, 20)


def test_box_size_falls_back_when_no_set_is_bound(bound):
    """with nothing bound there is no better answer than the review default - but it must not
    raise, because the sweep asks for a size before it checks anything else"""
    bound.state.uniform_width, bound.state.uniform_height = 64, 10
    bound.bind("")
    assert bound._trained_box_size() == (64, 10)


def test_sweep_cuts_kernels_into_the_pool(bound):
    """proposals must become KERNELS, not just a candidates file - but only on SEND.

    THE GAP THIS CATCHES: discard / promote renders its tiles from the npz pool, never from the
    candidates file. A sweep that wrote only the file left the proposals invisible - nothing to
    look at and no way to judge them, which is the entire purpose of the stage. They also carry
    their own tag, so a batch of proposals can be closed separately from the drawn tiles beside
    them in the pool.

    THE SWEEP ITSELF MUST CUT NOTHING. it used to cut every proposal down to SWEEP_KEEP_FLOOR the
    instant it finished, so the threshold slider was choosing from a pool that had already taken
    everything - "It also doesn't take the cut of the sweep... but takes everything."
    """
    bound.bind("set_a")
    bound.start(epochs=2, batch=2)
    _wait(bound)
    bound.start_sweep("rec")
    _wait_sweep(bound)
    assert bound.sweep_job["error"] is None
    assert not list(bound.state.unsorted_dir.glob("*.npz")), (
        "the sweep put tiles in the pool before a threshold was ever chosen"
    )
    assert bound.send_sweep(0.0)["sent"] > 0
    cut = sorted(bound.state.unsorted_dir.glob("*.npz"))
    assert cut, "a send that cuts no tiles is invisible in discard / promote"
    assert bound.sweep_job["tiles"] == len(cut)
    assert all("_cnn-" in p.stem for p in cut), "proposals need their own closable tag"


def test_send_only_cuts_proposals_above_the_threshold(bound):
    """the slider is the cut, and it must be able to reject most of a sweep.

    the floor the sweep keeps to is 0.05 (paths.SWEEP_KEEP_FLOOR) and deliberately far below any
    threshold worth sending, so that a threshold can be tried without paying for another run.
    """
    bound.bind("set_a")
    bound.start(epochs=2, batch=2)
    _wait(bound)
    bound.start_sweep("rec")
    _wait_sweep(bound)
    proposals = bound.sweep_job["proposals"]
    assert proposals, "nothing to threshold"

    cut_at = sorted(c["score"] for c in proposals)[-1]
    sent = bound.send_sweep(cut_at)
    assert sent["sent"] == sum(1 for c in proposals if c["score"] >= cut_at)
    assert sent["of"] == len(proposals)
    assert len(list(bound.state.unsorted_dir.glob("*.npz"))) == sent["tiles"]


def test_a_sent_kernel_is_named_by_its_row_in_the_candidates_file(bound):
    """A KERNEL'S INDEX IS ITS ADDRESS. promote resolves "<tag>_k<n>" by reading row n of the
    candidates file, which was written with EVERY proposal - so naming the survivors of a
    threshold 0..N would point each label at a different box than the one on screen.
    """
    bound.bind("set_a")
    bound.start(epochs=2, batch=2)
    _wait(bound)
    bound.start_sweep("rec")
    _wait_sweep(bound)
    proposals = bound.sweep_job["proposals"]
    scores = sorted(c["score"] for c in proposals)
    cut_at = scores[len(scores) // 2]
    kept = [i for i, c in enumerate(proposals) if c["score"] >= cut_at]
    bound.send_sweep(cut_at)

    from snapshot.review.state import _tile_index

    cut = {_tile_index(p.stem) for p in bound.state.unsorted_dir.glob("*.npz")}
    assert cut == set(kept)


def test_sweep_refuses_without_a_trained_model(bound):
    """it must say what is missing rather than sweep with an untrained or absent model"""
    bound.bind("set_a")
    assert bound.start_sweep("rec") == {"ok": True}
    _wait_sweep(bound)
    assert "no trained model" in (bound.sweep_job["error"] or "")


def _wait_sweep(trainer, limit: int = 600) -> None:
    import time

    for _ in range(limit):
        if not trainer.sweep_job["running"]:
            return
        time.sleep(0.5)
    raise AssertionError("sweep did not finish")


def _wait(trainer, limit: int = 600) -> None:
    import time

    for _ in range(limit):
        if not trainer.job["running"]:
            return
        time.sleep(0.5)
    raise AssertionError("training did not finish")


def test_a_saved_model_reloads_at_its_own_channel_count(tmp_path, bound):
    """load() must size the head FROM THE CHECKPOINT, not from a default of one class.

    THE BUG THIS PINS: load(path) defaulted to classes=1 and raised a torch size mismatch on any
    real model - "copying a param with shape [15, 48, 1, 1], the shape in current model is
    [1, 48, 1, 1]". Training holds its model in memory, so a run looked fine and the NEXT restart
    500'd every heatmap overlay. The channel picker then appeared broken when the actual failure
    was that no multi-class model could be reloaded at all.
    """
    from smolsmort.detect.train import load

    bound.bind("set_a")
    bound.start(epochs=2, batch=2)
    _wait(bound)
    assert bound.MODEL_PATH.is_file()

    reloaded = load(bound.MODEL_PATH)
    heads = [m for m in reloaded.modules() if hasattr(m, "out_channels")]
    assert heads[-1].out_channels == len(LABELS)


def test_a_set_mixing_capture_resolutions_is_refused(tmp_path, monkeypatch):
    """a plate is a fixed pixel size on ONE screen, so a set spanning two has no right box size.

    every row looks individually fine, so nothing downstream can catch this: the fitted size would
    be right for neither half, the sweep would emit proposals at a size fitting neither, and the
    loss would be asked to find one object that is two different sizes.
    """
    import json as _json

    import pytest as _pytest
    from PIL import Image as _Image
    from smolsmort.detect.dataset import DatasetError, build_training_set

    sessions = tmp_path / "sessions"
    rows = []
    for recording, size in (("retina", (3420, 2224)), ("other", (2560, 1440))):
        frames = sessions / recording / "frames"
        frames.mkdir(parents=True)
        _Image.new("RGB", size).save(frames / "0000001.jpg")
        rows.append(
            {
                "recording": recording,
                "frame": "0000001.jpg",
                "left": 10,
                "top": 10,
                "width": 100,
                "height": 20,
                "label": "hostile",
            }
        )
    path = tmp_path / "mixed.jsonl"
    path.write_text("".join(_json.dumps(r) + "\n" for r in rows))

    with _pytest.raises(DatasetError, match="mixes capture resolutions"):
        build_training_set(path, sessions)


def test_a_sweep_batch_resolves_back_to_its_recording(tmp_path, monkeypatch):
    """THE RETURN EDGE, and it had never worked. Sweep tiles are named
    "<recording>_cnn-<stamp>_k<n>", so their batch tag ends in an UNDERSCORE-separated suffix -
    but _session_frames_for only stripped DOT-separated ones and returned None for every sweep
    batch. promote_to_training then counted each as unresolved and dropped it: 24 labelled
    proposals discarded in one go while promote reported success and wrote the drawn rows only.
    """
    from snapshot.review import paths as review_paths
    from snapshot.review.state import _session_frames_for

    monkeypatch.setattr(review_paths, "SESSIONS_DIR", tmp_path)
    frames = tmp_path / "archive" / "walk" / "frames"
    frames.mkdir(parents=True)

    assert _session_frames_for("archive__walk") == frames
    assert _session_frames_for("archive__walk_cnn-20260903-015830") == frames
    assert _session_frames_for("archive__walk_drawn-20260903-0114") == frames
    assert _session_frames_for("archive__walk.band") == frames
    assert _session_frames_for("nothing__like__this") is None


def test_a_repoint_survives_a_restart(tmp_path, monkeypatch):
    """it used to live only in the running process, and the restart that reverted it cost a
    training set: the next promote resolved nothing and rewrote plates.jsonl with zero rows
    """
    from snapshot.review.state import ReviewState

    monkeypatch.setattr(ReviewState, "BASES_FILE", tmp_path / "review_bases.json")
    first = ReviewState(tmp_path, [], tmp_path, tmp_path / "d.json", "")
    first.set_bases({"sessions": str(tmp_path / "elsewhere")})

    second = ReviewState(tmp_path, [], tmp_path, tmp_path / "d.json", "")
    second.load_saved_bases()
    assert second.bases["sessions"] == str(tmp_path / "elsewhere")


def test_a_promote_that_resolved_nothing_leaves_the_old_set_alone(bound):
    """REWRITING WHOLE IS RIGHT, ERASING IS NOT. every tile failing to resolve means the frames
    could not be found - not that the labels were withdrawn - so the previous answer must stand.

    This is what actually happened: a restart reset the sessions base, the next promote could not
    find a single recording, and it wrote an empty training set over a good one.
    """
    bound.bind("set_a")
    bound.start(epochs=2, batch=2)
    _wait(bound)
    bound.start_sweep("rec")
    _wait_sweep(bound)
    bound.send_sweep(0.0)

    state = bound.state
    tiles = sorted(p.stem for p in state.unsorted_dir.glob("*.npz"))
    assert tiles, "the send should have cut tiles into the pool"
    for name in tiles:
        state.set_pool_label(name, "Friendly NPC")
    first = state.promote_to_training()
    out = pathlib.Path(first["out"])
    assert first["rows"] > 0 and out.is_file()
    kept = out.read_text()

    # the frames can no longer be found, exactly as after a base-reverting restart
    state.set_bases({"sessions": str(state.unsorted_dir / "nowhere")})
    second = state.promote_to_training()
    assert "error" in second, second
    assert second["rows"] == 0
    assert out.read_text() == kept, "the previous training set was overwritten"
