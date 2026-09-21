"""per-set backend and size mode, per-frame exhaustive mode, the not-an-object verdict and native
box sizes - at the state level, against a real tmp world and no torch."""

from __future__ import annotations

import json

import pytest
from review_world import drawn_boxes, make_world

from smolsmort.detect.dataset import build_training_set
from smolsmort.review import paths, setconfig
from smolsmort.review.state import ReviewState
from smolsmort.review.tiles import load_tile


@pytest.fixture
def world(tmp_path, monkeypatch):
    return make_world(tmp_path, monkeypatch, recordings=("rec_a",))


@pytest.fixture
def state(world):
    state = ReviewState(
        world.sessions, [], world.tiles / "_unbound.decisions.json", "", world.tiles
    )
    state.bind_recording("session", "rec_a")
    return state


def _rows(world, name: str) -> list[dict]:
    return [json.loads(x) for x in (world.sets / f"{name}.jsonl").read_text().splitlines()]


def _sweep_world(state, world) -> str:
    """a swept file: f00 holds a kept box (A) and an unjudged one (B), f01 the same (C, D)"""
    stamp = "20260101-000000"
    boxes = [
        {"path": "f00.png", "left": 20, "top": 20, "width": 30, "height": 10, "score": 0.9},
        {"path": "f00.png", "left": 120, "top": 80, "width": 30, "height": 10, "score": 0.8},
        {"path": "f01.png", "left": 20, "top": 20, "width": 30, "height": 10, "score": 0.9},
        {"path": "f01.png", "left": 120, "top": 80, "width": 30, "height": 10, "score": 0.8},
    ]
    (world.labels / f"rec_a.cnn-{stamp}.candidates.jsonl").write_text(
        "".join(json.dumps(b) + "\n" for b in boxes)
    )
    state.open_dataset(f"rec_a.cnn-{stamp}.candidates.jsonl")
    return f"rec_a_cnn-{stamp}"


# ---------------------------------------------------------------- backend per set


def test_a_set_without_a_backend_file_is_heatmap_and_uniform(world):
    assert setconfig.read_set_config("plain") is None
    assert setconfig.set_config("plain") == {"backend": "heatmap", "size_mode": "uniform"}


def test_writing_a_backend_persists_and_defaults_the_size_mode(world):
    assert setconfig.write_set_config("a", "box") == {"backend": "box", "size_mode": "native"}
    assert setconfig.write_set_config("b", "heatmap") == {
        "backend": "heatmap",
        "size_mode": "uniform",
    }
    assert setconfig.write_set_config("c", "box", "uniform")["size_mode"] == "uniform"
    assert setconfig.read_set_config("a") == {"backend": "box", "size_mode": "native"}
    assert (world.sets / "a._backend.json").is_file()


def test_an_unknown_backend_or_size_mode_is_refused_naming_the_known_ones(world):
    with pytest.raises(setconfig.SetConfigError, match="heatmap"):
        setconfig.write_set_config("a", "nope")
    with pytest.raises(setconfig.SetConfigError, match="size_mode"):
        setconfig.write_set_config("a", "box", "wide")
    assert not (world.sets / "a._backend.json").exists()


# ---------------------------------------------------------------- exhaustive frames


def test_an_exhaustive_frame_yields_negatives_and_an_unmarked_frame_does_not(state, world):
    tag = _sweep_world(state, world)
    state.buffer_label(f"{tag}_k00000", "alpha", "kinds")
    state.buffer_label(f"{tag}_k00002", "alpha", "kinds")
    state.save_labels()
    setconfig.write_frame_mode("rec_a", "f00.png", True)

    state.promote_to_training("ex")
    own = [r for r in _rows(world, "ex") if r["source"] == tag and r["negative"]]
    assert [(r["frame"], r["left"]) for r in own] == [("f00.png", 120)]

    examples, _ = build_training_set(world.sets / "ex.jsonl", world.sessions)
    by_frame = {e.path.name: e for e in examples}
    assert by_frame["f00.png"].exhaustive is True and by_frame["f01.png"].exhaustive is False
    assert (135.0, 85.0) in by_frame["f00.png"].negatives
    assert (135.0, 85.0) not in by_frame["f01.png"].negatives


def test_a_box_inside_a_kept_box_on_an_exhaustive_frame_is_not_a_negative(state, world):
    tag = _sweep_world(state, world)
    stamp = tag.split("_cnn-")[1]
    path = world.labels / f"rec_a.cnn-{stamp}.candidates.jsonl"
    boxes = [json.loads(x) for x in path.read_text().splitlines()]
    boxes.append({"path": "f00.png", "left": 22, "top": 21, "width": 30, "height": 10})
    path.write_text("".join(json.dumps(b) + "\n" for b in boxes))
    state.open_dataset(path.name)
    state.buffer_label(f"{tag}_k00000", "alpha", "kinds")
    state.save_labels()
    setconfig.write_frame_mode("rec_a", "f00.png", True)
    state.promote_to_training("near")
    lefts = [r["left"] for r in _rows(world, "near") if r["source"] == tag and r["negative"]]
    assert lefts == [120], "the misaligned copy at left=22 is dropped, the far box is background"


def test_frame_modes_default_to_explicit_and_turning_off_removes_the_entry(world):
    assert setconfig.read_frame_modes("rec_a") == {}
    setconfig.write_frame_mode("rec_a", "f00.png", True)
    assert setconfig.read_frame_modes("rec_a") == {"f00.png": {"exhaustive": True}}
    setconfig.write_frame_mode("rec_a", "f00.png", False)
    assert setconfig.read_frame_modes("rec_a") == {}
    assert (world.labels / "rec_a._frames.json").is_file()


# ---------------------------------------------------------------- not an object


def _pool(state) -> list[str]:
    state.save_drawn(drawn_boxes())
    return state.unsorted_names()


def test_not_object_reaches_disk_only_through_save_labels(state, world):
    names = _pool(state)
    assert state.buffer_not_object(names[0]) == 1
    assert not (world.tiles / "_labels.json").exists()
    grid = state.clusters()
    assert grid["counts"]["classes_assigned"] == 1
    item = next(i for c in grid["clusters"] for i in c["items"] if i["name"] == names[0])
    assert item["not_object"] is True and item["pending"] is True
    assert state.save_labels() == {"saved": 1}
    record = json.loads((world.tiles / "_labels.json").read_text())[names[0]]
    assert record == {"label": None, "excluded": False, "not_object": True}


def test_not_object_is_a_hard_negative_on_an_explicit_frame(state, world):
    names = _pool(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.buffer_not_object(names[1])
    state.save_labels()
    state.promote_to_training("no")
    hard = [r for r in _rows(world, "no") if r["source"] == "rec_a" and r.get("not_object")]
    assert len(hard) == 1 and hard[0]["negative"] is True and hard[0]["label"] is None
    examples, _ = build_training_set(world.sets / "no.jsonl", world.sessions)
    assert not any(e.exhaustive for e in examples)
    assert sum(len(e.negatives) for e in examples) >= 1


def test_a_label_clears_not_object_and_not_object_clears_a_label(state, world):
    names = _pool(state)
    state.buffer_not_object(names[0])
    state.save_labels()
    state.buffer_label(names[0], "beta", "kinds")
    state.save_labels()
    record = json.loads((world.tiles / "_labels.json").read_text())[names[0]]
    assert record["label"] == "beta" and "not_object" not in record
    state.buffer_not_object(names[0])
    state.save_labels()
    record = json.loads((world.tiles / "_labels.json").read_text())[names[0]]
    assert record["label"] is None and record["not_object"] is True and "classdef" not in record


def test_discard_clears_not_object_and_stays_a_plain_discard(state, world):
    names = _pool(state)
    state.buffer_not_object(names[0])
    state.save_labels()
    state.set_excluded(names[0], True)
    record = json.loads((world.tiles / "_labels.json").read_text())[names[0]]
    assert record["excluded"] is True and "not_object" not in record
    state.buffer_label(names[1], "alpha", "kinds")
    state.save_labels()
    state.promote_to_training("d")
    row = next(r for r in _rows(world, "d") if r["source"] == "rec_a" and r["negative"])
    assert "not_object" not in row, "a discard is not marked as a not-an-object verdict"


def test_a_record_without_the_key_loads_unchanged(state, world):
    names = _pool(state)
    (world.tiles / "_labels.json").write_text(
        json.dumps({names[0]: {"label": "alpha", "excluded": False}, names[1]: "beta"})
    )
    grid = state.clusters()
    assert all(i["not_object"] is False for c in grid["clusters"] for i in c["items"])
    assert grid["counts"]["classes_assigned"] == 2


# ---------------------------------------------------------------- native sizes


def _sized_boxes() -> list[dict]:
    return [
        {"path": "f00.png", "left": 20, "top": 30, "width": 20, "height": 8},
        {"path": "f00.png", "left": 100, "top": 60, "width": 44, "height": 16},
        {"path": "f01.png", "left": 30, "top": 40, "width": 32, "height": 12},
    ]


def test_native_keeps_drawn_sizes_through_save_recut_and_promote(state, world):
    setconfig.write_set_config("nat", "box")
    out = state.save_drawn(_sized_boxes(), set_name="nat")
    assert out["size_mode"] == "native"
    stored = [
        json.loads(x)
        for x in next(world.labels.glob("rec_a.drawn-*.jsonl")).read_text().splitlines()
    ]
    assert [(c["width"], c["height"]) for c in stored] == [(20, 8), (44, 16), (32, 12)]
    tiles = [load_tile(p) for p in sorted(world.tiles.glob("rec_a_k*.npz"))]
    # each tile is its own stored box plus the crop pad, so the shapes differ
    assert len({(t.width, t.height) for t in tiles}) == 3
    before = [(t.width, t.height) for t in tiles]
    state.recut_pool()
    after = [
        (t.width, t.height)
        for t in (load_tile(p) for p in sorted(world.tiles.glob("rec_a_k*.npz")))
    ]
    assert after == before
    for name in state.unsorted_names():
        state.buffer_label(name, "alpha", "kinds")
    state.save_labels()
    state.promote_to_training("nat")
    assert {(r["width"], r["height"]) for r in _rows(world, "nat") if not r["negative"]} == {
        (20, 8),
        (44, 16),
        (32, 12),
    }
    assert json.loads((world.sets / "nat.meta.json").read_text())["size_mode"] == "native"


def test_uniform_still_fits_one_size_to_the_set(state, world):
    setconfig.write_set_config("uni", "heatmap")
    out = state.save_drawn(_sized_boxes(), set_name="uni")
    assert out["size_mode"] == "uniform" and (out["width"], out["height"]) == (32, 12)
    stored = [
        json.loads(x)
        for x in next(world.labels.glob("rec_a.drawn-*.jsonl")).read_text().splitlines()
    ]
    assert {(c["width"], c["height"]) for c in stored} == {(32, 12)}
    assert all("size_mode" not in c for c in stored)


def test_no_set_named_is_uniform_as_before(state, world):
    out = state.save_drawn(_sized_boxes())
    assert out["size_mode"] == "uniform" and world.labels == paths.LABELS_DIR
