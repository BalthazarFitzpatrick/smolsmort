"""the review state, exercised directly - no socket. every test builds a tmp_path world and a
state around it, so nothing here can touch real recordings, boxes or tiles."""

from __future__ import annotations

import json

import numpy as np
import pytest
from review_world import BOX, drawn_boxes, make_world

from smolsmort.review import paths
from smolsmort.review.state import ReviewState
from smolsmort.review.tiles import Tile, cut_tile, load_tile, save_tile


@pytest.fixture
def world(tmp_path, monkeypatch):
    return make_world(tmp_path, monkeypatch, recordings=("rec_a", "set/two"))


@pytest.fixture
def state(world):
    state = ReviewState(
        world.sessions, [], world.tiles / "_unbound.decisions.json", "", world.tiles
    )
    state.bind_recording("session", "rec_a")
    return state


def _draw(state) -> dict:
    return state.save_drawn(drawn_boxes())


def _tile_names(state) -> list[str]:
    return state.unsorted_names()


# ---------------------------------------------------------------- find: drawing and cutting


def test_saving_drawn_boxes_writes_a_set_decisions_and_tiles(state, world):
    out = _draw(state)
    assert out["count"] == 6 and out["tiles"] == 6
    assert (out["width"], out["height"]) == BOX
    files = sorted(p.name for p in world.labels.glob("rec_a.drawn-*"))
    assert len(files) == 2 and files[1].endswith(".candidates.jsonl")
    assert files[0].endswith(".candidates.decisions.json")
    decisions = json.loads((world.labels / files[0]).read_text())
    assert all(d["keep"] for d in decisions.values())
    assert _tile_names(state)[0] == "rec_a_k00000"


def test_a_save_that_would_shrink_a_set_goes_to_a_new_file(state, world):
    _draw(state)
    state.save_drawn(drawn_boxes()[:2])
    assert len(list(world.labels.glob("rec_a.drawn-*.candidates.jsonl"))) == 2


def test_saving_needs_a_bound_dataset(state):
    state.unbind()
    assert "no dataset bound" in state.save_drawn(drawn_boxes())["error"]


def test_negatives_are_saved_flagged_and_come_back_split(state):
    state.save_drawn(drawn_boxes()[:2], negatives=[{"path": "f00.png", "left": 5, "top": 5}])
    reopened = state.draw_frames(percent=100)
    assert reopened["drawn"] == 2
    assert sum(len(v) for v in reopened["negatives"].values()) == 1


def test_draw_frames_spreads_evenly_and_carries_drawn_boxes(state):
    assert state.draw_frames(sample=2)["frames"] == ["f00.png", "f01.png"]
    _draw(state)
    state.unbind()
    state.bind_recording("session", "rec_a")
    frames = state.draw_frames(percent=34)
    assert frames["ready"] and frames["drawn"] == 6
    assert set(frames["boxes"]) == {"f00.png", "f01.png", "f02.png"}


def test_draw_frames_says_why_when_nothing_is_bound(state):
    state.unbind()
    assert state.draw_frames()["ready"] is False


# ---------------------------------------------------------------- crop geometry: one code path


def _cut_shape(state, world):
    _draw(state)
    return load_tile(world.tiles / "rec_a_k00000.npz").rgb.shape[:2]


def test_percent_free_is_todays_geometry(state, world):
    """the default rule reproduces the fraction-per-side pad the find tab has always used"""
    state.set_crop_settings(0.25, 0.9)
    pad_x, pad_y = round(BOX[0] * 0.25), round(BOX[1] * 0.9)
    assert _cut_shape(state, world) == (BOX[1] + 2 * pad_y, BOX[0] + 2 * pad_x)


def test_absolute_mode_cuts_the_requested_crop_size(state, world):
    state.uniform_width, state.uniform_height = BOX  # the preview is for the box being drawn
    state.set_crop_settings(crop_mode="absolute", crop_w=50, crop_h=30, aspect="free")
    assert _cut_shape(state, world) == (30, 50)
    assert state.crop_settings()["crop"] == [50, 30]


def test_fixed_aspect_keeps_the_box_shape(state, world):
    state.set_crop_settings(crop_mode="absolute", crop_w=60, aspect="fixed")
    height, width = _cut_shape(state, world)
    assert width == 60 and height == round(60 * BOX[1] / BOX[0])
    state.set_crop_settings(0.5, 0.1, crop_mode="percent", aspect="fixed")
    # one fraction for both axes, so pad_y is ignored
    assert state.crop_pads(*BOX) == (round(BOX[0] * 0.5), round(BOX[1] * 0.5))


def test_a_crop_is_never_smaller_than_the_box(state):
    state.set_crop_settings(crop_mode="absolute", crop_w=5, crop_h=5, aspect="free")
    assert state.crop_pads(*BOX) == (0, 0)


def test_the_preview_and_the_cut_agree(state, world):
    state.uniform_width, state.uniform_height = BOX
    state.set_crop_settings(crop_mode="absolute", crop_w=44, crop_h=26, aspect="free")
    preview = state.crop_settings()["crop"]
    assert list(reversed(_cut_shape(state, world))) == preview


def test_bad_crop_values_are_refused_by_name(state):
    with pytest.raises(ValueError, match="crop_mode"):
        state.set_crop_settings(crop_mode="diagonal")
    with pytest.raises(ValueError, match="aspect"):
        state.set_crop_settings(aspect="square")


def test_an_old_settings_file_with_only_pad_fields_still_loads(state, tmp_path):
    settings = tmp_path / "crop.json"
    settings.write_text(json.dumps({"pad_x": 0.4, "pad_y": 1.2}))
    state.crop_file = settings
    state.load_crop_settings()
    got = state.crop_settings()
    assert (got["pad_x"], got["pad_y"], got["crop_mode"], got["aspect"]) == (
        0.4,
        1.2,
        "percent",
        "free",
    )


def test_settings_survive_a_restart(state, world, tmp_path):
    settings = tmp_path / "crop.json"
    state.crop_file = settings
    state.set_crop_settings(crop_mode="absolute", crop_w=70, crop_h=40, aspect="fixed")
    again = ReviewState(
        world.sessions, [], world.tiles / "x.json", "", world.tiles, crop_file=settings
    )
    assert again.crop_settings()["crop_mode"] == "absolute"
    assert again.crop_w == 70 and again.aspect == "fixed"


def test_a_box_near_the_edge_falls_back_to_the_bare_box(state, world):
    state.set_crop_settings(2.0, 2.0)
    out = state.save_drawn([{"path": "f00.png", "left": 2, "top": 2, "width": 30, "height": 10}])
    assert out["tiles"] == 1
    assert load_tile(world.tiles / "rec_a_k00000.npz").rgb.shape[:2] == (10, 30)


# ---------------------------------------------------------------- find: binding and the list


def test_recordings_are_always_listed_with_their_drawn_work(state):
    _draw(state)
    rows = {r["name"]: r for r in state.bindable_recordings()["rows"]}
    assert set(rows) == {"rec_a", "set/two"}, "a nested recording is listed by its path"
    assert rows["rec_a"]["boxes"] == 6 and rows["rec_a"]["frame_count"] == 3
    assert state.bindable_recordings()["current"].startswith("rec_a.drawn-")


def test_binding_a_missing_recording_raises_and_an_unknown_kind_too(state):
    with pytest.raises(FileNotFoundError):
        state.bind_recording("session", "nope")
    with pytest.raises(ValueError, match="unknown kind"):
        state.bind_recording("banana", "x")


def test_unbinding_moves_nothing_on_disk(state, world):
    _draw(state)
    before = sorted(p.name for p in world.labels.iterdir())
    state.unbind()
    assert sorted(p.name for p in world.labels.iterdir()) == before
    assert state.session_tag == ""


def test_paging_is_stable_and_reviewed_rows_drop_out(state):
    _draw(state)  # save_drawn binds the set it wrote
    first = state.page(0, 4)
    assert first["total"] == 6 and first["total_pages"] == 2
    state.mark_reviewed(0)
    assert state.page(0, 4)["total"] == 5
    assert state.page(0, 4)["rows"][0]["index"] == 1


def test_the_guess_is_a_prefill_from_the_optional_guesser(world):
    class Guesser:
        def guess(self, candidate):
            return "alpha"

    state = ReviewState(
        world.sessions,
        [{"path": "f00.png", "left": 1, "top": 1, "width": 5, "height": 5}],
        world.tiles / "d.json",
        "t",
        world.tiles,
        guesser=Guesser(),
    )
    assert state.page(0, 4)["rows"][0]["guess"] == "alpha"
    state.guesser = None
    assert state.page(0, 4)["rows"][0]["guess"] is None


def test_height_delta_accumulates_and_floors_at_four(state):
    state.candidates = [{"path": "f00.png", "left": 1, "top": 1, "width": 5, "height": 5}]
    assert state.set_height_delta(0, 3) == 3
    assert state.set_height_delta(0, 2) == 5
    assert state.effective_height(0) == state.uniform_height + 5
    state.set_height_delta(0, -1000)
    assert state.effective_height(0) == 4


def test_the_crop_route_serves_a_padded_png_of_the_right_size(state):
    from io import BytesIO

    from PIL import Image

    state.candidates = [{"path": "f00.png", "left": 60, "top": 50, "width": 30, "height": 10}]
    image = Image.open(BytesIO(state.crop_bytes(0)))
    assert image.size == (30 + 2 * paths.MARGIN_X, 10 + 2 * paths.MARGIN_Y)
    with pytest.raises(IndexError):
        state.crop_bytes(5)


# ---------------------------------------------------------------- tiles on disk


def test_a_tile_round_trips_and_keeps_the_on_disk_layout(tmp_path):
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    tile = cut_tile(frame, "x_k00000", top=1, left=2, height=3, width=4)
    save_tile(tile, tmp_path)
    with np.load(tmp_path / "x_k00000.npz") as data:
        assert set(data.files) == {"rgb", "mask"}
        assert data["rgb"].shape == (3, 4, 3) and data["mask"].dtype == bool
    assert isinstance(load_tile(tmp_path / "x_k00000.npz"), Tile)


def test_a_tile_written_with_a_masked_corner_still_opens(tmp_path):
    np.savez(
        tmp_path / "old_k00001.npz", rgb=np.zeros((4, 4, 3), np.uint8), mask=np.eye(4, dtype=bool)
    )
    assert int(load_tile(tmp_path / "old_k00001.npz").mask.sum()) == 4


# ---------------------------------------------------------------- select: judging and the buffer


def _pool_names(state):
    _draw(state)
    return _tile_names(state)


def test_manual_label_buffers_and_writes_nothing(state, world):
    names = _pool_names(state)
    assert state.buffer_label(names[0], "alpha", "kinds") == 1
    assert state.buffer_label(names[1], "beta", "kinds") == 2
    assert not (world.tiles / "_labels.json").exists()
    assert state.labels_buffer() == {"pending": 2, "datasets": [{"tag": "rec_a", "pending": 2}]}


def test_save_labels_flushes_atomically_and_empties_the_buffer(state, world):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.buffer_label(names[1], "beta", "kinds")
    assert state.save_labels() == {"saved": 2}
    stored = json.loads((world.tiles / "_labels.json").read_text())
    assert stored[names[0]] == {"label": "alpha", "excluded": False, "classdef": "kinds"}
    assert state.labels_buffer() == {"pending": 0, "datasets": []}
    assert not list(world.tiles.glob("*.tmp")), "no half-written temp file is left behind"
    assert state.save_labels() == {"saved": 0}


def test_an_unknown_tile_or_empty_label_is_not_buffered(state):
    _pool_names(state)
    assert state.buffer_label("nope_k00000", "alpha", "kinds") is None
    assert state.buffer_label("rec_a_k00000", "", "kinds") is None


def test_promotion_reads_disk_only(state, world):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    out = state.promote_to_training("s")
    assert "nothing to promote" in out["error"] and out["pending"] == 1
    state.save_labels()
    assert state.promote_to_training("s")["rows"] >= 1


def test_the_grid_shows_pending_assignments_and_counts_them(state):
    names = _pool_names(state)
    before = state.clusters()["counts"]
    assert before == {"boxes_loaded": 6, "classes_assigned": 0}
    state.buffer_label(names[0], "alpha", "kinds")
    grid = state.clusters()
    assert grid["counts"] == {"boxes_loaded": 6, "classes_assigned": 1}
    first = grid["clusters"][0]
    assert first["label"] == "alpha" and first["items"][0]["pending"] is True
    assert [c["label"] for c in grid["clusters"]][-1] == "unjudged"


def test_clusters_order_classes_then_unjudged_then_not_a_class(state):
    names = _pool_names(state)
    state.buffer_label(names[0], "beta", "kinds")
    state.buffer_label(names[1], "alpha", "kinds")
    state.save_labels()
    state.set_excluded(names[2], True)
    assert [c["label"] for c in state.clusters()["clusters"]] == [
        "alpha",
        "beta",
        "unjudged",
        "not a class",
    ]


def test_an_empty_pool_has_zero_counts(state):
    assert state.clusters() == {
        "clusters": [],
        "counts": {"boxes_loaded": 0, "classes_assigned": 0},
    }


def test_excluding_is_declarative_and_drops_a_pending_label(state, world):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    assert state.set_excluded(names[0], True) is True
    assert state.pending_count() == 0, "the later press wins over an unsaved class"
    assert state.set_excluded(names[0], True) is True, "a repeat press changes nothing"
    log = (world.tiles / "_exclusions.jsonl").read_text().splitlines()
    assert len(log) == 1, "a no-op press is not history"
    assert state.set_excluded("missing_k00000", True) is None
    assert state.set_excluded(names[0], False) is False
    assert not (world.tiles / "_labels.json").read_text().count(names[0])


def test_saving_a_class_un_excludes(state):
    names = _pool_names(state)
    state.set_excluded(names[0], True)
    state.buffer_label(names[0], "alpha", "kinds")
    state.save_labels()
    record = state.pool_records()[names[0]]
    assert record["label"] == "alpha" and record["excluded"] is False


# ---------------------------------------------------------------- closing and opening sources


def test_closing_with_unsaved_labels_reports_and_stays_open(state):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    assert state.close_pool_source("rec_a") == {"tag": "rec_a", "unsaved": 1}
    assert state.pool_sources()["sources"][0]["closed"] is False
    assert state.pending_count() == 1, "the buffer is kept unless the request says discard"


def test_discarding_drops_the_buffer_and_the_saved_judgements_but_not_the_tiles(state, world):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.buffer_label(names[1], "alpha", "kinds")
    state.save_labels()
    state.buffer_label(names[2], "beta", "kinds")
    out = state.close_pool_source("rec_a", discard=True)
    assert out == {"tag": "rec_a", "closed": 6, "discarded": 2}
    assert state.pending_count() == 0 and state.pool_records() == {}
    assert len(list(world.tiles.glob("*.npz"))) == 6


def test_closing_a_saved_source_moves_nothing_and_keeps_its_judgements(state, world):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.save_labels()
    out = state.close_pool_source("rec_a")
    assert out["closed"] == 6 and out["discarded"] == 0
    assert len(list(world.tiles.glob("*.npz"))) == 6
    assert len(state.pool_records()) == 1
    assert state.closed_tags() == {"rec_a"}


def test_a_closed_source_leaves_the_grid_but_stays_listed(state):
    _pool_names(state)
    state.close_pool_source("rec_a")
    assert state.clusters()["clusters"] == []
    assert state.pool_sources()["sources"] == [{"tag": "rec_a", "tiles": 6, "closed": True}]


def test_reopening_is_instant_because_nothing_was_moved(state):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.save_labels()
    state.close_pool_source("rec_a")
    out = state.open_pool_source("rec_a")
    assert out["tiles"] == 6 and out["judged"] == 1
    assert state.clusters()["counts"]["boxes_loaded"] == 6


def test_the_page_can_ask_what_closing_would_cost(state):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.save_labels()
    state.buffer_label(names[1], "beta", "kinds")
    assert state.pool_source_state("rec_a") == {
        "tag": "rec_a",
        "tiles": 6,
        "judged": 1,
        "unsaved": 1,
    }
    assert state.close_pool_source("")["error"]


# ---------------------------------------------------------------- datasets, realign, recut


def test_a_dataset_not_in_the_pool_is_openable_and_opening_cuts_its_tiles(state, world):
    _draw(state)
    for path in world.tiles.glob("*.npz"):
        path.unlink()
    listed = state.openable_datasets()["datasets"]
    assert [d["tag"] for d in listed] == ["rec_a"] and listed[0]["source"] == "find"
    name = listed[0]["file"]
    out = state.open_dataset(name)
    assert out["tiles"] == 6 and state.openable_datasets() == {"datasets": []}
    assert "error" in state.open_dataset("nope.candidates.jsonl")


def test_an_empty_sweep_file_is_not_offered(state, world):
    (world.labels / "rec_a.cnn-20260101-000000.candidates.jsonl").write_text("")
    assert state.openable_datasets() == {"datasets": []}


def test_realigning_a_tile_keeps_its_name_and_its_label(state, world):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.save_labels()
    info = state.pool_box_info(names[0])
    assert (
        info["rect"]["left"] == paths.MARGIN_X
        and info["bounds"]["width"] == 30 + 2 * paths.MARGIN_X
    )
    before = load_tile(world.tiles / f"{names[0]}.npz").rgb.copy()
    out = state.realign_pool_tile(names[0], paths.MARGIN_X + 5, paths.MARGIN_Y + 3)
    assert out == {"name": names[0], "left": 45, "top": 43}
    assert not np.array_equal(before, load_tile(world.tiles / f"{names[0]}.npz").rgb)
    assert state.pool_records()[names[0]]["label"] == "alpha"
    assert "error" in state.realign_pool_tile("ghost_k00000", 0, 0)


def test_recutting_at_a_new_crop_rule_overwrites_in_place_and_keeps_labels(state, world):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "kinds")
    state.save_labels()
    state.set_crop_settings(crop_mode="absolute", crop_w=60, crop_h=40, aspect="free")
    out = state.recut_pool()
    assert out["recut"][0]["tiles"] == 6 and out["failed"] == []
    assert load_tile(world.tiles / f"{names[0]}.npz").rgb.shape[:2] == (40, 60)
    assert len(list(world.tiles.glob("*.npz"))) == 6
    assert state.pool_records()[names[0]]["label"] == "alpha"


def test_a_thumbnail_comes_from_the_tiles_own_recording(state):
    from io import BytesIO

    from PIL import Image

    names = _pool_names(state)
    state.unbind()
    thumb = Image.open(BytesIO(state.unsorted_thumb_bytes(names[0])))
    assert thumb.size[0] > 30 and thumb.size[1] >= 10
    assert state.unsorted_thumb_bytes("ghost_k00000") is None


# ---------------------------------------------------------------- promotion


def _judge_all(state, split: int = 3) -> list[str]:
    names = _pool_names(state)
    for i, name in enumerate(names):
        state.buffer_label(name, "alpha" if i < split else "beta", "kinds")
    state.save_labels()
    return names


def test_promoting_writes_a_named_set_and_its_provenance_sidecar(state, world):
    _judge_all(state)
    out = state.promote_to_training("first")
    assert out["mode"] == "new" and out["rows"] == 6
    assert out["classes"] == {"alpha": 3, "beta": 3}
    assert out["synthetic"] > 0 and out["negatives"] == out["synthetic"]
    rows = [json.loads(x) for x in (world.sets / "first.jsonl").read_text().splitlines()]
    assert {r["recording"] for r in rows} == {"rec_a"}
    meta = json.loads((world.sets / "first.meta.json").read_text())
    assert meta["sources"][0]["source"] == "rec_a" and meta["mode"] == "new"


def test_a_name_that_exists_is_a_decision_not_an_overwrite(state, world):
    _judge_all(state)
    state.promote_to_training("first")
    again = state.promote_to_training("first")
    assert again["exists"] is True and again["existing_rows"] > 0
    assert state.promote_to_training("first", "overwrite")["mode"] == "overwrite"
    assert state.promote_to_training("first", "merge")["mode"] == "merge"


def test_merging_keeps_rows_of_other_sources_and_replaces_its_own(state, world):
    _judge_all(state)
    state.promote_to_training("s")
    path = world.sets / "s.jsonl"
    other = {
        "recording": "elsewhere",
        "frame": "z.png",
        "left": 1,
        "top": 1,
        "width": 3,
        "height": 3,
        "label": "beta",
        "negative": False,
        "source": "other_src",
    }
    path.write_text(path.read_text() + json.dumps(other) + "\n")
    state.promote_to_training("s", "merge")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert sum(r["source"] == "other_src" for r in rows) == 1
    assert sum(r["source"] == "rec_a" and not r["negative"] for r in rows) == 6


def test_a_promote_with_nothing_labelled_leaves_the_last_set_alone(state, world):
    _judge_all(state)
    state.promote_to_training("keep")
    good = (world.sets / "keep.jsonl").read_text()
    (world.tiles / "_labels.json").write_text("{}")
    out = state.promote_to_training("keep", "overwrite")
    assert "nothing to promote" in out["error"] and "left alone" in out["error"]
    assert (world.sets / "keep.jsonl").read_text() == good


def test_the_reason_names_a_missing_recording(state, world):
    _judge_all(state)
    for path in (world.sessions / "rec_a" / "frames").glob("*"):
        path.unlink()
    (world.sessions / "rec_a" / "frames").rmdir()
    (world.sessions / "rec_a").rmdir()
    assert "failed to find their recording" in state.promote_to_training("x")["error"]


def test_two_definitions_in_one_pool_refuse_promotion_by_name(state):
    names = _pool_names(state)
    state.buffer_label(names[0], "alpha", "first_scheme")
    state.buffer_label(names[1], "alpha", "second_scheme")
    state.save_labels()
    error = state.promote_to_training("x")["error"]
    assert "first_scheme" in error and "second_scheme" in error


def test_an_excluded_swept_box_becomes_a_hard_negative_row(state, world):
    stamp = "20260101-000000"
    boxes = [
        {"path": "f00.png", "left": 20, "top": 20, "width": 30, "height": 10, "score": 0.9},
        {"path": "f01.png", "left": 60, "top": 20, "width": 30, "height": 10, "score": 0.8},
    ]
    (world.labels / f"rec_a.cnn-{stamp}.candidates.jsonl").write_text(
        "".join(json.dumps(b) + "\n" for b in boxes)
    )
    state.open_dataset(f"rec_a.cnn-{stamp}.candidates.jsonl")
    tag = f"rec_a_cnn-{stamp}"
    state.buffer_label(f"{tag}_k00000", "alpha", "kinds")
    state.save_labels()
    state.set_excluded(f"{tag}_k00001", True)
    rows = [
        json.loads(x)
        for x in (
            state.promote_to_training("h") and (world.sets / "h.jsonl").read_text().splitlines()
        )
    ]
    hard = [r for r in rows if r["negative"] and r["source"] == tag]
    assert len(hard) == 1 and hard[0]["left"] == 60
    assert [r["label"] for r in rows if r["source"] == tag and not r["negative"]] == ["alpha"]


def test_a_drag_correction_reaches_the_training_row(state, world):
    names = _judge_all(state)
    sidecar = next(world.labels.glob("rec_a.drawn-*.decisions.json"))
    decisions = json.loads(sidecar.read_text())
    decisions["0"]["rect"] = {
        "left": paths.MARGIN_X + 7,
        "top": paths.MARGIN_Y,
        "width": 30,
        "height": 10,
    }
    sidecar.write_text(json.dumps(decisions))
    state.promote_to_training("c")
    rows = [json.loads(x) for x in (world.sets / "c.jsonl").read_text().splitlines()]
    first = next(
        r for r in rows if r["frame"] == "f00.png" and r["top"] == 40 and not r["negative"]
    )
    assert first["left"] == 47 and names[0]


def test_set_names_cannot_escape_the_sets_directory(state, world):
    _judge_all(state)
    out = state.promote_to_training("../../evil name")
    assert (world.sets / "evil_name.jsonl").is_file() and ".." not in out["out"]


def test_the_training_sets_list_carries_class_counts(state):
    _judge_all(state)
    state.promote_to_training("listed")
    only = state.training_sets()["sets"][0]
    assert only["name"] == "listed" and only["classes"] == {"alpha": 3, "beta": 3}


# ---------------------------------------------------------------- bases and the directory tree


def test_the_tree_lists_one_level_and_says_which_rows_open(tmp_path, world):
    (tmp_path / "walk" / "one" / "deeper").mkdir(parents=True)
    (tmp_path / "walk" / "leaf").mkdir()
    (tmp_path / "walk" / ".hidden").mkdir()
    state = ReviewState(world.sessions, [], world.tiles / "d.json", "", world.tiles)
    state.bases["root"] = str(tmp_path / "walk")
    out = state.dir_tree()
    names = {e["name"]: e for e in out["entries"]}
    assert set(names) == {"one", "leaf"} and names["one"]["has_children"] is True
    assert names["leaf"]["has_children"] is False and out["parent"] is None


def test_dir_tree_can_be_rooted_at_each_configured_base(state, world):
    (world.tiles / "poolsub").mkdir()
    (world.checkpoints / "runs").mkdir()
    tiles = state.dir_tree(root="tiles")
    checkpoints = state.dir_tree(root="checkpoints")
    recordings = state.dir_tree(root="recordings")
    assert [e["name"] for e in tiles["entries"]] == ["poolsub"]
    assert [e["name"] for e in checkpoints["entries"]] == ["runs"]
    assert {e["name"] for e in recordings["entries"]} == {"rec_a", "set"}
    assert tiles["root"] != checkpoints["root"] and tiles["base"] == "tiles"


def test_dir_tree_refuses_paths_outside_the_chosen_base(state, world, tmp_path):
    (world.tiles / "poolsub").mkdir()
    out = state.dir_tree(under=str(world.sessions), root="tiles")
    assert out["here"] == str(world.tiles.resolve()) and out["refused"] is True
    assert state.dir_tree(under="/etc", root="checkpoints")["refused"] is True
    inside = state.dir_tree(under=str(world.tiles / "poolsub"), root="tiles")
    assert inside["refused"] is False and inside["parent"] == str(world.tiles.resolve())
    with pytest.raises(ValueError, match="root must be one of"):
        state.dir_tree(root="elsewhere")


def test_the_tree_refuses_to_leave_the_settings_root(tmp_path, world):
    (tmp_path / "inside").mkdir(exist_ok=True)
    state = ReviewState(world.sessions, [], world.tiles / "d.json", "", world.tiles)
    state.bases["root"] = str(tmp_path)
    assert state.dir_tree("/etc")["here"] == str(tmp_path.resolve())
    assert state.dir_tree(str(tmp_path / "inside"))["here"] == str((tmp_path / "inside").resolve())


def test_repointing_the_pool_moves_the_directories_derived_from_it(state, tmp_path):
    state._dataset_cache["stale"] = object()
    state._record_origin["a_k00000"] = state.pools[0]
    state.set_bases({"pool": str(tmp_path / "new_tiles")})
    assert state.pools == [tmp_path / "new_tiles"]
    assert state._record_origin == {} and state._dataset_cache == {}


def test_repointing_sessions_and_labels_moves_the_module_paths(state, tmp_path):
    state.set_bases({"sessions": str(tmp_path / "s2"), "labels": str(tmp_path / "l2")})
    assert tmp_path / "s2" == paths.SESSIONS_DIR and tmp_path / "l2" == paths.LABELS_DIR


def test_an_unknown_base_key_is_ignored_rather_than_stored(state, tmp_path):
    out = state.set_bases({"nonsense": "/tmp", "labels": str(tmp_path)})
    assert "nonsense" not in out and out["labels"] == str(tmp_path)


def test_a_pool_may_be_a_list_or_a_bare_string(state, tmp_path):
    state.set_bases({"pool": [str(tmp_path / "a"), str(tmp_path / "b")]})
    assert state.pools == [tmp_path / "a", tmp_path / "b"]
    state.set_bases({"pool": str(tmp_path / "c")})
    assert state.pools == [tmp_path / "c"]


def test_bases_survive_a_restart(state, world, tmp_path):
    state.bases_file = tmp_path / "bases.json"
    state.set_bases({"labels": str(tmp_path / "kept")})
    fresh = ReviewState(
        world.sessions, [], world.tiles / "d.json", "", world.tiles, bases_file=state.bases_file
    )
    fresh.load_saved_bases()
    assert fresh.bases["labels"] == str(tmp_path / "kept")


def test_dir_list_offers_sessions_and_candidate_files(state):
    _draw(state)
    listed = state.dir_list()
    assert len(listed["sessions"]) == 2 and listed["labels"][0].endswith(".candidates.jsonl")


def test_a_second_pool_is_read_but_never_written_to(state, world, tmp_path):
    names = _pool_names(state)
    extra = tmp_path / "extra_pool"
    extra.mkdir()
    (world.tiles / f"{names[0]}.npz").rename(extra / f"{names[0]}.npz")
    state.set_bases({"pool": [str(world.tiles), str(extra)]})
    state.buffer_label(names[0], "alpha", "kinds")
    state.save_labels()
    assert names[0] in json.loads((extra / "_labels.json").read_text())
    assert not (world.tiles / "_labels.json").exists()
