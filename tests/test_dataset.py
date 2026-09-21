"""the cnn's training examples, assembled from review decisions."""

from __future__ import annotations

import json

import pytest

from smolsmort.detect.dataset import (
    MARGIN_X,
    MARGIN_Y,
    DatasetError,
    build_from,
    build_training_set,
    centre_of,
    summarise,
)


def candidate(path="f.jpg", left=500, top=300, width=132, height=20):
    return {"path": path, "left": left, "top": top, "width": width, "height": height}


def test_centre_without_a_rect_is_the_candidate_box_centre():
    assert centre_of(candidate(), None, 64, 14) == (566.0, 310.0)


def test_a_reviewed_rect_wins_over_the_detector_guess():
    """the rect is the human correction; it is crop-local so the padding has to come back off"""
    decision = {"rect": {"left": 30, "top": 12, "width": 64, "height": 14}}
    x, y = centre_of(candidate(), decision, 64, 14)
    # rect at exactly the padding offset means the object sits at the candidate's own origin
    assert (x, y) == (500 - MARGIN_X + 30 + 32.0, 300 - MARGIN_Y + 12 + 7.0)


def test_height_delta_moves_the_centre():
    base = centre_of(candidate(), {"rect": {"left": 0, "top": 0}}, 64, 14)
    taller = centre_of(candidate(), {"rect": {"left": 0, "top": 0}, "height_delta": 6}, 64, 14)
    assert taller[1] == base[1] + 3.0


def test_kept_candidates_become_centres(tmp_path):
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"x")
    examples = build_from([candidate()], {"0": {"keep": True}}, tmp_path)
    assert len(examples) == 1 and examples[0].object_count == 1


def test_a_discard_is_ignored_not_a_negative(tmp_path):
    """REGRESSION. Discard means "not wanted as a tile" - misaligned, clipped, redundant - not
    "not an object". Drawing the labels showed a discarded candidate sitting on a legible one,
    so training it as background would teach the model to suppress real objects."""
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"x")
    examples = build_from([candidate()], {"0": {"discard": True}}, tmp_path)
    assert examples[0].object_count == 0
    assert len(examples[0].ignore) == 1
    assert examples[0].negatives == []


def test_an_unreviewed_candidate_is_ignored(tmp_path):
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"x")
    examples = build_from([candidate()], {}, tmp_path)
    assert examples[0].object_count == 0 and len(examples[0].ignore) == 1


def test_candidates_group_by_frame(tmp_path):
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"x")
    cands = [candidate("a.jpg"), candidate("a.jpg", left=900), candidate("b.jpg")]
    examples = build_from(
        cands, {"0": {"keep": True}, "1": {"keep": True}, "2": {"keep": True}}, tmp_path
    )
    by_name = {e.path.name: e for e in examples}
    assert by_name["a.jpg"].object_count == 2
    assert by_name["b.jpg"].object_count == 1


def test_frames_that_do_not_exist_are_dropped(tmp_path):
    assert build_from([candidate("missing.jpg")], {"0": {"keep": True}}, tmp_path) == []


def test_summarise_counts_what_the_caller_needs(tmp_path):
    (tmp_path / "f.jpg").write_bytes(b"x")
    cands = [candidate(), candidate(left=900), candidate(left=1200)]
    got = summarise(build_from(cands, {"0": {"keep": True}, "1": {"discard": True}}, tmp_path))
    assert got["frames"] == 1 and got["objects"] == 1
    assert got["ignored"] == 2  # the discard AND the unreviewed one
    assert got["max_per_frame"] == 1


def test_exhaustive_frame_turns_a_clean_discard_into_a_negative(tmp_path):
    """THE PLAN'S VERIFY CASE. one frame, three candidates: kept at left=500, discarded well away
    at left=1500, discarded at left=510 whose centre sits inside the kept box. explicit -> both
    discards are ignored. exhaustive -> the misaligned one at 510 is dropped, the clean one at
    1500 becomes a negative."""
    (tmp_path / "f.jpg").write_bytes(b"x")
    cands = [candidate(left=500), candidate(left=1500), candidate(left=510)]
    decisions = {"0": {"keep": True}, "1": {"discard": True}, "2": {"discard": True}}

    explicit = build_from(cands, decisions, tmp_path)[0]
    assert explicit.object_count == 1
    assert len(explicit.ignore) == 2
    assert explicit.negatives == []

    exhaustive = build_from(cands, decisions, tmp_path, exhaustive_frames={"f.jpg"})[0]
    assert exhaustive.object_count == 1
    assert exhaustive.ignore == []
    assert len(exhaustive.negatives) == 1
    assert exhaustive.negatives[0][0] == 1500 + 132 / 2.0  # centre x of the candidate at left=1500
    assert exhaustive.exhaustive is True


def test_exhaustive_frame_result_is_order_independent(tmp_path):
    """REGRESSION. a discard exempted by a kept box can come BEFORE the kept candidate in list
    order, so testing extent needs two passes - one pass would miss the exemption."""
    (tmp_path / "f.jpg").write_bytes(b"x")
    cands = [candidate(left=510), candidate(left=1500), candidate(left=500)]
    decisions = {"0": {"discard": True}, "1": {"discard": True}, "2": {"keep": True}}
    example = build_from(cands, decisions, tmp_path, exhaustive_frames={"f.jpg"})[0]
    assert example.object_count == 1
    assert example.ignore == []
    assert len(example.negatives) == 1


def test_unreviewed_candidate_on_an_exhaustive_frame_is_a_negative(tmp_path):
    (tmp_path / "f.jpg").write_bytes(b"x")
    example = build_from([candidate()], {}, tmp_path, exhaustive_frames={"f.jpg"})[0]
    assert example.object_count == 0
    assert example.negatives == [(566.0, 310.0)]
    assert example.ignore == []


def test_kept_without_a_rect_uses_the_candidates_own_size(tmp_path):
    (tmp_path / "f.jpg").write_bytes(b"x")
    example = build_from([candidate(width=132, height=20)], {"0": {"keep": True}}, tmp_path)[0]
    assert example.sizes == [(132, 20)]


def test_kept_with_a_rect_uses_uniform_width_and_delta_height(tmp_path):
    (tmp_path / "f.jpg").write_bytes(b"x")
    decision = {"keep": True, "rect": {"left": 0, "top": 0}, "height_delta": 6}
    example = build_from(
        [candidate()], {"0": decision}, tmp_path, uniform_width=64, uniform_height=14
    )[0]
    assert example.sizes == [(64, 20)]


def test_summarise_counts_exhaustive_frames(tmp_path):
    (tmp_path / "f.jpg").write_bytes(b"x")
    examples = build_from([candidate()], {}, tmp_path, exhaustive_frames={"f.jpg"})
    assert summarise(examples)["exhaustive_frames"] == 1


def _training_row(recording="rec", frame="f.jpg", left=500, top=300, width=132, height=20, **extra):
    return {
        "recording": recording,
        "frame": frame,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "label": "object",
        **extra,
    }


def test_build_training_set_fills_sizes_and_marks_exhaustive_frames(tmp_path):
    frames = tmp_path / "rec" / "frames"
    frames.mkdir(parents=True)
    (frames / "f.jpg").write_bytes(b"x")
    row = _training_row(width=132, height=20, exhaustive=True)
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps(row) + "\n")
    examples, _ = build_training_set(path, tmp_path)
    assert examples[0].sizes == [(132, 20)]
    assert examples[0].exhaustive is True


def test_build_training_set_allows_mixed_resolutions_when_asked(tmp_path):
    from PIL import Image

    for recording, size in (("rec_a", (200, 50)), ("rec_b", (400, 100))):
        frames = tmp_path / recording / "frames"
        frames.mkdir(parents=True)
        Image.new("RGB", size).save(frames / "f.jpg")
    rows = [_training_row(recording="rec_a"), _training_row(recording="rec_b")]
    path = tmp_path / "training.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    examples, _ = build_training_set(path, tmp_path, refuse_mixed_resolutions=False)
    assert len(examples) == 2

    with pytest.raises(DatasetError):
        build_training_set(path, tmp_path)


def test_a_row_missing_a_required_key_is_skipped_and_counted(tmp_path):
    frames = tmp_path / "rec" / "frames"
    frames.mkdir(parents=True)
    (frames / "f.jpg").write_bytes(b"x")
    good = _training_row(frame="f.jpg")
    bad = _training_row(frame="g.jpg")
    del bad["frame"]
    path = tmp_path / "training.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (good, bad)) + "\n")

    examples, classes = build_training_set(path, tmp_path)
    assert len(examples) == 1 and examples[0].object_count == 1
    assert classes.skipped == {"frame": 1}


def test_a_non_numeric_box_value_is_skipped_and_counted(tmp_path):
    frames = tmp_path / "rec" / "frames"
    frames.mkdir(parents=True)
    (frames / "f.jpg").write_bytes(b"x")
    good = _training_row(frame="f.jpg")
    bad = _training_row(frame="f.jpg", left="abc")
    path = tmp_path / "training.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (good, bad)) + "\n")

    examples, classes = build_training_set(path, tmp_path)
    assert examples[0].object_count == 1
    assert classes.skipped == {"left: non-numeric": 1}


def test_the_dataset_margins_are_the_review_tools_margins():
    """detect/dataset.py keeps its own copy so the library needs nothing from review; the two
    must not drift, or a centre computed here lands elsewhere than the tool drew it"""
    from smolsmort.detect import dataset
    from smolsmort.review import paths

    assert (dataset.MARGIN_X, dataset.MARGIN_Y) == (paths.MARGIN_X, paths.MARGIN_Y)
