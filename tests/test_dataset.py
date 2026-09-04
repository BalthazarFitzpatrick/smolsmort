"""the cnn's training examples, assembled from review decisions."""

from __future__ import annotations

from smolsmort.detect.dataset import MARGIN_X, MARGIN_Y, build_from, centre_of, summarise


def candidate(path="f.jpg", left=500, top=300, width=132, height=20):
    return {"path": path, "left": left, "top": top, "width": width, "height": height}


def test_centre_without_a_rect_is_the_candidate_box_centre():
    assert centre_of(candidate(), None, 64, 14) == (566.0, 310.0)


def test_a_reviewed_rect_wins_over_the_detector_guess():
    """the rect is the human correction; it is crop-local so the padding has to come back off"""
    decision = {"rect": {"left": 30, "top": 12, "width": 64, "height": 14}}
    x, y = centre_of(candidate(), decision, 64, 14)
    # rect at exactly the padding offset means the plate sits at the candidate's own origin
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
    so training it as background would teach the model to suppress real plates."""
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
