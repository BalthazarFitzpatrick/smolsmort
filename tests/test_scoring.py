"""scoring a detector against reviewed labels, and the pass marks the plan committed to."""

from __future__ import annotations

import pytest

from smolsmort.detect.box import Box as PlateBox
from smolsmort.detect.model import Peak
from smolsmort.detect.scoring import (
    MAX_FALSE_POSITIVE_RATE,
    TARGET_RECALL,
    ScoringError,
    boxes_from_candidates,
    boxes_from_peaks,
    iou_match,
    score,
)


def _box(left=800, top=400):
    return PlateBox(left=left, top=top, width=132, height=12)


def test_a_perfect_detector_passes():
    truths = [[_box()], [_box(left=200)], [], []]
    result = score([[_box()], [_box(left=200)], [], []], truths)
    assert result.recall == 1.0
    assert result.false_positive_rate == 0.0
    assert result.passes


def test_a_frame_with_no_plate_is_where_a_false_positive_is_counted():
    """the three measured false-positive classes only appear on frames with nothing to find"""
    result = score([[], [_box()]], [[], []])
    assert result.frames_without_object == 2
    assert result.false_positive_frames == 1
    assert result.false_positive_rate == 0.5
    assert not result.passes


def test_a_near_miss_on_the_same_bar_still_counts_as_a_hit():
    """eight pixels of vertical offset drops IoU to 0.12 on a shape this thin, so matching is by
    shared horizontal span and row distance - see labels.PlateBox.overlaps
    """
    result = score([[_box(left=810, top=406)]], [[_box(left=800, top=400)]])
    assert result.hits == 1
    assert result.spurious == 0


def test_a_box_on_a_different_plate_is_spurious_not_a_hit():
    result = score([[_box(left=1800)]], [[_box(left=200)]])
    assert result.hits == 0
    assert result.misses == 1
    assert result.spurious == 1


def test_an_extra_box_beside_a_correct_one_is_counted_but_the_frame_still_hits():
    result = score([[_box(), _box(left=1900)]], [[_box()]])
    assert result.hits == 1
    assert result.spurious == 1


def test_the_thresholds_are_the_ones_the_plan_committed_to():
    assert TARGET_RECALL == 0.90
    assert MAX_FALSE_POSITIVE_RATE == 0.05


def test_a_detector_at_the_teachers_level_does_not_beat_it():
    """~75% recall and ~75% precision is what the hand-written rules already deliver, so matching
    it is not a reason to have built anything
    """
    truths = [[_box()]] * 8 + [[]] * 4
    predictions = [[_box()]] * 6 + [[]] * 2 + [[_box()]] * 1 + [[]] * 3
    result = score(predictions, truths)
    assert result.recall == pytest.approx(0.75)
    assert not result.beats_the_teacher()
    assert not result.passes


def test_scoring_needs_both_kinds_of_frame_to_mean_anything():
    """all-positive frames cannot show a false-positive rate, so such a run cannot pass"""
    result = score([[_box()]], [[_box()]])
    assert result.recall == 1.0
    assert not result.passes, "no empty frames means the false-positive rate is untested"


def test_mismatched_lengths_are_refused():
    with pytest.raises(ScoringError):
        score([[]], [[], []])


def test_peaks_become_boxes_of_the_known_plate_size():
    boxes = boxes_from_peaks([Peak(x=900, y=500, score=0.9)])
    assert boxes[0].width == 132
    assert boxes[0].left == 900 - 66
    assert boxes[0].origin == "model"


def test_the_report_says_pass_or_fail_in_words():
    result = score([[_box()], []], [[_box()], []])
    assert any("PASS" in line for line in result.lines())


def test_iou_of_two_squares_20px_apart():
    a = PlateBox(left=0, top=0, width=100, height=100)
    b = PlateBox(left=20, top=0, width=100, height=100)
    assert a.iou(b) == pytest.approx(0.6667, abs=1e-3)


def test_iou_matched_boxes_count_as_a_hit():
    a = PlateBox(left=0, top=0, width=100, height=100)
    b = PlateBox(left=20, top=0, width=100, height=100)
    result = score([[a]], [[b]], match=iou_match(0.5))
    assert result.hits == 1


def test_disjoint_boxes_have_zero_iou():
    a = PlateBox(left=0, top=0, width=10, height=10)
    b = PlateBox(left=100, top=100, width=10, height=10)
    assert a.iou(b) == 0.0


def test_identical_boxes_have_iou_one():
    a = PlateBox(left=5, top=5, width=40, height=40)
    assert a.iou(a) == 1.0


def test_a_zero_area_box_has_zero_iou():
    a = PlateBox(left=0, top=0, width=0, height=10)
    b = PlateBox(left=0, top=0, width=10, height=10)
    assert a.iou(b) == 0.0


def test_a_box_inside_one_four_times_its_area():
    outer = PlateBox(left=0, top=0, width=20, height=20)
    inner = PlateBox(left=0, top=0, width=10, height=10)
    assert outer.iou(inner) == pytest.approx(0.25)
    assert not score([[inner]], [[outer]], match=iou_match(0.5)).hits
    assert score([[inner]], [[outer]], match=iou_match(0.2)).hits == 1


def test_the_thin_bar_near_miss_is_a_hit_by_default_and_a_miss_by_iou():
    """this is the case that keeps overlaps the default - see box.Box.overlaps"""
    predicted, truth = _box(left=810, top=406), _box(left=800, top=400)
    assert score([[predicted]], [[truth]]).hits == 1
    assert score([[predicted]], [[truth]], match=iou_match(0.5)).hits == 0


def test_candidates_keep_their_own_size():
    boxes = boxes_from_candidates(
        [
            {"left": 10, "top": 20, "width": 30, "height": 40},
            {"left": 50, "top": 60, "width": 5, "height": 5},
        ]
    )
    assert boxes[0].width == 30 and boxes[0].height == 40
    assert boxes[1].width == 5 and boxes[1].height == 5
    assert boxes[0].origin == "model"
