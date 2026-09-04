"""scoring a detector against reviewed labels, and the pass marks the plan committed to."""

from __future__ import annotations

import pytest

from smolsmort.detect.box import Box as PlateBox
from smolsmort.detect.model import Peak
from smolsmort.detect.scoring import (
    MAX_FALSE_POSITIVE_RATE,
    TARGET_RECALL,
    ScoringError,
    boxes_from_peaks,
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
