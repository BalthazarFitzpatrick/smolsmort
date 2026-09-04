"""scoring a detector against hand-reviewed labels.

THE CRITERION THIS ENCODES, written down before training rather than after: on held-out frames from
a zone NOT in the training split, recall >= 90% of frames that contain an object, with a false-positive
rate <= 5% of frames that contain none. Anything that does not clearly beat the border-pair teacher's
measured ~75% / ~75% on the same frames means the learned route is abandoned rather than tuned - so
the teacher's own output is scored through this exact function too, and is the number to beat.

MATCHING IS PER-FRAME AND POSITIONAL, not IoU. detect/box.Box.overlaps already carries the
right test, and it is right for a reason worth not rediscovering: the reference object is long and thin - about 132x12, so
eight pixels of vertical offset drops IoU to 0.12 - area overlap is hopeless on a shape this thin.
What identifies it is that the two boxes lie along the same row and cover the same span.

AN EMPTY FRAME IS SCORED, NOT SKIPPED. it is the only place a false positive can be counted
cleanly, and it is where all three measured false-positive classes live - the chat frame, brown dirt
and orange foliage. A scorer that only looked at frames containing objects would report the detector
as perfect while it lit up on every tree.
"""

from __future__ import annotations

from dataclasses import dataclass

from smolsmort.detect.box import Box

# the plan's pass marks, kept here so the numbers and the code that checks them cannot drift
TARGET_RECALL = 0.90
MAX_FALSE_POSITIVE_RATE = 0.05

# the teacher's measured performance, which any learned model has to beat to be worth having
TEACHER_RECALL = 0.75
TEACHER_PRECISION = 0.75


class ScoringError(Exception):
    pass


@dataclass(frozen=True)
class Score:
    frames_with_object: int
    frames_without_object: int
    hits: int  # frames with an object where at least one prediction matched a label
    misses: int
    false_positive_frames: int  # frames with NO object that got a prediction anyway
    spurious: int  # predictions that matched no label, over all frames

    @property
    def recall(self) -> float:
        return self.hits / self.frames_with_object if self.frames_with_object else 0.0

    @property
    def false_positive_rate(self) -> float:
        if not self.frames_without_object:
            return 0.0
        return self.false_positive_frames / self.frames_without_object

    @property
    def passes(self) -> bool:
        return (
            self.recall >= TARGET_RECALL
            and self.false_positive_rate <= MAX_FALSE_POSITIVE_RATE
            and self.frames_with_object > 0
            and self.frames_without_object > 0
        )

    def beats_the_teacher(self) -> bool:
        """the bar that decides whether a learned detector was worth building at all"""
        return self.recall > TEACHER_RECALL and self.false_positive_rate < (1 - TEACHER_PRECISION)

    def lines(self) -> list[str]:
        return [
            f"frames          {self.frames_with_object} with an object, "
            f"{self.frames_without_object} without",
            f"recall          {self.recall:.0%}  ({self.hits} hit, {self.misses} missed)"
            f"   target {TARGET_RECALL:.0%}",
            f"false positives {self.false_positive_rate:.0%} of empty frames "
            f"({self.false_positive_frames})   limit {MAX_FALSE_POSITIVE_RATE:.0%}",
            f"spurious boxes  {self.spurious} across all frames",
            f"verdict         {'PASS' if self.passes else 'FAIL'}"
            f"   {'beats' if self.beats_the_teacher() else 'does NOT beat'} the teacher",
        ]


def _matches(predicted: Box, truth: list[Box]) -> bool:
    return any(predicted.overlaps(box) for box in truth)


def score(predictions: list[list[Box]], truths: list[list[Box]]) -> Score:
    """one entry per frame in each list, in the same order"""
    if len(predictions) != len(truths):
        raise ScoringError(
            f"{len(predictions)} predicted frames against {len(truths)} labelled ones - "
            "they have to line up frame for frame"
        )
    if not truths:
        raise ScoringError("nothing to score")

    with_object = without_object = hits = misses = fp_frames = spurious = 0
    for predicted, truth in zip(predictions, truths, strict=True):
        unmatched = sum(1 for box in predicted if not _matches(box, truth))
        spurious += unmatched
        if truth:
            with_object += 1
            if any(_matches(box, truth) for box in predicted):
                hits += 1
            else:
                misses += 1
        else:
            without_object += 1
            if predicted:
                fp_frames += 1
    return Score(
        frames_with_object=with_object,
        frames_without_object=without_object,
        hits=hits,
        misses=misses,
        false_positive_frames=fp_frames,
        spurious=spurious,
    )


def boxes_from_peaks(peaks, width: int = 132, height: int = 12) -> list[Box]:
    """model peaks are a centre in capture pixels; the object's size is known, so the box follows"""
    return [
        Box(
            left=int(peak.x - width // 2),
            top=int(peak.y - height // 2),
            width=width,
            height=height,
            origin="model",
        )
        for peak in peaks
    ]
