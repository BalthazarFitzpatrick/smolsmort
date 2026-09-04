"""a small heatmap cnn that finds objects of a FIXED KNOWN SIZE.

THIS IS NOT GENERAL OBJECT DETECTION AND MUST NOT BECOME IT. The one assumption this model is built
on is that the thing it looks for does not change size - so there is nothing to regress a width or
a height for. The only unknown is WHERE, which makes the natural output a per-pixel "is an object
centred here" heatmap and the natural model a handful of convolutions: ~50k parameters rather than
the millions a box-regression detector carries around to solve a problem this one does not have.

That assumption is a CONSTRAINT ON WHAT THIS CAN DETECT, not a fact about the world. A consumer
whose objects vary in size needs a different model, and should be told so rather than discovering
it in the loss curve. (Provenance: it was written for a 132px border over ~12px of
bar on a 2560px capture, measured off real frames.)

WHY A MODEL AT ALL, given a classical detector already finds these by their border pair: that
teacher measures ~75% precision and ~75% recall on real frames. Near, large instances are trivial
for it; FADED ONES ARE NOT, and loosening its thresholds to catch them starts admitting texture (a
patch of bear fur and an orange aoe decal both passed at one point). The faint, distant ones are
exactly the ones that matter downstream, so the squeeze lands on the case that is cared about most.
The classical detector stays as the auto-labelling TEACHER for the easy majority; this learns the
rest from human corrections.

RESOLUTION. The net runs on a 4x-downscaled frame - where the 132px reference object is 33px wide,
still several cells across at stride 4, and a sixteenth of the pixels to convolve. The heatmap is
therefore at stride 4 of the input, i.e. 1/16 of the capture, and peak positions are scaled back up
on decode.

WHAT IS NOT DECIDED HERE: how many channels a consumer asks for. `build_model(classes=...)` takes
that as a parameter, and the caller decides what a channel means - see the consumer for why some
distinctions are held back until data exists to support them. Adding a channel before there are
examples of it is inventing a label, whatever the domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STRIDE = 4  # heatmap cell : input pixel
DOWNSCALE = 4  # capture pixel : input pixel
PEAK_MIN_SCORE = 0.35
PEAK_MIN_SEPARATION = 3  # heatmap cells; two objects never overlap this closely


class ModelError(Exception):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a declared dependency
        raise ModelError("torch is not installed; run uv add torch") from exc
    return torch


def build_model(channels: int = 24, classes: int = 1):
    """a small fully-convolutional net: rgb in, one heatmap channel PER CLASS out.

    fully convolutional on purpose - it is trained on crops and run on whole frames, and anything
    with a fixed-size dense layer would forbid that.

    ONE CHANNEL PER CLASS, which is the whole reason this can replace a template matcher. The
    object is a fixed known size, so there is nothing to regress - the only unknowns are WHERE and
    WHICH, and a per-class heatmap answers both at once. Balthazar Fitzpatrick, 2026-09-01: "it will be able to
    draw a heatmap per 10 categories via 10 output channels, and that's all the x and y you need".
    classes=1 stays the default so every existing caller and checkpoint keeps working.
    """
    torch = _torch()
    nn = torch.nn

    def block(in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(3, channels, stride=2),  # /2
        block(channels, channels),
        block(channels, channels * 2, stride=2),  # /4 = STRIDE
        block(channels * 2, channels * 2),
        # a wide horizontal kernel, because the thing being found is a long thin horizontal bar and
        # a stack of 3x3s would need to be much deeper to see across one
        nn.Conv2d(channels * 2, channels * 2, (3, 9), padding=(1, 4)),
        nn.BatchNorm2d(channels * 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels * 2, classes, 1),
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass(frozen=True)
class Peak:
    """a detection in CAPTURE pixel coordinates, not heatmap or input coordinates"""

    x: int
    y: int
    score: float


def gaussian_target(
    shape: tuple[int, int], centres: list[tuple[float, float]], sigma: float = 1.5
) -> np.ndarray:
    """a heatmap target with a soft blob at each object centre.

    soft rather than a single hot cell because a one-cell target makes almost every pixel negative
    and the net learns to answer zero everywhere; the blob also stops a one-cell localisation error
    being punished as hard as a miss.
    """
    height, width = shape
    target = np.zeros((height, width), dtype=np.float32)
    radius = int(math.ceil(3 * sigma))
    for cx, cy in centres:
        left, right = max(0, int(cx) - radius), min(width, int(cx) + radius + 1)
        top, bottom = max(0, int(cy) - radius), min(height, int(cy) + radius + 1)
        if left >= right or top >= bottom:
            continue
        ys = np.arange(top, bottom, dtype=np.float32)[:, None]
        xs = np.arange(left, right, dtype=np.float32)[None, :]
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)))
        target[top:bottom, left:right] = np.maximum(target[top:bottom, left:right], blob)
    return target


def class_target(
    shape: tuple[int, int],
    centres: list[tuple[float, float]],
    labels: list[int],
    classes: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """(classes, h, w) - the same gaussian blob, drawn into the channel its label names.

    an object of one class is not a negative example for the others: the channels are trained
    independently, so an object simply leaves every other channel empty where it stands. that is
    what makes adding a class cheap, and what keeps a rare class from being drowned out by a common
    one - which a single softmax over ten classes would not.
    """
    stacked = np.zeros((classes, *shape), dtype=np.float32)
    for label in set(labels):
        wanted = [c for c, lab in zip(centres, labels, strict=True) if lab == label]
        if 0 <= label < classes:
            stacked[label] = gaussian_target(shape, wanted, sigma)
    return stacked


def focal_loss(prediction, target, alpha: float = 2.0, beta: float = 4.0):
    """centrenet-style penalty-reduced focal loss on a gaussian heatmap.

    plain BCE fails here for a structural reason: a frame has one or two objects against thousands of
    background cells, so predicting zero everywhere already scores well. this weights the rare
    positives up and discounts negatives near a peak, which are near-misses rather than errors.
    """
    torch = _torch()
    prediction = torch.sigmoid(prediction).clamp(1e-4, 1 - 1e-4)
    positive = target.ge(0.99).float()
    negative = 1.0 - positive
    positive_loss = -((1 - prediction) ** alpha) * torch.log(prediction) * positive
    negative_loss = (
        -((1 - target) ** beta) * (prediction**alpha) * torch.log(1 - prediction) * negative
    )
    count = positive.sum()
    total = positive_loss.sum() + negative_loss.sum()
    return total / count if count > 0 else negative_loss.sum()


def decode_peaks(
    heatmap: np.ndarray,
    min_score: float = PEAK_MIN_SCORE,
    min_separation: int = PEAK_MIN_SEPARATION,
    limit: int | None = None,
) -> list[Peak]:
    """local maxima above a threshold, returned in capture-pixel coordinates, strongest first.

    `limit` STOPS EARLY, and callers that only want the strongest few should always pass it.
    suppression compares each candidate against every peak already kept, so the cost grows with
    the number KEPT - and at a low floor an undertrained model answers warm nearly everywhere.
    MEASURED at floor 0.05 over 15 channels of one frame: 33,112 candidates, 44 SECONDS, of which
    the caller then kept twelve. Candidates are walked strongest first, so cutting off after
    `limit` peaks returns exactly the same top few for a bounded cost.
    """
    if heatmap.ndim != 2:
        raise ModelError(f"expected a 2d heatmap, got shape {heatmap.shape}")
    scale = STRIDE * DOWNSCALE
    candidates = [
        (float(heatmap[y, x]), x, y) for y, x in zip(*np.where(heatmap >= min_score), strict=True)
    ]
    peaks: list[Peak] = []
    for score, x, y in sorted(candidates, reverse=True):
        if any(
            abs(x - p.x // scale) < min_separation and abs(y - p.y // scale) < min_separation
            for p in peaks
        ):
            continue
        peaks.append(
            Peak(x=int(x * scale + scale // 2), y=int(y * scale + scale // 2), score=score)
        )
        if limit is not None and len(peaks) >= limit:
            break
    return peaks
