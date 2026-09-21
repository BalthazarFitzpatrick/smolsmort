"""the two things every window-sampling training loop in this package does the same way: pick
where a training window lands, and carve the ignore regions out of its mask.

detect (fixed-size heatmap) and boxes (size-aware) build different targets from a window, but
the policy for WHERE the window goes is one policy: an object roll, a hard-negative roll, else
anywhere - with the object allowed anywhere in the window rather than pinned near the middle.
the rng is consumed in exactly this order, so a seeded run is reproducible across both.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

import numpy as np

Point = tuple[float, float]


def window_origin(
    rng: random.Random,
    side: int,
    width: int,
    height: int,
    *,
    centres: Sequence[Point],
    negatives: Sequence[Point],
    jitter_fraction: float,
    object_share: float,
    negative_share: float,
) -> tuple[int, int]:
    """(left, top) of one training window inside a (height, width) input.

    `centres` and `negatives` are in INPUT px already. with probability `object_share` the
    window lands around a random object, jittered by `jitter_fraction` of the side so the object
    can sit anywhere in it; otherwise up to `negative_share` around a hard negative (side/6
    jitter, tight on purpose - a negative is a point the model fired on, not a region); else
    uniformly anywhere. clamped so the window stays inside the input.
    """
    roll = rng.random()
    if centres and roll < object_share:
        cx, cy = rng.choice(centres)
        jitter = side * jitter_fraction
        left = int(cx - side / 2 + rng.uniform(-jitter, jitter))
        top = int(cy - side / 2 + rng.uniform(-jitter, jitter))
    elif negatives and roll < negative_share:
        cx, cy = rng.choice(negatives)
        left = int(cx - side / 2 + rng.uniform(-side / 6, side / 6))
        top = int(cy - side / 2 + rng.uniform(-side / 6, side / 6))
    else:
        left, top = rng.randrange(0, max(1, width - side)), rng.randrange(0, max(1, height - side))
    return max(0, min(left, width - side)), max(0, min(top, height - side))


def carve_ignore(
    mask: np.ndarray,
    ignore: Sequence[tuple[float, float, float, float]],
    left: int,
    top: int,
    stride: int,
    to_input: Callable[[float], float],
) -> None:
    """zero the mask cells under each ignore box. `ignore` boxes are in frame px; `to_input`
    maps a frame coordinate to input px (a downscale or a scale factor, the caller's choice).
    the far edge rounds up so a box never loses its last partial cell"""
    cells_h, cells_w = mask.shape[-2:]
    for x0, y0, x1, y1 in ignore:
        a = int((to_input(x0) - left) / stride)
        b = int((to_input(y0) - top) / stride)
        c = math.ceil((to_input(x1) - left) / stride)
        d = math.ceil((to_input(y1) - top) / stride)
        a, b = max(0, a), max(0, b)
        c, d = min(cells_w, c), min(cells_h, d)
        if a < c and b < d:
            mask[b:d, a:c] = 0.0
