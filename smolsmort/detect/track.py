"""following ONE object across a recording, given per-frame detections.

WHY THIS IS SEPARATE FROM THE THING THAT CALLS IT. A detector answers per frame and has no memory;
turning a stack of independent answers into a single trajectory is a different job with its own
failure modes, and none of them are about what the object is. A consumer supplies the detections
and the tuning; everything here is arithmetic over positions.

THE THREE FAILURE MODES, in the order they were found, because each fix exists because the previous
one was not enough:

  1. NO HISTORY. At the start of a recording, and after being lost, the strongest peak is the only
     evidence there is. `pick` falls back to it.
  2. A DECOY NEARBY. Preferring the NEAREST peak chases whichever noise sits closest, so `pick`
     takes the strongest among the plausible ones instead - an object that dipped in score is still
     the object.
  3. A SMOOTH, WRONG RUN. The step check only ever compares a frame to the one before, so a
     re-acquire onto a decoy is then tracked consistently and looks perfectly well-behaved.
     `despike` is the only test that catches it, comparing each frame to a MEDIAN over its own
     neighbourhood - median rather than mean precisely because it is unmoved by the outliers being
     looked for.

`chrome_cells` is the fourth, and it needs something the detections alone cannot provide: which way
the observer was facing. A real object in the world MUST move on screen when the observer turns, so
a detection answering at one screen position across most of a rotation is part of the interface, not
part of the world. That holds for any moving-camera capture; only the facing signal is the
consumer's to supply.
"""

from __future__ import annotations

import math

# how far an object may move between frames and still be the same object. generous on purpose: at a
# low sample rate a near object crosses a lot of screen, and being too strict drops exactly the
# close-range samples that matter most
MAX_STEP_PX = 600.0

# how many frames of nothing before the tracker admits it has lost the object and re-acquires from
# scratch. NO ANSWER BEATS A WRONG ONE - falling back to the strongest peak the moment nothing sits
# near the last position is what let a corner decoy in: it became `previous`, and every following
# frame tracked the decoy instead
LOST_AFTER_FRAMES = 5

# the neighbourhood despike compares against, and how far a frame may sit from its own median
DESPIKE_WINDOW = 9
DESPIKE_TOLERANCE_PX = 250.0

# a detection answering across MORE than this much of a full turn, from one screen cell, is
# interface rather than world. needs at least CHROME_MIN_FRAMES of evidence before ruling
CHROME_FACING_SPREAD_DEG = 90.0
CHROME_MIN_FRAMES = 12
CHROME_CELL_PX = 50


def pick(centres, previous, max_step: float = MAX_STEP_PX):
    """the detection for this frame, or None when nothing here can be what was being tracked.

    `centres` is [(x, y, score)]. With no history the strongest is the only evidence there is.
    """
    if not centres:
        return None
    if previous is None:
        return max(centres, key=lambda c: c[2])
    near = [c for c in centres if math.hypot(c[0] - previous[0], c[1] - previous[1]) <= max_step]
    if not near:
        return None
    # among the plausible ones prefer the STRONGEST, not the nearest - see failure mode 2 above
    return max(near, key=lambda c: c[2])


def despike(
    found: dict[str, tuple[float, float]],
    paths: list[str],
    window: int = DESPIKE_WINDOW,
    tolerance: float = DESPIKE_TOLERANCE_PX,
) -> dict:
    """drop frames whose position disagrees with their own neighbourhood - failure mode 3."""
    ordered = [(p, found[p]) for p in paths if p in found]
    if len(ordered) < window:
        return found
    kept = {}
    half = window // 2
    for i, (path, (x, y)) in enumerate(ordered):
        lo, hi = max(0, i - half), min(len(ordered), i + half + 1)
        neighbourhood = [c for j, (_, c) in enumerate(ordered[lo:hi], start=lo) if j != i]
        mx = sorted(c[0] for c in neighbourhood)[len(neighbourhood) // 2]
        my = sorted(c[1] for c in neighbourhood)[len(neighbourhood) // 2]
        if math.hypot(x - mx, y - my) <= tolerance:
            kept[path] = (x, y)
    return kept


def _cell(peak, size: int = CHROME_CELL_PX) -> tuple[int, int]:
    """a detection's position rounded to a cell, so the same spot across frames lands in one bucket"""
    cx = peak["left"] + peak["width"] / 2.0
    cy = peak["top"] + peak["height"] / 2.0
    return (round(cx / size), round(cy / size))


def chrome_cells(
    per_frame: dict,
    facings: dict[str, float],
    spread_degrees: float = CHROME_FACING_SPREAD_DEG,
    min_frames: int = CHROME_MIN_FRAMES,
) -> set:
    """screen cells that stay put while the observer turns - interface, not world.

    THE DISCRIMINATOR IS FACING SPREAD, NOT POSITION. Position rules were tried and are wrong: an
    excluded top strip deletes exactly the distant samples a range fit depends on, because a far
    object legitimately sits high on screen. `facings` maps a frame path to radians.
    """
    seen: dict[tuple[int, int], list[float]] = {}
    for path, peaks in per_frame.items():
        facing = facings.get(path)
        if facing is None:
            continue
        for peak in peaks:
            seen.setdefault(_cell(peak), []).append(math.degrees(facing) % 360.0)

    chrome = set()
    for cell, angles in seen.items():
        if len(angles) < min_frames:
            continue
        # the widest gap subtracted from the circle, so a spread either side of north is not read
        # as a full turn by a naive max-minus-min
        ordered = sorted(angles)
        gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
        gaps.append(ordered[0] + 360.0 - ordered[-1])
        if 360.0 - max(gaps) > spread_degrees:
            chrome.add(cell)
    return chrome


def track(
    per_frame: dict,
    paths: list[str],
    max_step: float = MAX_STEP_PX,
    lost_after: int = LOST_AFTER_FRAMES,
) -> dict[str, tuple[float, float]]:
    """{frame path: (x, y)} for every frame one object could be followed into.

    `paths` MUST be in recording order - each frame is read against the previous one, so shuffling
    them silently turns tracking back into independent peak picking.
    """
    found: dict[str, tuple[float, float]] = {}
    previous: tuple[float, float] | None = None
    missed = 0
    for path in paths:
        peaks = per_frame.get(path) or []
        centres = [
            (p["left"] + p["width"] / 2.0, p["top"] + p["height"] / 2.0, p["score"]) for p in peaks
        ]
        chosen = pick(centres, previous, max_step)
        if chosen is None:
            missed += 1
            if missed > lost_after:
                previous = None  # genuinely lost - let the next frame re-acquire from scratch
            continue
        missed = 0
        found[path] = (chosen[0], chosen[1])
        previous = (chosen[0], chosen[1])
    return despike(found, paths)
