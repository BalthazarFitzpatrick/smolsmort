"""snap a roughly-right plate box onto the plate, keeping the crop size.

THE REVIEW ANSWER THIS EXISTS FOR is "yes, that IS a nameplate - but not there". A detector that
finds the right object a dozen pixels off is correct about the only thing a human can judge quickly,
and making them redraw the box to say so spends their attention on arithmetic rather than judgement.
So the reviewer says yes, and this puts the box where the plate actually is.

IT KEEPS THE CROP SIZE AND MOVES ONLY THE CENTRE. A training set wants uniform crops - a net fed
boxes of drifting size learns the size as a feature - so the width and height that come in are the
width and height that go out. What changes is where they sit.

IT DOES NOT USE THE RED FILL TEST. `_looks_like_a_health_bar` is red-only and a plate may be any
colour.

LOCATING IS NOT DETECTING, and conflating the two is what made the first attempt at this useless.
A detector sweeping a whole frame must decide plate-vs-terrain, so it thresholds. Here the reviewer
has ALREADY decided - they said yes by not marking the crop "not a class" - so the only question
left is where in the crop the plate sits. `confirmed_locator` therefore ranks and never rejects,
and always returns its best guess. Measured on the 113 hand-marked plates of
camera_calibration/2026-09-06 at 2560x1665, nudging each box up to 14 px and asking for it back:

    border_locator (thresholds)     7 of 60 located, median 20 px from the human's box
    confirmed_locator (ranks)      60 of 60 located, median  3 px, 48 of 60 inside 5 px

`border_locator` is kept for the whole-frame case, where rejecting IS the job. Its numbers below.

THE THRESHOLDING LOCATOR IS NOT GOOD ENOUGH ON A REAL CAPTURE, and this is measured rather than feared.
Against the 113 hand-marked plates of `camera_calibration/2026-09-06` at 2560x1665, nudging each
box by a known amount and asking the border-pair locator to put it back recovered **7 of 60**, at a
median 20 px from where the human had it. Sweeping every gap from 4 to 33 over 40 of those frames
found ~25 pairs in total, their widths clustering at 50-69 px against a plate the human drew at
169 px wide. That is the same failure task 90 recorded from the other direction - "43 candidates
over 26 frames and EVERY ONE is ground texture" - and it is why calibration locates plates with
the trained cnn (`vision/plate_track.locate_plates`) rather than this.

So `locator` is an argument. The geometry here - search a window around where the detector pointed,
prefer the candidate nearest it, keep the crop size, clamp at the frame edge - is the part worth
reusing; which locator fills it in is the caller's decision, and the border one is a placeholder
that should be replaced by the model locator before this is trusted on real review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snapshot.tools.spike_nameplate_border import (
    CLOSE_GAP,
    GAP_MAX,
    GAP_MIN,
    X_TOLERANCE,
    _warm_runs,
)

__all__ = ["Recentred", "recentre", "confirmed_locator", "border_locator", "RecentreError"]

# how far the plate is allowed to be from where the detector put it, as a fraction of the crop.
# HALF, because beyond that the "plate" found is as likely to be a neighbouring one - and the
# reviewer said this box was nearly right, not that there was a plate somewhere on the screen
DEFAULT_SEARCH_FRACTION = 0.5

# the width band searched when the screen's own plate width is not known. deliberately wide: the
# bar shrinks with distance, and this is a local search around a box a human already vouched for,
# so a loose band costs a little time and a tight one costs the correction entirely
FALLBACK_WIDTH = (40, 460)

# how far either side of a measured plate width to look. same tolerance the detector uses
WIDTH_TOLERANCE = 0.14

# the gap band confirmed_locator searches when the screen's own is unknown. wide, because it is
# ranking rather than filtering - a wrong-sized pair simply scores lower than the right one
# the band of bar heights searched when the screen's own is unknown. The upper end was 40 and a
# plate on a 3420-wide capture stands up to 46px tall, so the largest real plates had no pair to
# match and the search settled for whatever else agreed.
GAP_SEARCH_MIN, GAP_SEARCH_MAX = 5, 64

# how far above and below a row to look when asking "is this a thin line or part of a block".
# 3 rows clears a 2px border without reaching into the other border of a short bar
LINE_REACH = 3

# a bar shorter than this is not a bar. the narrowest plate measured is 157px on a 2560
# capture and 46px on a 768 one, so this sits under the smallest real one by a margin
MIN_BAR_RUN = 24

# how bright both borders must be, relative to the bar's OWN typical brightness, for a column to
# count as inside it. Measured against the brightest column instead, the level badge - far warmer
# than the border it hangs off - set the bar of comparison, the border fell under 35% of it, and
# the run collapsed to the badge end. A high percentile is the bar's own level and ignores one
# bright blob; half of that keeps a border dimmed by downscaling.
#
# SWEPT, not chosen: percentile 75-100 against fraction 0.25-0.5 over 400 generated plates.
# 0.5 wins at every percentile, and 90 is where coverage stops improving faster than accuracy
# degrades - 389 of 400 located, horizontal p90 2.1px, vertical p90 1.0px, 358 inside 3px.
# 100 (the plain maximum) locates only 374 because one bright badge raises the bar for the
# whole row; 75 locates 395 with a 4.6px horizontal tail.
BAR_LEVEL_PERCENTILE = 90
BAR_EDGE_FRACTION = 0.5


class RecentreError(Exception):
    pass


@dataclass(frozen=True)
class Recentred:
    """the corrected box, and how far it had to move to get there"""

    left: int
    top: int
    width: int
    height: int
    shift_x: int
    shift_y: int
    # the border pair this locked onto, for a reviewer who wants to see what it found
    bar_left: int
    bar_width: int
    bar_top: int
    bar_bottom: int

    @property
    def moved(self) -> int:
        """chebyshev distance, because a reviewer thinks in "a few pixels off", not euclidean"""
        return max(abs(self.shift_x), abs(self.shift_y))


def _pairs_in(window: np.ndarray, lo: int, hi: int, gaps: range) -> list[tuple[int, int, int, int]]:
    """every (left, width, top, bottom) border pair in the window, unfiltered by plausibility"""
    rows: dict[int, list[tuple[int, int]]] = {}
    for y in range(window.shape[0]):
        runs = _warm_runs(window[y], lo, hi)
        if runs:
            rows[y] = runs

    out = []
    for top in sorted(rows):
        for left, width in rows[top]:
            for gap in gaps:
                below = rows.get(top + gap)
                if not below:
                    continue
                # the lower border must sit under the upper one AND be about as long - a run that
                # starts in the right place but runs twice as far is terrain, not the other border
                if any(
                    abs(x - left) <= X_TOLERANCE and abs(w - width) <= max(CLOSE_GAP, width * 0.25)
                    for x, w in below
                ):
                    out.append((left, width, top, top + gap))
                    break
    return out


def confirmed_locator(
    window: np.ndarray, plate_width: int | None, border_gap: int | None
) -> list[tuple[int, int, int, int]]:
    """THE CROP ALREADY HOLDS A PLATE, so this ranks and never rejects.

    That is the whole difference from the border locator, and it is why this one works. A detector
    sweeping a frame must decide plate-vs-terrain, so it thresholds, and a threshold tuned to keep
    terrain out throws real plates away - 7 of 60 recovered. Here the human has already said this
    is a plate by not marking it "not a class"; the only question left is WHERE in the crop it
    sits. So the score only has to rank rows against each other, and the best pair always wins.

    Measured on the 113 hand-marked plates of camera_calibration/2026-09-06: 60 of 60 located,
    median 3 px from the human's own box after nudging each one up to 14 px away, 48 of 60 inside
    5 px.
    """
    w = window.astype(int)
    r, g, b = w[:, :, 0], w[:, :, 1], w[:, :, 2]
    # the two borders are the warmest horizontal structure in the crop. no absolute level decides
    # anything - this is a ranking, so a dim plate scores low everywhere and still ranks correctly
    warm = np.clip(r - b, 0, None) + np.clip(r - g, 0, None)

    # A BORDER IS A THIN LINE; A FILL IS A BLOCK, and ranking on warmth alone cannot tell them
    # apart. Gold scores about 156 on this measure - a red or orange bar fill scores 470, so on any
    # plate that is not green the locator locked onto the FILL and reported its rows as the border.
    # Every marked plate in this repo is a green friendly one, where the fill is cold, which is
    # exactly why the corpus never showed it. Measured on generated plates of random fill colour:
    # 20.5px median horizontal error before this, and the vertical error was 7.5px.
    #
    # Subtracting the neighbourhood a few rows away leaves a thin line standing and flattens a
    # block to nothing, because a block's neighbours look like itself.
    rows = warm.sum(axis=1).astype(float)
    if rows.max() <= 0:
        return []

    height = len(rows)
    gaps = (
        range(GAP_SEARCH_MIN, min(GAP_SEARCH_MAX, height))
        if border_gap is None
        else range(max(2, border_gap - 2), border_gap + 3)
    )

    # THE BAR IS A LONG UNBROKEN AGREEMENT, and that is what to rank on. Two rows being warm in
    # the same columns SOMEWHERE is satisfied by the name text over a bright fill; being warm in
    # the same columns for a hundred consecutive pixels is not satisfied by anything else in a
    # plate. Ranking on the run's length also means the pair that wins is the pair whose extent is
    # then measured - the score and the answer stop being two different questions.
    #
    # Measured against nothing but the interior: a border's inner neighbour is the fill, whose
    # colour is arbitrary, so only what lies OUTSIDE the pair enters the score.
    def longest_run(values: np.ndarray, floor: float) -> tuple[int, int]:
        lit = values >= floor
        if not lit.any():
            return 0, 0
        edges = np.flatnonzero(np.diff(np.concatenate(([0], lit.view(np.int8), [0]))))
        starts, ends = edges[::2], edges[1::2]
        widest = int(np.argmax(ends - starts))
        return int(starts[widest]), int(ends[widest])

    best, score, best_span = None, -np.inf, (0, 0)
    for gap in gaps:
        if gap >= height:
            break
        agree = np.minimum(warm[:-gap], warm[gap:])
        for top in range(agree.shape[0]):
            row = agree[top]
            peak = float(np.percentile(row, BAR_LEVEL_PERCENTILE))
            if peak <= 0:
                continue
            lo, hi = longest_run(row, peak * BAR_EDGE_FRACTION)
            run = hi - lo
            if run < MIN_BAR_RUN:
                continue
            outside = 0.0
            if top - LINE_REACH >= 0:
                outside += rows[top - LINE_REACH]
            if top + gap + LINE_REACH < height:
                outside += rows[top + gap + LINE_REACH]
            # the run's length carries the decision; its strength only breaks ties between runs
            value = run * float(row[lo:hi].mean()) - 0.5 * outside
            if value > score:
                score, best, best_span = value, (top, top + gap), (lo, hi)
    if best is None:
        return []

    top, bottom = best
    # the bar runs as far as both borders are warm together, which is what excludes the level badge
    # hanging off one end and the name text floating above
    # MINIMUM, not sum: "both borders warm together" is what excludes a feature bright on one row
    # alone - the level badge hanging off the top border passed a sum test on its brightness
    left, right = best_span
    return [(left, right - left, int(top), int(bottom))]


def border_locator(
    window: np.ndarray, plate_width: int | None, border_gap: int | None
) -> list[tuple[int, int, int, int]]:
    """the placeholder locator: gold border pairs. see the module docstring for its measured miss
    rate - it is here so the geometry can be tested, not because it works on a real capture"""
    if plate_width is not None:
        lo = int(plate_width * (1 - WIDTH_TOLERANCE))
        hi = int(plate_width * (1 + WIDTH_TOLERANCE))
    else:
        lo, hi = FALLBACK_WIDTH
    gaps = (
        range(GAP_MIN, GAP_MAX + 1) if border_gap is None else range(border_gap - 1, border_gap + 2)
    )
    return _pairs_in(window, lo, hi, gaps)


def recentre(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    plate_width: int | None = None,
    border_gap: int | None = None,
    search_fraction: float = DEFAULT_SEARCH_FRACTION,
    locator=confirmed_locator,
) -> Recentred | None:
    """move `box` so the plate inside it sits in the middle. None when no plate is found.

    `box` is (left, top, width, height) in the frame's own pixels. `plate_width` and `border_gap`
    are this screen's measured geometry from the profile's [nameplates] block when it has been
    measured; without them the search runs on a wide band, which is slower and looser but works.

    Returning None rather than raising is deliberate: "I could not find it" is an ordinary outcome
    for a box the detector got badly wrong, and the caller's answer is to leave the box alone and
    let the human place it.
    """
    left, top, width, height = box
    if width <= 0 or height <= 0:
        raise RecentreError(f"a box needs a positive size, got {width}x{height}")
    frame_h, frame_w = image.shape[:2]

    margin_x = max(1, int(width * search_fraction))
    margin_y = max(1, int(height * search_fraction))
    x0, y0 = max(0, left - margin_x), max(0, top - margin_y)
    x1 = min(frame_w, left + width + margin_x)
    y1 = min(frame_h, top + height + margin_y)
    window = image[y0:y1, x0:x1]
    if window.size == 0:
        return None

    pairs = locator(window, plate_width, border_gap)
    if not pairs:
        return None

    # NEAREST TO WHERE THE DETECTOR POINTED, not the biggest or the first. the reviewer vouched for
    # THIS box, so among several plates in the window the intended one is the closest
    want_x = left + width / 2 - x0
    want_y = top + height / 2 - y0

    def distance(pair: tuple[int, int, int, int]) -> float:
        bl, bw, bt, bb = pair
        return abs(bl + bw / 2 - want_x) + abs((bt + bb) / 2 - want_y)

    bar_left, bar_width, bar_top, bar_bottom = min(pairs, key=distance)

    centre_x = x0 + bar_left + bar_width / 2
    centre_y = y0 + (bar_top + bar_bottom) / 2
    new_left = int(round(centre_x - width / 2))
    new_top = int(round(centre_y - height / 2))
    # a crop that runs off the frame is not a crop; clamping keeps the size and gives up the centring
    new_left = max(0, min(new_left, frame_w - width))
    new_top = max(0, min(new_top, frame_h - height))

    return Recentred(
        left=new_left,
        top=new_top,
        width=width,
        height=height,
        shift_x=new_left - left,
        shift_y=new_top - top,
        bar_left=x0 + bar_left,
        bar_width=bar_width,
        bar_top=y0 + bar_top,
        bar_bottom=y0 + bar_bottom,
    )
