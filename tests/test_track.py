"""tracking the one plate through a recording, and refusing to answer when it cannot be found.

the model itself is not exercised here - that needs torch and a trained checkpoint. What IS tested
is everything around it, because every failure this module exists to prevent was a SELECTION
failure rather than a detection one: the right peak was on offer and the wrong one was taken.
"""

from __future__ import annotations

import math

from smolsmort.detect.track import (
    CHROME_FACING_SPREAD_DEG,
    DESPIKE_TOLERANCE_PX,
    LOST_AFTER_FRAMES,
    MAX_STEP_PX,
    chrome_cells,
    despike,
    pick,
)


def peak(x, y, score=0.5, w=229, h=38):
    return {"left": x - w / 2, "top": y - h / 2, "width": w, "height": h, "score": score}


def centre(x, y, score=0.5):
    return (float(x), float(y), score)


# ---------------------------------------------------------------- choosing a peak


def test_with_no_history_the_strongest_peak_wins():
    """the only evidence there is, at the start of a recording and after the tracker is lost"""
    chosen = pick([centre(100, 100, 0.3), centre(900, 900, 0.7)], None, MAX_STEP_PX)
    assert chosen[:2] == (900.0, 900.0)


def test_a_peak_too_far_from_the_last_one_is_not_the_same_plate():
    """A PLATE CANNOT CROSS THE SCREEN IN 0.2s. The corner decoy sat ~1600px from the plate and
    answered a steady 0.25 while the plate ran 0.35-0.51, so on any frame the plate dipped the
    strongest peak was the decoy - 12 jumps of ~1600px over one walk.
    """
    previous = (1280.0, 700.0)
    assert pick([centre(2400, 1400, 0.9)], previous, MAX_STEP_PX) is None


def test_among_plausible_peaks_the_strongest_wins_not_the_nearest():
    """a real plate that dipped in score is still the plate; nearest alone chases whichever noise
    happens to sit closest to the last position
    """
    previous = (1280.0, 700.0)
    chosen = pick([centre(1290, 705, 0.3), centre(1400, 760, 0.8)], previous, MAX_STEP_PX)
    assert chosen[:2] == (1400.0, 760.0)


def test_nothing_on_offer_is_answered_with_nothing():
    assert pick([], (1280.0, 700.0), MAX_STEP_PX) is None
    assert pick([], None, MAX_STEP_PX) is None


# ---------------------------------------------------------------- chrome, by facing spread


def test_a_peak_that_holds_one_spot_through_a_rotation_is_chrome():
    """MEASURED: a target frame answered at one position in 226 of 234 frames spanning 355 degrees,
    while real plates held a position over 5-10 degrees. Balthazar Fitzpatrick spotted it - "I think its the model
    finding the target frame in the pictures".
    """
    per_frame, facings = {}, {}
    for i in range(60):
        name = f"{i:07d}.jpg"
        angle = 2 * math.pi * i / 60
        # a fixed hud element, and a plate that moves with the camera
        per_frame[name] = [peak(2400, 50), peak(1280 + 400 * math.cos(angle), 700)]
        facings[name] = angle

    chrome = chrome_cells(per_frame, facings)
    assert chrome, "the fixed element was not detected"
    assert any(abs(cx * 50 - 2400) < 60 for cx, _ in chrome)
    # and the moving plate is never called chrome, wherever it happens to sit
    assert not any(abs(cy * 50 - 700) < 30 for _, cy in chrome)


def test_a_position_seen_only_briefly_is_not_chrome():
    """a plate parked while the player stands still must not be mistaken for a hud element - the
    discriminator is the FACING spread, not how many frames a position appears in
    """
    per_frame = {f"{i:07d}.jpg": [peak(1280, 700)] for i in range(40)}
    facings = dict.fromkeys(per_frame, 1.5)  # standing still, one heading
    assert not chrome_cells(per_frame, facings)


def test_chrome_needs_both_enough_frames_and_enough_spread():
    per_frame = {f"{i:07d}.jpg": [peak(2400, 50)] for i in range(4)}
    facings = {n: 2 * math.pi * i / 4 for i, n in enumerate(per_frame)}
    assert not chrome_cells(per_frame, facings), "four frames is not evidence of anything"
    assert CHROME_FACING_SPREAD_DEG < 180, "the threshold must admit a half turn"


# ---------------------------------------------------------------- despiking a smooth-but-wrong run


def test_a_frame_disagreeing_with_its_neighbours_is_dropped():
    """THE TRACKER ALONE IS NOT ENOUGH: it only compares a frame to the one before, so a re-acquire
    onto a decoy produces a smooth, wrong run that the step check cannot see.
    """
    paths = [f"{i:07d}.jpg" for i in range(20)]
    found = {p: (1280.0 + i * 4, 700.0) for i, p in enumerate(paths)}
    found[paths[10]] = (1280.0 + DESPIKE_TOLERANCE_PX * 3, 700.0)

    kept = despike(found, paths)
    assert paths[10] not in kept
    assert len(kept) == len(paths) - 1


def test_a_steady_track_survives_despiking_untouched():
    paths = [f"{i:07d}.jpg" for i in range(20)]
    found = {p: (1280.0 + i * 8, 700.0 + i * 2) for i, p in enumerate(paths)}
    assert despike(found, paths) == found


def test_too_few_frames_to_have_a_neighbourhood_are_left_alone():
    paths = [f"{i:07d}.jpg" for i in range(4)]
    found = {p: (float(i * 900), 700.0) for i, p in enumerate(paths)}
    assert despike(found, paths) == found


def test_the_lost_threshold_allows_a_brief_occlusion():
    """a plate hidden behind something for a frame or two must not hand the track to a decoy, and
    a plate genuinely gone must not lock the tracker out for the rest of the recording
    """
    assert 1 < LOST_AFTER_FRAMES < 30


# ---------------------------------------------------------------- offered to the calibration tools
