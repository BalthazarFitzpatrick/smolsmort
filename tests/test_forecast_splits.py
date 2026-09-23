"""cuts keep time order, leave the test rows time to resolve, and relabel as an old export would"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from smolsmort.forecast.splits import SplitError, labels_as_of, row_cuts, series_cuts

AS_OF = dt.date(2026, 9, 21)


def test_row_cuts_leave_the_test_rows_time_to_resolve():
    anchors = np.array(["2024-01-01", "2025-06-01", "2025-10-15", "2026-01-05", "2026-09-01"])
    cuts = row_cuts(anchors, AS_OF)
    assert cuts.test_end == AS_OF - dt.timedelta(weeks=26)
    assert cuts.val_start < cuts.test_start < cuts.test_end
    train, val, test = cuts.masks(anchors)
    assert train.tolist() == [True, True, False, False, False]
    assert val.tolist() == [False, False, True, False, False]
    assert test.tolist() == [False, False, False, True, False]
    assert not (train & val).any() and not (val & test).any()


def test_row_cuts_refuse_a_history_too_short_to_split():
    with pytest.raises(SplitError, match="shorter than"):
        row_cuts(np.array(["2026-06-01"]), AS_OF)


def test_labels_as_of_reopen_rows_that_resolved_after_the_cut():
    anchors = np.array(["2026-01-05", "2026-01-05", "2026-01-05", "2026-03-02"])
    lower = np.array([4.0, 10.0, 12.0, 3.0])
    upper = np.array([4.0, 10.0, np.inf, 3.0])
    cut = dt.date(2026, 2, 16)  # six weeks after the first three rows were created
    lo, hi = labels_as_of(anchors, lower, upper, cut, 7)
    # resolved at 4 weeks: known at the cut
    assert (lo[0], hi[0]) == (4.0, 4.0)
    # resolved at 10 weeks: still open at the cut, with the 6 weeks it had waited
    assert lo[1] == pytest.approx(6.0) and np.isinf(hi[1])
    # open today: open at the cut too
    assert lo[2] == pytest.approx(6.0) and np.isinf(hi[2])
    # created after the cut: did not exist
    assert np.isnan(lo[3]) and np.isnan(hi[3])


def test_series_cuts_follow_the_horizon_and_shrink_for_short_history():
    cuts = series_cuts(156, 8)
    assert (cuts.steps - cuts.test_start, cuts.test_start - cuts.val_start) == (13, 26)
    short = series_cuts(80, 8)
    assert short.val_start >= 26 and short.steps - short.test_start >= 8
    with pytest.raises(SplitError, match="cannot hold"):
        series_cuts(30, 8)
