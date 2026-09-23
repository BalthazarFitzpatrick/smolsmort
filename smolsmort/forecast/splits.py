"""chronological train/val/test cuts, and labels as an export taken at a cut would have shown them.
the test rows are touched once, after the winner is frozen"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

# a first guess at how long a row needs to resolve; set it past the target's own 95th percentile
MATURITY_WEEKS = 26
# one quarter each, as in the gate; the search can shorten them when history is thin
VAL_WEEKS = 13
TEST_WEEKS = 13


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class RowCuts:
    """train < val_start <= val < test_start <= test < test_end <= as_of"""

    val_start: dt.date
    test_start: dt.date
    test_end: dt.date
    as_of: dt.date

    def masks(self, anchors):
        days = _days(anchors)
        v, t, e = (np.datetime64(d, "D") for d in (self.val_start, self.test_start, self.test_end))
        return days < v, (days >= v) & (days < t), (days >= t) & (days < e)


def _days(anchors) -> np.ndarray:
    return np.asarray(anchors).astype("datetime64[D]")


def row_cuts(
    anchors,
    as_of: dt.date,
    *,
    maturity_weeks: int = MATURITY_WEEKS,
    val_weeks: int = VAL_WEEKS,
    test_weeks: int = TEST_WEEKS,
) -> RowCuts:
    """test ends `maturity_weeks` before the export so its rows had time to resolve; maturity 0
    suits a target that is known the moment a row exists"""
    test_end = as_of - dt.timedelta(weeks=maturity_weeks)
    test_start = test_end - dt.timedelta(weeks=test_weeks)
    val_start = test_start - dt.timedelta(weeks=val_weeks)
    days = _days(anchors)
    if not len(days) or days.min() >= np.datetime64(val_start, "D"):
        raise SplitError(
            f"no rows before {val_start} to train on - history is shorter than "
            f"maturity {maturity_weeks} + val {val_weeks} + test {test_weeks} weeks"
        )
    return RowCuts(val_start, test_start, test_end, as_of)


def labels_as_of(anchors, lower, upper, cut: dt.date, unit_days: float):
    """bounds as an export taken at `cut` would have shown them: a row resolved after the cut is
    still open there, with the time it had waited by then as its lower bound"""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    waited = (np.datetime64(cut, "D") - _days(anchors)).astype(float) / unit_days
    resolved = np.isfinite(upper) & (lower <= waited)
    lo = np.where(resolved, lower, np.maximum(waited, 0.0))
    hi = np.where(resolved, upper, np.inf)
    # a row created after the cut did not exist yet
    exists = waited >= 0
    return np.where(exists, lo, np.nan), np.where(exists, hi, np.nan)


@dataclass(frozen=True)
class SeriesCuts:
    """step indices into the sorted distinct steps: train < val_start <= val < test_start <= test"""

    val_start: int
    test_start: int
    steps: int


def series_cuts(steps: int, horizon: int, *, min_train: int = 26) -> SeriesCuts:
    """test = max(horizon, 13) steps, val = max(26, 2 x horizon) steps, shortened when the history
    cannot hold them and still leave `min_train` steps to learn from"""
    test = max(horizon, 13)
    val = max(26, 2 * horizon)
    if steps - test - val < min_train:
        spare = steps - min_train
        if spare < 2 * horizon:
            raise SplitError(
                f"{steps} steps cannot hold {min_train} to train plus a horizon of {horizon} "
                "each for validation and test"
            )
        test = max(horizon, spare // 3)
        val = spare - test
    return SeriesCuts(steps - test - val, steps - test, steps)
