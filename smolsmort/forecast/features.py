"""model inputs from a prepared table. row mode: one row per entity. series mode: one table per
horizon bucket, every feature read at the origin d - b, so nothing after the origin can leak in"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from smolsmort.forecast.profile import _detect_periods, _series_totals
from smolsmort.forecast.spec import ANCHOR, SERIES, STEP, PrepSpec
from smolsmort.forecast.tables import is_text

# one direct model per bucket, lags >= the bucket's largest step (design review 2026-09-23)
BUCKETS = ((1, 1), (2, 4), (5, 8), (9, 13), (14, 26), (27, 52))
LAGS = (0, 1, 2, 3)
WINDOWS = (4, 8, 13, 26, 52)
ALPHAS = (0.1, 0.3, 0.5)


@dataclass
class Features:
    x: np.ndarray
    names: list[str]
    types: list[str]
    families: list[str]

    def pick(self, families) -> tuple[np.ndarray, list[str], list[str]]:
        """the columns of the chosen families, in table order"""
        keep = [i for i, f in enumerate(self.families) if f in families]
        return self.x[:, keep], [self.names[i] for i in keep], [self.types[i] for i in keep]


class _Builder:
    def __init__(self, rows: int):
        self.rows, self.cols, self.names, self.types, self.families = rows, [], [], [], []

    def add(self, name, values, kind="q", family=None):
        self.cols.append(np.asarray(values, dtype=np.float32))
        self.names.append(name)
        self.types.append(kind)
        self.families.append(family or name)

    def done(self) -> Features:
        x = np.column_stack(self.cols) if self.cols else np.zeros((self.rows, 0), np.float32)
        return Features(x.astype(np.float32), self.names, self.types, self.families)


def _codes(values, train) -> np.ndarray:
    """category codes from the levels seen in training; anything else is missing"""
    text = np.array(["" if v is None or v != v else str(v) for v in values], dtype=object)
    levels = {v: i for i, v in enumerate(sorted({v for v in text[train] if v}))}
    return np.array([levels.get(v, np.nan) for v in text], dtype=float)


def _calendar(days: np.ndarray) -> dict[str, np.ndarray]:
    days = days.astype("datetime64[D]")
    day_of_year = (days - days.astype("datetime64[Y]")).astype(int)
    month = days.astype("datetime64[M]").astype(int) % 12 + 1
    return {
        "week": day_of_year // 7 + 1,
        "month": month,
        "quarter": (month - 1) // 3 + 1,
        # 1970-01-01 was a thursday; monday = 0
        "weekday": (days.astype(int) + 3) % 7,
    }


@lru_cache(maxsize=32)
def _holiday_dates(country: str, first_year: int, last_year: int) -> np.ndarray:
    try:
        import holidays
    except ImportError as exc:
        raise ImportError(
            "holiday features need the holidays library; run uv sync --extra holidays"
        ) from exc
    calendar = holidays.country_holidays(country, years=range(first_year - 1, last_year + 2))
    return np.array(sorted(calendar), dtype="datetime64[D]").astype(int)


def _series_calendar(days: np.ndarray, unit: str, country: str | None = None) -> dict:
    """additional families read only the target calendar, including known future dates"""
    days = days.astype("datetime64[D]")
    if unit == "week":
        days = days - ((_calendar(days)["weekday"]).astype("timedelta64[D]"))
    parts = _calendar(days)
    years = days.astype("datetime64[Y]")
    next_year = years + np.timedelta64(1, "Y")
    year_days = (next_year.astype("datetime64[D]") - years.astype("datetime64[D]")).astype(int)
    day_of_year = (days - years).astype(int)
    cycles = {"month": (parts["month"] - 1) / 12, "day_of_year": day_of_year / year_days}
    if unit == "day":
        cycles = {"weekday": parts["weekday"] / 7, **cycles}
    cols = {}
    for part, cycle in cycles.items():
        angle = 2 * np.pi * cycle
        cols[f"cal_{part}_sin"] = ("cal:cyclic", np.sin(angle))
        cols[f"cal_{part}_cos"] = ("cal:cyclic", np.cos(angle))
    if unit == "day":
        months = days.astype("datetime64[M]")
        dom = (days - months).astype(int) + 1
        next_month = months + np.timedelta64(1, "M")
        month_end = next_month.astype("datetime64[D]") - np.timedelta64(1, "D")
        for name, values in {
            "day_of_month": dom,
            "month_start": dom == 1,
            "month_end": days == month_end,
            "week_of_month": (dom - 1) // 7 + 1,
        }.items():
            cols[f"cal_{name}"] = ("cal:dom", values)
    if country is not None:
        year_numbers = years.astype(int) + 1970
        dates = _holiday_dates(country, int(year_numbers.min()), int(year_numbers.max()))
        ordinal = days.astype(int)
        bounds = np.concatenate((ordinal, dates))
        padded = np.concatenate(([bounds.min() - 31], dates, [bounds.max() + 31]))
        next_index = np.searchsorted(padded, ordinal, side="left")
        last_index = np.searchsorted(padded, ordinal, side="right") - 1
        for name, values in {
            "holiday": np.isin(ordinal, dates),
            "days_to_holiday": np.minimum(padded[next_index] - ordinal, 30),
            "days_since_holiday": np.minimum(ordinal - padded[last_index], 30),
        }.items():
            cols[f"cal_{name}"] = ("cal:holiday", values)
    return cols


def buckets_for(horizon: int) -> list[tuple[int, int]]:
    out = [(lo, min(hi, horizon)) for lo, hi in BUCKETS if lo <= horizon]
    if horizon > BUCKETS[-1][1]:
        out.append((BUCKETS[-1][1] + 1, horizon))
    return out


# ---------------------------------------------------------------- row mode


def row_features(spec: PrepSpec, table: dict, train) -> Features:
    """every candidate input column for row mode; the search picks families from these"""
    train = np.asarray(train, dtype=bool)
    build = _Builder(len(table[ANCHOR]))
    for name in spec.inputs():
        values = table[name]
        if is_text(values):
            codes = _codes(values, train)
            build.add(f"{name}", codes, "c", f"dim:{name}")
            counts = np.bincount(codes[train & np.isfinite(codes)].astype(int)) / max(
                train.sum(), 1
            )
            freq = np.array(
                [counts[int(c)] if np.isfinite(c) and c < len(counts) else 0 for c in codes]
            )
            build.add(f"{name}_frequency", freq, family=f"freq:{name}")
        else:
            build.add(name, values, family=f"measure:{name}")
            seen = values[train & np.isfinite(values)]
            if len(seen) and seen.min() >= 0:
                build.add(f"{name}_log", np.log1p(values), family=f"log:{name}")
    anchors = table[ANCHOR].astype("datetime64[D]")
    for part, values in _calendar(anchors).items():
        build.add(f"anchor_{part}", values, family=f"date:{part}")
    start = anchors[train].min() if train.any() else anchors.min()
    build.add("anchor_years", (anchors - start).astype(int) / 365.25, family="date:trend")
    return build.done()


# ---------------------------------------------------------------- series mode


def step_index(steps: np.ndarray, unit: str) -> np.ndarray:
    """whole steps since a fixed origin, so a step difference is a count of steps"""
    days = steps.astype("datetime64[D]")
    if unit == "day":
        return days.astype(int)
    if unit == "week":
        # 1970-01-05 was the first monday; prep truncates weeks to mondays
        return (days.astype(int) - 4) // 7
    months = days.astype("datetime64[M]").astype(int)
    return {"month": months, "quarter": months // 3, "year": months // 12}[unit]


def _fill_value(aggregation: str | None) -> float | None:
    """what a step past a series' end holds: nothing sold is zero, an average is unknown"""
    if aggregation in ("sum", "count"):
        return 0.0
    if aggregation == "last":
        return None  # carried forward
    return np.nan


def _extend(values: np.ndarray, length: int, aggregation) -> np.ndarray:
    out = np.full(length, np.nan)
    out[: len(values)] = values
    fill = _fill_value(aggregation)
    if len(values) < length:
        if fill is None:
            last = values[np.isfinite(values)]
            out[len(values) :] = last[-1] if len(last) else np.nan
        else:
            out[len(values) :] = fill
    return out


def _rolling(values: np.ndarray, window: int):
    """mean, std, max and zero share over the `window` values ending at each position"""
    n = len(values)
    finite = np.isfinite(values)
    zero_filled = np.where(finite, values, 0.0)
    cs = np.concatenate([[0], np.cumsum(zero_filled)])
    cs2 = np.concatenate([[0], np.cumsum(zero_filled**2)])
    cn = np.concatenate([[0], np.cumsum(finite)])
    cz = np.concatenate([[0], np.cumsum(finite & (values == 0))])
    end = np.arange(1, n + 1)
    start = np.maximum(end - window, 0)
    count = cn[end] - cn[start]
    enough = count >= max(window // 2, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cs[end] - cs[start]) / count
        var = (cs2[end] - cs2[start]) / count - mean**2
        zeros = (cz[end] - cz[start]) / count
    padded = np.concatenate([np.full(window - 1, -np.inf), np.where(finite, values, -np.inf)])
    high = np.lib.stride_tricks.sliding_window_view(padded, window).max(axis=1)
    high = np.where(np.isfinite(high), high, np.nan)
    nan = np.full(n, np.nan)
    return (
        np.where(enough, mean, nan),
        np.where(enough, np.sqrt(np.maximum(var, 0)), nan),
        np.where(enough, high, nan),
        np.where(enough, zeros, nan),
    )


def _ewm(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full(len(values), np.nan)
    level = np.nan
    for i, v in enumerate(values):
        if np.isfinite(v):
            level = v if not np.isfinite(level) else alpha * v + (1 - alpha) * level
        out[i] = level
    return out


def _shift(values: np.ndarray, by: int) -> np.ndarray:
    """values[i - by] at position i, missing where that is before the start"""
    out = np.full(len(values), np.nan)
    if by < len(values):
        out[by:] = values[: len(values) - by]
    return out


@dataclass
class SeriesFrame:
    features: Features
    y: np.ndarray
    series: np.ndarray
    step: np.ndarray
    origin: np.ndarray
    future: np.ndarray


def series_features(
    spec: PrepSpec,
    panel: dict,
    scaffold: dict,
    bucket: tuple[int, int],
    unit: str,
    train,
    *,
    period: int | None = None,
    periods: tuple[int, ...] | None = None,
    measure_lags: dict[str, int] | None = None,
) -> SeriesFrame:
    """the rows one bucket's model trains, is judged and predicts on. a target at step d with
    origin d - b sees the target and measures up to the origin only, and the calendar of d"""
    lo, b = bucket
    target = spec.named("target")[0]
    aggregation = {c.name: c.aggregation for c in spec.columns}
    measures = [m for m in spec.named("measure") if m in panel]
    dims = [d for d in spec.named("dimension") if d in panel]
    train = np.asarray(train, dtype=bool)
    history = len(np.unique(panel[STEP][train]))
    if unit != "week" and periods is None:
        _, totals = _series_totals(panel, train, target)
        periods = tuple(_detect_periods(totals, unit))
    detected = set(periods or ())
    if period:
        detected.add(period)
    absolute_lags, windows = _step_features(unit, history, detected)
    dim_codes = {d: _codes(panel[d], train) for d in dims}
    last = step_index(panel[STEP], unit).max()
    parts = []
    for key in np.unique(panel[SERIES]):
        mine = np.flatnonzero(panel[SERIES] == key)
        mine = mine[np.argsort(panel[STEP][mine])]
        index = step_index(panel[STEP][mine], unit)
        first = index[0]
        length = max(index[-1], last) - first + 1
        series_values = {
            name: _extend(panel[name][mine], length, aggregation.get(name))
            for name in [target, *measures]
        }
        base = _series_base(
            series_values,
            target,
            measures,
            b,
            period,
            measure_lags or {},
            absolute_lags,
            windows,
            preserve_relative=unit != "week",
        )
        rows = [(p, panel[target][mine][p], False) for p in range(len(mine)) if p - b >= 0]
        future = np.flatnonzero(scaffold[SERIES] == key)
        f_index = step_index(scaffold[STEP][future], unit)
        for fi, p in zip(future, f_index - first, strict=True):
            horizon = p - (last - first)
            if lo <= horizon <= b:
                rows.append((p, np.nan, True, fi))
        if not rows:
            continue
        positions = np.array([r[0] for r in rows])
        origins = positions - b
        step = np.array(
            [panel[STEP][mine][r[0]] if not r[2] else scaffold[STEP][r[3]] for r in rows]
        ).astype("datetime64[D]")
        cols = {name: (family, col[origins]) for name, (family, col) in base.items()}
        for part, values in _calendar(step).items():
            if part != "weekday" or unit == "day":
                cols[f"cal_{part}"] = (f"cal:{part}", values)
        for d in dims:
            cols[d] = (f"dim:{d}", np.full(len(rows), dim_codes[d][mine[0]]))
        cols.update(_series_calendar(step, unit, spec.holidays_country))
        parts.append(
            {
                "cols": cols,
                "y": np.array([r[1] for r in rows], dtype=float),
                "series": np.full(len(rows), key, dtype=object),
                "step": step,
                "origin": step - np.array(b * _unit_days(unit), dtype="timedelta64[D]"),
                "future": np.array([r[2] for r in rows]),
            }
        )
    names = list(parts[0]["cols"]) if parts else []
    build = _Builder(sum(len(p["y"]) for p in parts))
    for name in names:
        family = parts[0]["cols"][name][0]
        values = np.concatenate([p["cols"][name][1] for p in parts])
        build.add(name, values, "c" if family.startswith("dim:") else "q", family)
    return SeriesFrame(
        build.done(),
        np.concatenate([p["y"] for p in parts]),
        np.concatenate([p["series"] for p in parts]),
        np.concatenate([p["step"] for p in parts]),
        np.concatenate([p["origin"] for p in parts]),
        np.concatenate([p["future"] for p in parts]),
    )


def _unit_days(unit: str) -> int:
    # only used to date the origin for display and the leak test; months vary, weeks do not
    return {"day": 1, "week": 7, "month": 30, "quarter": 91, "year": 365}[unit]


def _step_features(unit, history, periods):
    """weekly families stay fixed; other units expose only history-supported windows and lags"""
    if unit == "week":
        return (), WINDOWS
    lags, windows = {
        "day": ((7, 14, 28, 364), (7, 14, 28, 91, 182)),
        "month": ((12,), (3, 6, 12)),
        "quarter": ((4,), (2, 4, 8)),
        "year": ((2, 3), (2, 3, 5)),
    }[unit]
    return (
        tuple(sorted(k for k in set(lags) | set(periods) if k < history)),
        tuple(w for w in windows if w <= history),
    )


def _series_base(
    values,
    target,
    measures,
    b,
    period,
    measure_lags,
    absolute_lags=(),
    windows=WINDOWS,
    *,
    preserve_relative=False,
) -> dict:
    """name -> (family, values) at every origin position g, built from positions <= g only.
    families are relative to the bucket (lag+0 is the freshest lag any bucket may read), so one
    genome means the same thing in every bucket's model"""
    y = values[target]
    base = {f"lag_{b + j}": (f"lag+{j}", _shift(y, j)) for j in LAGS}
    if period and period >= b:
        name = f"lag_{period}"
        if preserve_relative and name in base:
            name = f"season_lag_{period}"
        base[name] = ("season", _shift(y, period - b))
    for k in absolute_lags:
        if k >= b:
            # separate columns retain relative-lag and season families when their offsets coincide
            base[f"absolute_lag_{k}"] = (f"lag:{k}", _shift(y, k - b))
    for w in windows:
        for stat, col in zip(("mean", "std", "max", "zeros"), _rolling(y, w), strict=True):
            base[f"roll{w}_{stat}"] = (f"roll:{w}", col)
    for a in ALPHAS:
        base[f"ewm{a}"] = (f"ewm:{a}", _ewm(y, a))
    for m in measures:
        for j in LAGS:
            base[f"{m}_lag_{b + j}"] = (f"measure:{m}", _shift(values[m], j))
        k = measure_lags.get(m)
        if k and k >= b:
            base[f"{m}_lag_{k}"] = (f"lead:{m}", _shift(values[m], k - b))
    return base
