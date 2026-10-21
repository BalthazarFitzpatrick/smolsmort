"""model inputs from a prepared table. row mode: one row per entity. series mode: one table per
horizon bucket, every feature read at the origin d - b, so nothing after the origin can leak in"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import quote, unquote

import numpy as np

from smolsmort.forecast.profile import _detect_periods, _series_totals
from smolsmort.forecast.spec import ANCHOR, SERIES, STEP, PrepSpec
from smolsmort.forecast.tables import is_text

# one direct model per bucket, lags >= the bucket's largest step (design review 2026-09-23)
BUCKETS = ((1, 1), (2, 4), (5, 8), (9, 13), (14, 26), (27, 52))
LAGS = (0, 1, 2, 3)
WINDOWS = (4, 8, 13, 26, 52)
ALPHAS = (0.1, 0.3, 0.5)
LEGACY_ROLL_STATS = ("mean", "std", "max", "zeros")
ROLL_STATS = (*LEGACY_ROLL_STATS, "min", "median", "q10", "q90")
# absolute lags, rolling windows, origin differences and fallback calendar season
STEP_FEATURES = {
    "week": ((*range(1, 14), 26, 52), WINDOWS, (1, 4, 13, 52), 52),
    "day": ((7, 14, 28, 364), (7, 14, 28, 91, 182), (1, 7, 28, 365), 7),
    "month": ((12,), (3, 6, 12), (1, 3, 12), 12),
    "quarter": ((4,), (2, 4, 8), (1, 2, 4), 4),
    "year": ((2, 3), (2, 3, 5), (1, 2, 3), None),
}
LEGACY_WINDOWS = {"week": WINDOWS}
LEGACY_PREFIXES = {
    "cal",
    "date",
    "dim",
    "freq",
    "measure",
    "log",
    "lead",
    "mix",
    "spread",
    "lag",
    "roll",
    "diff",
    "slope",
    "momentum",
    "ewm",
}


def family_name(op: str, column: str | None = None, clock: str | None = None) -> str:
    """name an operation by its source column and, when temporal, its source clock"""

    def encode_source(source):
        encoded = quote(source, safe="")
        # reserved words must not collide with unversioned saved names such as dim:measure
        return f"%{ord(encoded[0]):02X}{encoded[1:]}" if encoded in LEGACY_PREFIXES else encoded

    owner = encode_source(column) if column is not None else ""
    if clock is not None:
        owner += ("@" if column is not None else "") + encode_source(clock)
    return f"{owner}:{op}" if column is not None or clock is not None else op


def split_family_name(name: str) -> tuple[str | None, str | None, str]:
    """split source-aware names; an old name remains an unqualified operation"""
    owner, separator, op = name.partition(":")
    if owner in LEGACY_PREFIXES:
        return None, None, name
    if separator and "@" in owner:
        column, clock = owner.split("@", 1)
        return unquote(column), unquote(clock), op
    if separator and op.partition(":")[0] == "cal":
        return None, unquote(owner), op
    if separator and op in {"dim", "freq", "measure", "log", "lead", "mix", "spread"}:
        return unquote(owner), None, op
    return None, None, name


def family_sort_key(name: str) -> str:
    """retain the old sampling and tie order when source labels change"""
    column, clock, op = split_family_name(name)
    if column is not None and op in {"dim", "freq", "measure", "log", "lead", "mix", "spread"}:
        return f"{op}:{column}"
    return op


def family_aliases(spec: PrepSpec, names=()) -> dict[str, str]:
    """old saved names mapped with the prep spec, including detected periods and roll aliases"""
    clock = spec.named("time" if spec.mode == "series" else "anchor")[0]
    target = spec.named("target")[0]
    aliases = {}
    for column in spec.columns:
        for op in ("dim", "freq", "measure", "log", "lead", "mix", "spread"):
            temporal = spec.mode == "series" and op in {"measure", "lead", "mix", "spread"}
            aliases[f"{op}:{column.name}"] = family_name(
                op, column.name, clock if temporal else None
            )
    for part in ("week", "month", "quarter", "weekday", "cyclic", "dom", "holiday", "trend"):
        aliases[f"{'cal' if spec.mode == 'series' else 'date'}:{part}"] = family_name(
            f"cal:{part}", clock=clock
        )
    if spec.mode == "series":
        operations = {"season", "cycle_ratio", *(f"lag+{j}" for j in LAGS)}
        operations.update(f"ewm:{alpha}" for alpha in ALPHAS)
        for lags, windows, differences, _ in STEP_FEATURES.values():
            operations.update(f"lag:{lag}" for lag in lags)
            operations.update(f"diff:{lag}" for lag in differences)
            for window in windows:
                operations.update(f"{op}:{window}" for op in ("roll", "slope", "momentum"))
                operations.update(f"roll:{window}:{stat}" for stat in ROLL_STATS)
        for name in names:
            column, time, op = split_family_name(name)
            kind = op.partition(":")[0].partition("+")[0]
            if (
                column is None
                and time is None
                and kind
                in {"lag", "roll", "diff", "slope", "momentum", "cycle_ratio", "season", "ewm"}
                and (":" in op or op in {"season", "cycle_ratio"} or "+" in op)
            ):
                operations.add(op)
        aliases.update({op: family_name(op, target, clock) for op in operations})
    return aliases


@dataclass
class Features:
    x: np.ndarray
    names: list[str]
    types: list[str]
    families: list[str]
    aliases: dict[str, str] = field(default_factory=dict)

    def pick(self, families) -> tuple[np.ndarray, list[str], list[str]]:
        """the columns of the chosen families, in table order"""
        selected = {self.aliases.get(name, name) for name in families}
        for family in tuple(selected):
            _, _, op = split_family_name(family)
            if op.partition(":")[0] == "roll" and op.count(":") == 1:
                selected.update(f"{family}:{stat}" for stat in LEGACY_ROLL_STATS)
        keep = [i for i, f in enumerate(self.families) if f in selected]
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


def _series_calendar(
    days: np.ndarray, unit: str, country: str | None = None, *, clock: str | None = None
) -> dict:
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
    return {name: (family_name(op, clock=clock), values) for name, (op, values) in cols.items()}


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
            build.add(f"{name}", codes, "c", family_name("dim", name))
            counts = np.bincount(codes[train & np.isfinite(codes)].astype(int)) / max(
                train.sum(), 1
            )
            freq = np.array(
                [counts[int(c)] if np.isfinite(c) and c < len(counts) else 0 for c in codes]
            )
            build.add(f"{name}_frequency", freq, family=family_name("freq", name))
        else:
            build.add(name, values, family=family_name("measure", name))
            seen = values[train & np.isfinite(values)]
            if len(seen) and seen.min() >= 0:
                build.add(f"{name}_log", np.log1p(values), family=family_name("log", name))
    anchors = table[ANCHOR].astype("datetime64[D]")
    clock = spec.named("anchor")[0]
    for part, values in _calendar(anchors).items():
        build.add(f"anchor_{part}", values, family=family_name(f"cal:{part}", clock=clock))
    start = anchors[train].min() if train.any() else anchors.min()
    build.add(
        "anchor_years",
        (anchors - start).astype(int) / 365.25,
        family=family_name("cal:trend", clock=clock),
    )
    features = build.done()
    features.aliases = family_aliases(spec)
    return features


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


def _window_sum(values: np.ndarray, window: int) -> np.ndarray:
    total = np.concatenate(([0], np.cumsum(values)))
    end = np.arange(1, len(values) + 1)
    return total[end] - total[np.maximum(end - window, 0)]


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values)
    count = _window_sum(finite, window)
    total = _window_sum(np.where(finite, values, 0.0), window)
    return np.divide(
        total, count, out=np.full(len(values), np.nan), where=count >= max(window // 2, 1)
    )


def _rolling(values: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """finite-value statistics ending at each origin, with the legacy half-window threshold"""
    n = len(values)
    finite = np.isfinite(values)
    zero_filled = np.where(finite, values, 0.0)
    count = _window_sum(finite, window)
    enough = count >= max(window // 2, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = _window_sum(zero_filled, window) / count
        var = _window_sum(zero_filled**2, window) / count - mean**2
        zeros = _window_sum(finite & (values == 0), window) / count
    padded = np.concatenate([np.full(window - 1, -np.inf), np.where(finite, values, -np.inf)])
    high = np.lib.stride_tricks.sliding_window_view(padded, window).max(axis=1)
    high = np.where(np.isfinite(high), high, np.nan)
    nan = np.full(n, np.nan)
    padded = np.concatenate([np.full(window - 1, np.inf), np.where(finite, values, np.inf)])
    ordered = np.sort(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
    stats = {
        "mean": np.where(enough, mean, nan),
        "std": np.where(enough, np.sqrt(np.maximum(var, 0)), nan),
        "max": np.where(enough, high, nan),
        "zeros": np.where(enough, zeros, nan),
        "min": np.where(enough, ordered[:, 0], nan),
    }
    rows = np.flatnonzero(enough)
    for name, quantile in (("median", 0.5), ("q10", 0.1), ("q90", 0.9)):
        position = (count[rows] - 1) * quantile
        low = ordered[rows, np.floor(position).astype(int)]
        high = ordered[rows, np.ceil(position).astype(int)]
        column = np.full(n, np.nan)
        column[rows] = low + (high - low) * (position - np.floor(position))
        stats[name] = column
    return stats


def _slope(values: np.ndarray, window: int) -> np.ndarray:
    """least-squares change per step, retaining the time gaps of missing observations"""
    padded = np.concatenate([np.full(window - 1, np.nan), values])
    seen = np.lib.stride_tricks.sliding_window_view(padded, window)
    finite = np.isfinite(seen)
    count = finite.sum(axis=1)
    x = np.arange(window, dtype=float)
    y = np.where(finite, seen, 0.0)
    sum_x = (finite * x).sum(axis=1)
    sum_x2 = (finite * x**2).sum(axis=1)
    numerator = count * (y * x).sum(axis=1) - sum_x * y.sum(axis=1)
    denominator = count * sum_x2 - sum_x**2
    enough = (count >= max(window // 2, 2)) & (denominator > 0)
    return np.divide(numerator, denominator, out=np.full(len(values), np.nan), where=enough)


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
    base: np.ndarray


def _build_input_values(spec: PrepSpec, panel: dict, inputs: dict | None, train) -> dict:
    """input shares and spread aligned to the panel, with levels frozen across all series"""
    if inputs is None or not spec.named("input"):
        return {}
    rows = len(panel[STEP])
    panel_keys = zip(panel[SERIES], panel[STEP].astype("datetime64[D]"), strict=True)
    positions = {key: index for index, key in enumerate(panel_keys)}
    input_keys = zip(inputs[SERIES], inputs[STEP].astype("datetime64[D]"), strict=True)
    indices = np.array([positions.get(key, -1) for key in input_keys], dtype=int)
    columns = {}
    clock = spec.named("time")[0]
    for column in spec.columns:
        if column.role != "input":
            continue
        name = column.name
        label = quote(name, safe="")
        keep = (inputs["__column"] == name) & (indices >= 0)
        if column.skip_missing:
            keep &= inputs["__level"] != "(none)"
        index = indices[keep]
        levels = inputs["__level"][keep]
        target = np.asarray(inputs["target"][keep], dtype=float)
        train_rows = train[index]
        seen, codes = np.unique(levels[train_rows], return_inverse=True)
        totals = np.bincount(
            codes, weights=np.nan_to_num(target[train_rows], nan=0.0), minlength=len(seen)
        )
        top = sorted(zip(seen, totals, strict=True), key=lambda item: (-item[1], item[0]))[:5]
        top_levels = [level for level, _ in top]
        count = np.bincount(index, minlength=rows)
        total = np.bincount(index, weights=target, minlength=rows)
        # an observed zero total has zero shares; steps without source rows remain missing
        shares = np.divide(
            target,
            total[index],
            out=np.zeros(len(index)),
            where=total[index] != 0,
        )
        for level in top_levels:
            values = np.bincount(
                index[levels == level], weights=shares[levels == level], minlength=rows
            )
            level_label = quote(level, safe="")
            columns[f"mix:{label}:level:{level_label}"] = (
                family_name("mix", name, clock),
                np.where(count, values, np.nan),
            )
        other = ~np.isin(levels, top_levels)
        values = np.bincount(index[other], weights=shares[other], minlength=rows)
        columns[f"mix:{label}:other"] = (
            family_name("mix", name, clock),
            np.where(count, values, np.nan),
        )
        columns[f"spread:{label}:count"] = (
            family_name("spread", name, clock),
            np.where(count, count, np.nan),
        )
        hhi = np.bincount(index, weights=shares**2, minlength=rows)
        columns[f"spread:{label}:hhi"] = (
            family_name("spread", name, clock),
            np.where(count, hhi, np.nan),
        )
    return columns


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
    inputs: dict | None = None,
) -> SeriesFrame:
    """the rows one bucket's model trains, is judged and predicts on. a target at step d with
    origin d - b sees the target and measures up to the origin only, and the calendar of d"""
    lo, b = bucket
    target = spec.named("target")[0]
    clock = spec.named("time")[0]
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
    _, _, differences, fallback_period = STEP_FEATURES[unit]
    differences = tuple(k for k in differences if k < history)
    cycle_period = period or min(detected, default=fallback_period)
    if cycle_period is not None and cycle_period >= history:
        cycle_period = None
    dim_codes = {d: _codes(panel[d], train) for d in dims}
    input_values = _build_input_values(spec, panel, inputs, train)
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
            differences=differences,
            dynamics_windows=tuple(w for w in windows if w <= history),
            cycle_period=cycle_period,
            clock=clock,
        )
        for name, (family, values) in input_values.items():
            extended = _extend(values[mine], length, None)
            for j in LAGS:
                base[f"{name}_lag_{b + j}"] = (family, _shift(extended, j))
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
                cols[f"cal_{part}"] = (family_name(f"cal:{part}", clock=clock), values)
        for d in dims:
            cols[d] = (family_name("dim", d), np.full(len(rows), dim_codes[d][mine[0]]))
        cols.update(_series_calendar(step, unit, spec.holidays_country, clock=clock))
        parts.append(
            {
                "cols": cols,
                "y": np.array([r[1] for r in rows], dtype=float),
                "series": np.full(len(rows), key, dtype=object),
                "step": step,
                "origin": step - np.array(b * _unit_days(unit), dtype="timedelta64[D]"),
                "future": np.array([r[2] for r in rows]),
                "base": series_values[target][origins],
            }
        )
    names = list(parts[0]["cols"]) if parts else []
    build = _Builder(sum(len(p["y"]) for p in parts))
    for name in names:
        family = parts[0]["cols"][name][0]
        values = np.concatenate([p["cols"][name][1] for p in parts])
        build.add(name, values, "c" if split_family_name(family)[2] == "dim" else "q", family)
    features = build.done()
    features.aliases = family_aliases(spec, (family_sort_key(name) for name in features.families))
    return SeriesFrame(
        features,
        np.concatenate([p["y"] for p in parts]),
        np.concatenate([p["series"] for p in parts]),
        np.concatenate([p["step"] for p in parts]),
        np.concatenate([p["origin"] for p in parts]),
        np.concatenate([p["future"] for p in parts]),
        np.concatenate([p["base"] for p in parts]),
    )


def _unit_days(unit: str) -> int:
    # only used to date the origin for display and the leak test; months vary, weeks do not
    return {"day": 1, "week": 7, "month": 30, "quarter": 91, "year": 365}[unit]


def _step_features(unit, history, periods):
    """history-supported candidates, retaining weekly windows for saved recipes"""
    lags, windows, _, _ = STEP_FEATURES[unit]
    return (
        tuple(sorted(k for k in set(lags) | set(periods) if k < history)),
        tuple(w for w in windows if w <= history or w in LEGACY_WINDOWS.get(unit, ())),
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
    differences=(),
    dynamics_windows=(),
    cycle_period=None,
    clock=None,
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
        for stat, col in _rolling(y, w).items():
            base[f"roll{w}_{stat}"] = (f"roll:{w}:{stat}", col)
    finite_y = np.where(np.isfinite(y), y, np.nan)
    for k in differences:
        base[f"diff_{k}"] = (f"diff:{k}", finite_y - _shift(finite_y, k))
    for w in dynamics_windows:
        base[f"slope_{w}"] = (f"slope:{w}", _slope(y, w))
        half = w // 2
        mean = _rolling_mean(y, half)
        base[f"momentum_{w}"] = (f"momentum:{w}", mean - _shift(mean, half))
    if cycle_period:
        previous = _shift(finite_y, cycle_period)
        ratio = np.divide(
            finite_y,
            previous,
            out=np.full(len(y), np.nan),
            where=np.isfinite(previous) & (previous != 0),
        )
        base["cycle_ratio"] = ("cycle_ratio", ratio)
    for a in ALPHAS:
        base[f"ewm{a}"] = (f"ewm:{a}", _ewm(y, a))
    for m in measures:
        for j in LAGS:
            base[f"{m}_lag_{b + j}"] = (f"measure:{m}", _shift(values[m], j))
        k = measure_lags.get(m)
        if k and k >= b:
            base[f"{m}_lag_{k}"] = (f"lead:{m}", _shift(values[m], k - b))
    return {
        name: (
            family_name(op, target, clock)
            if op.partition(":")[0] not in {"measure", "lead"}
            else family_name(op.partition(":")[0], op.partition(":")[2], clock),
            values,
        )
        for name, (op, values) in base.items()
    }
