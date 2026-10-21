"""the contract between forecast units: what a prep spec says and what a prepared table holds.
the spec hashes to the cache key; the reserved column names below are the only coupling between
units, and user columns keep their own names beside them"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, cast

Role = Literal["time", "anchor", "dimension", "input", "measure", "target", "ignore"]
Mode = Literal["timeseries", "entity"]
Task = Literal["regression", "classification"]
Aggregation = Literal["sum", "mean", "median", "min", "max", "count", "last"]
Step = Literal["day", "week", "month", "quarter", "year"]

ROW = "__row"
ANCHOR = "__anchor"
LOWER = "__lower"
UPPER = "__upper"
STEP = "__step"
SERIES = "__series"

# what prep writes per mode; series rows are gap-filled only inside each series' own span
FILES = {
    "entity": {"rows.parquet": (ROW, ANCHOR, LOWER, UPPER), "predict.parquet": (ROW, ANCHOR)},
    "timeseries": {"panel.parquet": (STEP, SERIES), "scaffold.parquet": (STEP, SERIES)},
}

# saved specs use these names; the same mapping preserves their prepared cache keys
MODE_ALIASES = {"series": "timeseries", "row": "entity"}

# days per unit, for censoring bounds in the target's own unit
UNIT_DAYS = {"day": 1, "week": 7, "month": 30.4375, "quarter": 91.3125, "year": 365.25}


class SpecError(ValueError):
    pass


def normalize_mode(mode: str) -> Mode:
    """accept saved mode names at input boundaries and use canonical names inside"""
    if not isinstance(mode, str):
        raise SpecError("spec.mode must be timeseries or entity")
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in FILES:
        raise SpecError("spec.mode must be timeseries or entity")
    return cast(Mode, mode)


def normalize_saved_modes(data: dict) -> dict:
    """normalize mode fields in saved forecast records without changing their other fields"""
    normalized = data
    if "mode" in data:
        mode = normalize_mode(data["mode"])
        if mode != data["mode"]:
            normalized = {**normalized, "mode": mode}
    for name in ("spec", "summary", "recipe"):
        if isinstance(data.get(name), dict):
            nested = normalize_saved_modes(data[name])
            if nested is not data[name]:
                normalized = {**normalized, name: nested}
    return normalized


@dataclass(frozen=True)
class Column:
    name: str
    role: Role
    # time series mode: how a measure or target rolls up to one step
    aggregation: Aggregation | None = None
    # entity mode: whether the value exists when a prediction is made. a column set later than the
    # anchor (a value recorded only once the outcome is known) must never become an input
    known: bool = True
    # input shares and spread use only levels with a value
    skip_missing: bool = False

    def __post_init__(self):
        if type(self.skip_missing) is not bool:
            raise SpecError("column.skip_missing must be a boolean")
        if self.skip_missing and self.role != "input":
            raise SpecError("column.skip_missing is only supported for input columns")


@dataclass(frozen=True)
class Censor:
    # an open row still says "at least this much": lower = (as_of - anchor) in unit, upper = inf
    anchor: str
    unit: Step = "week"
    # the export date; empty means the latest anchor value in the file
    as_of: str | None = None
    # the target holds an event date instead of a duration when set
    event: str | None = None


@dataclass(frozen=True)
class PrepSpec:
    source: str
    mode: Mode
    task: Task
    columns: tuple[Column, ...]
    encoding: str = "auto"
    # time series mode: the step every series is aggregated to; empty means infer it from the data
    step: Step | None = None
    horizon: int = 8
    # extra sql filters on top of the role-driven ones, applied before anything else
    where: str | None = None
    predict_where: str | None = None
    censor: Censor | None = None
    # time series mode: the export date; steps from the one holding it onward are scheduled, not history
    as_of: str | None = None
    holidays_country: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "mode", normalize_mode(self.mode))
        roles = [c.role for c in self.columns]
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise SpecError("a column appears twice in the spec")
        if self.mode == "entity" and "input" in roles:
            raise SpecError("the input role is only supported in time series mode")
        if "target" not in roles:
            raise SpecError("the spec names no target column")
        clock = "time" if self.mode == "timeseries" else "anchor"
        if roles.count(clock) != 1:
            raise SpecError(f"{self.mode} mode needs exactly one {clock} column")
        if self.mode == "timeseries":
            for col in self.columns:
                if col.role in ("measure", "target") and col.aggregation is None:
                    raise SpecError(f"{col.name!r} needs an aggregation in time series mode")
        if self.censor and self.censor.anchor not in names:
            raise SpecError(f"censor anchor {self.censor.anchor!r} is not a column")
        if self.censor and self.censor.event is not None:
            if self.mode != "entity":
                raise SpecError("censor.event is only supported in entity mode")
            if self.task != "regression":
                raise SpecError("censor.event requires regression")
            if self.censor.event != self.named("target")[0]:
                raise SpecError("censor.event must name the target column")
            if self.censor.anchor != self.named("anchor")[0]:
                raise SpecError(
                    "censor.anchor must name the anchor column when censor.event is set"
                )
        if self.horizon < 1:
            raise SpecError("horizon must be at least one step")

    def named(self, role: Role) -> list[str]:
        return [c.name for c in self.columns if c.role == role]

    def inputs(self) -> list[str]:
        """columns a model may read: dimensions and measures known when predicting"""
        return [c.name for c in self.columns if c.role in ("dimension", "measure") and c.known]

    def digest(self) -> str:
        """the spec half of the cache key; prep adds the source file's content hash"""
        data = asdict(self)
        data["mode"] = next(old for old, mode in MODE_ALIASES.items() if mode == self.mode)
        for column in data["columns"]:
            if not column["skip_missing"]:
                column.pop("skip_missing")
        if self.holidays_country is None:
            data.pop("holidays_country")
        if self.censor and self.censor.event is None:
            data["censor"].pop("event")
        text = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def spec_from_dict(data: dict) -> PrepSpec:
    """a spec from its json form, the shape the data screen posts"""
    columns = tuple(Column(**c) for c in data.get("columns", ()))
    censor = Censor(**data["censor"]) if data.get("censor") else None
    rest = {k: v for k, v in data.items() if k not in ("columns", "censor")}
    return PrepSpec(columns=columns, censor=censor, **rest)
