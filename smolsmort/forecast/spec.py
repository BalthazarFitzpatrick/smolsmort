"""the contract between forecast units: what a prep spec says and what a prepared table holds.
the spec hashes to the cache key; the reserved column names below are the only coupling between
units, and user columns keep their own names beside them"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

Role = Literal["time", "anchor", "dimension", "measure", "target", "ignore"]
Mode = Literal["row", "series"]
Task = Literal["regression", "classification"]
Aggregation = Literal["sum", "mean", "min", "max", "count", "last"]
Step = Literal["day", "week", "month", "quarter", "year"]

ROW = "__row"
ANCHOR = "__anchor"
LOWER = "__lower"
UPPER = "__upper"
STEP = "__step"
SERIES = "__series"

# what prep writes per mode; series rows are gap-filled only inside each series' own span
FILES = {
    "row": {"rows.parquet": (ROW, ANCHOR, LOWER, UPPER), "predict.parquet": (ROW, ANCHOR)},
    "series": {"panel.parquet": (STEP, SERIES), "scaffold.parquet": (STEP, SERIES)},
}

# days per unit, for censoring bounds in the target's own unit
UNIT_DAYS = {"day": 1, "week": 7, "month": 30.4375, "quarter": 91.3125, "year": 365.25}


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Column:
    name: str
    role: Role
    # series mode: how a measure or target rolls up to one step
    aggregation: Aggregation | None = None
    # row mode: whether the value exists when a prediction is made. a column set later than the
    # anchor (a value recorded only once the outcome is known) must never become an input
    known: bool = True


@dataclass(frozen=True)
class Censor:
    # an open row still says "at least this much": lower = (as_of - anchor) in unit, upper = inf
    anchor: str
    unit: Step = "week"
    # the export date; empty means the latest anchor value in the file
    as_of: str | None = None


@dataclass(frozen=True)
class PrepSpec:
    source: str
    mode: Mode
    task: Task
    columns: tuple[Column, ...]
    encoding: str = "auto"
    # series mode: the step every series is aggregated to; empty means infer it from the data
    step: Step | None = None
    horizon: int = 8
    # extra sql filters on top of the role-driven ones, applied before anything else
    where: str | None = None
    predict_where: str | None = None
    censor: Censor | None = None
    # series mode: the export date; steps from the one holding it onward are scheduled, not history
    as_of: str | None = None

    def __post_init__(self):
        roles = [c.role for c in self.columns]
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise SpecError("a column appears twice in the spec")
        if "target" not in roles:
            raise SpecError("the spec names no target column")
        clock = "time" if self.mode == "series" else "anchor"
        if roles.count(clock) != 1:
            raise SpecError(f"{self.mode} mode needs exactly one {clock} column")
        if self.mode == "series":
            for col in self.columns:
                if col.role in ("measure", "target") and col.aggregation is None:
                    raise SpecError(f"{col.name!r} needs an aggregation in series mode")
        if self.censor and self.censor.anchor not in names:
            raise SpecError(f"censor anchor {self.censor.anchor!r} is not a column")
        if self.horizon < 1:
            raise SpecError("horizon must be at least one step")

    def named(self, role: Role) -> list[str]:
        return [c.name for c in self.columns if c.role == role]

    def inputs(self) -> list[str]:
        """columns a model may read: dimensions and measures known when predicting"""
        return [c.name for c in self.columns if c.role in ("dimension", "measure") and c.known]

    def digest(self) -> str:
        """the spec half of the cache key; prep adds the source file's content hash"""
        text = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def spec_from_dict(data: dict) -> PrepSpec:
    """a spec from its json form, the shape the data screen posts"""
    columns = tuple(Column(**c) for c in data.get("columns", ()))
    censor = Censor(**data["censor"]) if data.get("censor") else None
    rest = {k: v for k, v in data.items() if k not in ("columns", "censor")}
    return PrepSpec(columns=columns, censor=censor, **rest)
