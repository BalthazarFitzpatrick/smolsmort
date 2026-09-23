"""synthetic sources for the forecast tests, with the answers they were built from.
every test uses these; no real data ever enters the repo"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BRANCHES = {"north": 4.0, "south": 7.0, "east": 10.0, "west": 5.5, "large": 14.0}
AS_OF = dt.date(2026, 6, 29)


@dataclass
class RowTruth:
    path: Path
    rows: int
    open_rows: int
    leak: str
    base_by_branch: dict


def write_rows(folder: Path, n: int = 3000, seed: int = 0) -> RowTruth:
    """items over 3 years; lead weeks = branch base + size + season + planned offset + noise.
    lines whose fixed date falls after AS_OF are open, which censors the slow ones by construction"""
    rng = np.random.default_rng(seed)
    start = AS_OF - dt.timedelta(weeks=156)
    names = list(BRANCHES)
    path = folder / "rows.csv"
    open_rows = 0
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            ["line", "created", "branch", "size", "planned_offset", "lead_weeks", "leak_plus1"]
        )
        for line in range(n):
            created = start + dt.timedelta(days=int(rng.integers(0, 156 * 7)))
            branch = names[int(rng.integers(0, len(names)))]
            size = float(np.round(rng.lognormal(2.0, 0.8), 1))
            # a planned start exists for about 60% of lines, like the placeholder-ridden real field
            planned = float(np.round(rng.normal(4, 3), 1)) if rng.random() < 0.6 else None
            season = 3.0 * np.sin(2 * np.pi * created.timetuple().tm_yday / 365.25)
            lead = BRANCHES[branch] + 0.08 * size + season + 0.6 * (planned or 0.0)
            lead = max(0.2, lead + rng.gamma(2.0, 1.5) - 3.0)
            fixed = created + dt.timedelta(weeks=lead)
            if fixed > AS_OF:
                open_rows += 1
                target, leak = "", ""
            else:
                target, leak = f"{lead:.1f}", f"{lead + 1:.1f}"
            out.writerow(
                [line, created.isoformat(), branch, size, "" if planned is None else planned]
                + [target, leak]
            )
    return RowTruth(path, n, open_rows, "leak_plus1", dict(BRANCHES))


@dataclass
class PanelTruth:
    path: Path
    series: int
    weeks: int
    lead_lag: int
    period: int
    ended: str
    ended_week: int
    sparse: str
    gap: tuple[str, int, int]


def write_panel(folder: Path, weeks: int = 156, seed: int = 0) -> PanelTruth:
    """3 projects x 4 products, weekly; `orders` leads `units` by 6 weeks; a yearly season;
    one product 40% zeros; project C ends at week 120; a 3-week hole in one series.
    two transaction rows per series and week, so prep's aggregation is exercised"""
    rng = np.random.default_rng(seed)
    start = dt.date(2023, 7, 3)  # a monday
    lag, period = 6, 52
    path = folder / "panel.csv"
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["day", "project", "product", "orders", "units"])
        for project in "ABC":
            for product in ("p1", "p2", "p3", "p4"):
                level = rng.uniform(20, 60)
                orders = level * (1 + 0.4 * np.sin(2 * np.pi * np.arange(weeks + lag) / period))
                orders = orders + rng.normal(0, 2, weeks + lag)
                for week in range(weeks):
                    if project == "C" and week >= 120:
                        break
                    if (project, product) == ("A", "p2") and 70 <= week < 73:
                        continue
                    units = 0.8 * orders[week] + rng.normal(0, 1.5)
                    if product == "p4" and rng.random() < 0.4:
                        units = 0.0
                    for half, share in ((0, 0.5), (3, 0.5)):
                        day = start + dt.timedelta(weeks=week, days=half)
                        # orders known now drive units `lag` weeks later
                        row_orders = orders[week + lag] * share
                        out.writerow(
                            [day.isoformat(), project, product]
                            + [f"{row_orders:.2f}", f"{max(units, 0) * share:.2f}"]
                        )
    return PanelTruth(path, 12, weeks, lag, period, "C", 120, "p4", ("A|p2", 70, 73))
