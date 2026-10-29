"""input search: which input columns a run gets, decided before the model search runs.
a correlation screen ranks the candidate columns on training rows, short runs try weighted input
sets, the best sets get longer runs (halving), and the winner trains with all families until it
stops improving. rounds repeat until the deadline. pinned columns are in every set"""

from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from smolsmort.forecast.spec import SERIES, STEP, PrepSpec
from smolsmort.forecast.splits import timeseries_cuts
from smolsmort.forecast.tables import read_table

# one season per step; the screen compares each step with the same step a season earlier
SEASON = {"day": 7, "week": 52, "month": 12, "quarter": 4, "year": 1}
CANDIDATE_ROLES = ("measure", "input")
FAST_FAMILIES = ("lightgbm", "ridge", "xgboost")


@dataclass
class InputSearch:
    deadline: float  # epoch seconds
    explore_s: float = 360
    halve_s: tuple[float, ...] = (1200, 1800)
    halve_top: int = 4
    explore_share: float = 0.45
    explore_cap_s: float = 7200
    fast_families: tuple[str, ...] = FAST_FAMILIES
    final_families: tuple[str, ...] | None = None
    control: bool = True
    seed: int = 0
    run_options: dict = field(default_factory=dict)


def candidates(spec: PrepSpec) -> list[str]:
    return [c.name for c in spec.columns if c.role in CANDIDATE_ROLES]


def pinned(spec: PrepSpec) -> set[str]:
    return {c.name for c in spec.columns if c.pin}


def with_inputs(spec: PrepSpec, keep) -> PrepSpec:
    """the spec with only the kept candidates; time, target and dimensions stay"""
    keep = set(keep)
    columns = tuple(c for c in spec.columns if c.role not in CANDIDATE_ROLES or c.name in keep)
    return replace(spec, columns=columns)


def screen_inputs(spec: PrepSpec, folder: Path, *, top_levels: int = 6) -> list[dict]:
    """rank candidate columns by spearman of their value now with the deseasoned log target
    1, horizon/2 and horizon steps later; training rows only, mean over series"""
    panel = read_table(Path(folder) / "panel.parquet", order_by=f"{SERIES}, {STEP}")
    steps = np.unique(panel[STEP])
    cuts = timeseries_cuts(len(steps), spec.horizon)
    target = spec.named("target")[0]
    values = _candidate_values(spec, folder, panel, top_levels)
    leads = sorted({1, max(1, spec.horizon // 2), spec.horizon})
    season = SEASON.get(spec.step or "week", 1)
    train_end = steps[cuts.val_start]
    rho: dict[str, dict[int, list[float]]] = {name: {lead: [] for lead in leads} for name in values}
    for key in np.unique(panel[SERIES]):
        rows = (panel[SERIES] == key) & (panel[STEP] < train_end)
        y = np.log1p(np.clip(np.nan_to_num(panel[target][rows].astype(float)), 0, None))
        lag = season if len(y) > season + spec.horizon + 20 else 1
        change = np.full(len(y), np.nan)
        change[lag:] = y[lag:] - y[:-lag]
        for name, levels in values.items():
            for lead in leads:
                future = np.full(len(y), np.nan)
                future[:-lead] = change[lead:]
                best = 0.0
                for x in levels:
                    r = _spearman(x[rows], future)
                    best = r if abs(r) > abs(best) else best
                rho[name][lead].append(best)
    pins = pinned(spec)
    ranking = []
    for name in values:
        by_lead = {lead: float(np.mean(v)) if v else 0.0 for lead, v in rho[name].items()}
        score = float(np.mean([abs(v) for v in by_lead.values()]))
        ranking.append({"column": name, "score": score, "rho": by_lead, "pinned": name in pins})
    ranking.sort(key=lambda entry: (-entry["score"], entry["column"]))
    return ranking


def _candidate_values(spec, folder, panel, top_levels) -> dict[str, list[np.ndarray]]:
    """per candidate, one or more panel-aligned arrays: a measure's value, an input's top shares"""
    values = {c: [panel[c].astype(float)] for c in spec.named("measure")}
    inputs = spec.named("input")
    if not inputs:
        return values
    table = read_table(Path(folder) / "inputs.parquet")
    index = {(s, t): i for i, (s, t) in enumerate(zip(panel[SERIES], panel[STEP], strict=True))}
    for name in inputs:
        rows = table["__column"] == name
        series, step, level = table[SERIES][rows], table[STEP][rows], table["__level"][rows]
        counts = table["rows"][rows].astype(float)
        totals: dict[tuple, float] = {}
        for s, t, n in zip(series, step, counts, strict=True):
            totals[(s, t)] = totals.get((s, t), 0.0) + n
        names, weights = np.unique(level, return_counts=True)
        top = names[np.argsort(-weights, kind="stable")][:top_levels]
        seen = np.array([(s, t) in totals for s, t in zip(panel[SERIES], panel[STEP], strict=True)])
        arrays = []
        for wanted in top:
            share = np.full(len(panel[STEP]), np.nan)
            pick = level == wanted
            for s, t, n in zip(series[pick], step[pick], counts[pick], strict=True):
                at = index.get((s, t))
                if at is not None:
                    share[at] = n / totals[(s, t)]
            # a step without the level holds a zero share, not a gap
            share[seen & np.isnan(share)] = 0.0
            arrays.append(share)
        values[name] = arrays
    return values


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20 or np.unique(x[ok]).size < 3:
        return 0.0
    rx, ry = _ranks(x[ok]), _ranks(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    # ties share their mean rank
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return sums[inverse] / counts[inverse]


def input_odds(ranking: list[dict]) -> dict[str, float]:
    """inclusion odds 0.15-0.85 scaled to the best score; pinned columns 1"""
    top = max((entry["score"] for entry in ranking), default=0.0)
    return {
        entry["column"]: 1.0
        if entry["pinned"]
        else (0.15 + 0.7 * entry["score"] / top if top > 0 else 0.5)
        for entry in ranking
    }


def draw_set(odds, rng, seen, parents=()) -> tuple[str, ...] | None:
    """an unseen input set: a fresh weighted draw, or one or two weighted flips of a parent.
    pinned columns (odds 1) are never dropped; none when nothing new turns up"""
    pins = {name for name, p in odds.items() if p >= 1}
    flippable = {name: p for name, p in odds.items() if p < 1}
    for _ in range(200):
        if parents and flippable and rng.random() < 0.6:
            genes = set(rng.choice(list(parents)))
            names = list(flippable)
            for name in rng.choices(
                names, weights=[flippable[n] for n in names], k=rng.randint(1, 2)
            ):
                genes ^= {name}
        else:
            genes = {name for name, p in flippable.items() if rng.random() < p}
        genes = tuple(sorted(genes | pins))
        if genes not in seen:
            return genes
    return None


def search_inputs(
    runs_root: Path,
    cache_root: Path,
    spec: PrepSpec,
    plan: InputSearch,
    *,
    trial: Callable | None = None,
    clock: Callable[[], float] = time.time,
) -> Path:
    """run the whole input search; writes <runs_root>/input-<id>/search.json after every trial
    and returns that folder. `trial(spec, families, budget_s, until_plateau)` -> result row"""
    from smolsmort.forecast.prep import prepare

    folder = Path(runs_root) / f"input-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True)
    trial = trial or _run_trial(runs_root, cache_root, plan.run_options)
    ranking = screen_inputs(spec, prepare(spec, cache_root).folder)
    odds = input_odds(ranking)
    state = {"spec": spec.digest(), "screen": ranking, "odds": odds, "trials": []}

    def left() -> float:
        return plan.deadline - clock()

    def run(genes, families, budget_s, stage, until_plateau=False) -> dict:
        row = trial(with_inputs(spec, genes), list(families), budget_s, until_plateau)
        row = {"stage": stage, "inputs": list(genes), "budget_s": budget_s, **row}
        state["trials"].append(row)
        _save(folder, state)
        return row

    rng = random.Random(plan.seed)
    seen: set[tuple[str, ...]] = set()
    finals = list(plan.final_families or ())
    rounds = 0
    _save(folder, state)
    while left() > plan.explore_s + 600:
        rounds += 1
        explore_end = clock() + min(plan.explore_cap_s, left() * plan.explore_share)
        rows = []
        if rounds == 1:
            if plan.control:
                seen.add(())
                run((), plan.fast_families, plan.explore_s, "control")
            core = tuple(sorted(pinned(spec)))
            if core and core not in seen:
                seen.add(core)
                rows.append(run(core, plan.fast_families, plan.explore_s, "explore1"))
        while clock() < explore_end and left() > plan.explore_s + 300:
            parents = [tuple(r["inputs"]) for r in _ranked(rows)[: plan.halve_top]]
            genes = draw_set(odds, rng, seen, parents)
            if genes is None:
                break
            seen.add(genes)
            rows.append(run(genes, plan.fast_families, plan.explore_s, f"explore{rounds}"))
        pool = [tuple(r["inputs"]) for r in _ranked(rows)[: plan.halve_top]]
        for budget_s in plan.halve_s:
            if len(pool) < 2 or left() < len(pool) * budget_s + 1800:
                break
            done = [run(g, plan.fast_families, budget_s, f"halve{rounds}") for g in pool]
            pool = [tuple(r["inputs"]) for r in _ranked(done)[: len(pool) // 2]]
        if not pool or left() < 1200:
            break
        run(pool[0], finals or None, left() - 600, f"final{rounds}", until_plateau=True)
    state["best"] = (
        _ranked([t for t in state["trials"] if t["stage"].startswith("final")])
        or _ranked(state["trials"])
        or [None]
    )[0]
    _save(folder, state)
    return folder


def _ranked(rows: list[dict]) -> list[dict]:
    return sorted((r for r in rows if r.get("val") is not None), key=lambda r: r["val"])


def _save(folder: Path, state: dict) -> None:
    (folder / "search.json").write_text(json.dumps(state, indent=1, default=str))


def _run_trial(runs_root: Path, cache_root: Path, options: dict) -> Callable:
    """one forecast run per trial; short trials use the whole slice, the final stops at plateau"""
    from smolsmort.forecast.prep import prepare
    from smolsmort.forecast.runs import RunError, start_run, wait_run

    def trial(spec, families, budget_s, until_plateau) -> dict:
        prepared = prepare(spec, cache_root)
        chosen = families or options.get("families") or list(FAST_FAMILIES)
        run_id = start_run(
            runs_root,
            spec,
            prepared,
            version=2,
            families=[{"name": n, "method": "genetic"} for n in chosen],
            time_budget_s=budget_s,
            transform=options.get("transform", "none"),
            plateau=options.get("plateau", 10),
            min_generations=3 if until_plateau else 1_000_000,
            prune_sigma=options.get("prune_sigma", 2.0),
            max_concurrent_families=options.get("max_concurrent_families"),
            holdout=options.get("holdout", "block"),
        )
        row = {"run": run_id}
        try:
            state = wait_run(runs_root, run_id, timeout=budget_s * 1.5 + 900)
            row["state"] = state["state"]
        except RunError as exc:
            row["error"] = str(exc)
            return row
        result = Path(runs_root) / run_id / "result.json"
        if result.exists():
            data = json.loads(result.read_text())
            accuracy = data.get("verdict", {}).get("accuracy") or {}
            row.update(
                val=accuracy.get("val_error"),
                test=accuracy.get("model_error"),
                ratio=accuracy.get("ratio"),
                winner=(data.get("winner") or {}).get("family"),
            )
        return row

    return trial
