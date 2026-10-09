"""grid-seeded genetic search over feature families, objective and tree settings. it stops on a
plateau or at the time cap, and only ever reads validation - the test split waits for the winner"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.families import get_family
from smolsmort.forecast.model import ModelError
from smolsmort.forecast.pipeline import (
    Genome,
    PipelineError,
    Workspace,
    genome_from_dict,
    make_genome,
    score,
)

# budget defaults: the design review's population and plateau, a 20 minute cap - unmeasured
POPULATION = 16
ELITE = 3
TOURNAMENT = 3
PLATEAU = 5
TIME_CAP = 20 * 60
MAX_GENERATIONS = 60
# above this many training rows, new candidates are first scored on a sample
HALVING_ROWS = 50_000
SAMPLE_SHARE = 0.2

DEFAULTS = {
    "max_depth": 6,
    "eta": 0.1,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
}
SPACE = {
    "max_depth": (3, 10, "int"),
    "eta": (0.03, 0.3, "log"),
    "min_child_weight": (1.0, 100.0, "log"),
    "subsample": (0.5, 1.0, "lin"),
    "colsample_bytree": (0.5, 1.0, "lin"),
    "lambda": (0.1, 10.0, "log"),
}
OBJECTIVE_SPACE = {
    "tweedie": {"tweedie_variance_power": (1.1, 1.9, "lin")},
    # logistic diverged in U5 (2026-09-23), so only these two
    "aft": {
        "aft_loss_distribution": ("normal", "extreme"),
        "aft_loss_distribution_scale": (0.3, 1.2, "lin"),
    },
}
OBJECTIVE_DEFAULTS = {
    "tweedie": {"tweedie_variance_power": 1.5},
    "aft": {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.8},
}


@dataclass
class Budget:
    population: int = POPULATION
    plateau: int = PLATEAU
    time_cap: float = TIME_CAP
    max_generations: int = MAX_GENERATIONS
    seed: int = 0
    nthread: int = 1


@dataclass
class Entry:
    genome: Genome
    fitness: float
    contrib: np.ndarray | None
    metrics: dict
    generation: int
    note: str = ""
    eval_history: dict | None = None

    def rank(self):
        # ties go to the smaller recipe
        return (self.fitness, len(self.genome.families))

    def to_dict(self) -> dict:
        return {
            "key": self.genome.key(),
            "genome": self.genome.to_dict(),
            "fitness": self.fitness if np.isfinite(self.fitness) else None,
            "metrics": self.metrics,
            "generation": self.generation,
            "families": len(self.genome.families),
            "note": self.note,
        }


class _Variation:
    def __init__(self, ws: Workspace, rng: np.random.Generator):
        self.families = list(ws.families)
        self.objectives = list(ws.profile["genes"]["objectives"])
        self.rng = rng
        family = get_family("xgboost")
        task = getattr(getattr(ws, "spec", None), "task", "regression")
        self.objectives = [name for name in self.objectives if name in family.objectives(task)]
        catalog = {param.name: param for param in family.space()}
        self.space = {}
        self.defaults = {}
        for name in SPACE:
            param = catalog[name]
            self.space[name] = (param.lo, param.hi, "lin" if param.kind == "float" else param.kind)
            self.defaults[name] = param.default

    def params_for(self, objective: str, base: dict | None = None) -> dict:
        params = {k: v for k, v in (base or self.defaults).items() if k in self.space}
        params.update(OBJECTIVE_DEFAULTS.get(objective, {}))
        return params

    def random(self) -> Genome:
        keep = [f for f in self.families if self.rng.random() < 0.7] or [
            self.rng.choice(self.families)
        ]
        objective = str(self.rng.choice(self.objectives))
        params = {name: self._draw(spec) for name, spec in self.space.items()}
        params.update({n: self._draw(s) for n, s in OBJECTIVE_SPACE.get(objective, {}).items()})
        return make_genome(keep, objective, params)

    def crossover(self, a: Genome, b: Genome) -> Genome:
        """whole blocks swap: each family from either parent, the objective with its own settings
        from one parent, each tree setting from either"""
        mine, theirs = set(a.families), set(b.families)
        families = [
            f for f in self.families if (f in mine if self.rng.random() < 0.5 else f in theirs)
        ]
        donor = a if self.rng.random() < 0.5 else b
        pa, pb, pd = dict(a.params), dict(b.params), dict(donor.params)
        params = {
            k: (pa if self.rng.random() < 0.5 else pb).get(k, self.defaults[k]) for k in self.space
        }
        params.update({k: pd[k] for k in OBJECTIVE_SPACE.get(donor.objective, {}) if k in pd})
        return make_genome(
            families or list(donor.families), donor.objective, params, family=donor.family
        )

    def mutate(self, genome: Genome) -> Genome:
        families = set(genome.families)
        rate = 1 / max(len(self.families), 1)
        for f in self.families:
            if self.rng.random() < rate:
                families ^= {f}
        objective = genome.objective
        params = dict(genome.params)
        if len(self.objectives) > 1 and self.rng.random() < 0.1:
            objective = str(self.rng.choice([o for o in self.objectives if o != objective]))
            params = self.params_for(objective, params)
        for name, spec in {**self.space, **OBJECTIVE_SPACE.get(objective, {})}.items():
            if self.rng.random() < 0.3:
                params[name] = self._nudge(params.get(name), spec)
        return make_genome(
            families or [self.rng.choice(self.families)], objective, params, family=genome.family
        )

    def _draw(self, spec):
        if isinstance(spec[0], str):
            return str(self.rng.choice(spec))
        lo, hi, kind = spec
        if kind == "log":
            return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
        if kind == "int":
            return int(self.rng.integers(lo, hi + 1))
        return float(self.rng.uniform(lo, hi))

    def _nudge(self, value, spec):
        if value is None or isinstance(spec[0], str):
            return self._draw(spec)
        lo, hi, kind = spec
        if kind == "int":
            return int(np.clip(value + self.rng.choice([-1, 1]), lo, hi))
        if kind == "log":
            return float(np.clip(value * np.exp(self.rng.normal(0, 0.3)), lo, hi))
        return float(np.clip(value + self.rng.normal(0, 0.1 * (hi - lo)), lo, hi))


class _FamilyVariation(_Variation):
    def __init__(self, ws, rng, family, overrides):
        super().__init__(ws, rng)
        self.family = family
        self.space = {}
        self.defaults = {}
        self.choices = set()
        for param in family.space():
            self.space[param.name] = (
                param.choices if param.kind == "choice" else (param.lo, param.hi, param.kind)
            )
            self.defaults[param.name] = param.default
            if param.kind == "choice":
                self.choices.add(param.name)
        for name, values in (overrides or {}).items():
            if name not in self.space or not isinstance(values, list) or not values:
                raise ValueError(f"invalid space override {name!r}")
            self.space[name] = tuple(values)
            self.choices.add(name)
            if self.defaults[name] not in values:
                self.defaults[name] = values[0]
        self.overrides = overrides or {}
        objectives = getattr(family, "objectives", lambda _task: ("squared",))
        self.objectives = list(objectives(ws.spec.task))
        if family.name == "xgboost":
            self.objectives = [
                name for name in ws.profile["genes"]["objectives"] if name in self.objectives
            ]
        if family.needs == "raw_series":
            self.families = []

    def params_for(self, objective, base=None):
        params = {**self.defaults, **(base or {})}
        for name, values in self.overrides.items():
            if params.get(name) not in values:
                params[name] = self.defaults[name]
        if self.family.name == "xgboost":
            active = {**SPACE, **OBJECTIVE_SPACE.get(objective, {})}
            params = {name: value for name, value in params.items() if name in active}
        return params

    def random(self):
        objective = str(self.rng.choice(self.objectives))
        keep = [name for name in self.families if self.rng.random() < 0.7]
        if self.families and not keep:
            keep = [str(self.rng.choice(self.families))]
        params = {
            name: self._draw(spec) if name not in self.choices else self.rng.choice(spec).item()
            for name, spec in self.space.items()
        }
        return make_genome(keep, objective, self.params_for(objective, params), self.family.name)

    def crossover(self, a, b):
        pa, pb = dict(a.params), dict(b.params)
        params = {
            name: (pa if self.rng.random() < 0.5 else pb).get(name, self.defaults[name])
            for name in self.space
        }
        params = self.params_for(a.objective, params)
        mine, theirs = set(a.families), set(b.families)
        keep = [
            name for name in self.families if name in (mine if self.rng.random() < 0.5 else theirs)
        ]
        return make_genome(keep or a.families, a.objective, params, self.family.name)

    def mutate(self, genome):
        params = dict(genome.params)
        keep = set(genome.families)
        for name in self.families:
            if self.rng.random() < 1 / len(self.families):
                keep ^= {name}
        objective = genome.objective
        if len(self.objectives) > 1 and self.rng.random() < 0.1:
            objective = str(
                self.rng.choice([name for name in self.objectives if name != objective])
            )
        for name, spec in self.space.items():
            if self.rng.random() < 0.3:
                params[name] = (
                    self.rng.choice(spec).item()
                    if name in self.choices
                    else self._nudge(params.get(name), spec)
                )
        return make_genome(
            keep or genome.families,
            objective,
            self.params_for(objective, params),
            self.family.name,
        )


def _family_population(var, size, warm, method):
    if not var.space:
        return [make_genome(var.families, var.objectives[0], {}, var.family.name)]
    if method == "grid":
        choices = [var.overrides.get(name, [var.defaults[name]]) for name in var.space]
        return [
            make_genome(
                var.families,
                objective,
                var.params_for(objective, dict(zip(var.space, values, strict=True))),
                var.family.name,
            )
            for objective in var.objectives
            for values in itertools.product(*choices)
        ]
    seeds = [
        genome_from_dict(data)
        for data in warm or []
        if data.get("family", "xgboost") == var.family.name
    ]
    seeds.extend(
        make_genome(var.families, o, var.params_for(o), var.family.name) for o in var.objectives
    )
    while len(seeds) < size:
        seeds.append(var.random())
    return seeds[:size]


def _seed_population(ws, var: _Variation, size: int, warm) -> list[Genome]:
    everything = list(ws.families)
    seeds = [make_genome(everything, o, var.params_for(o)) for o in var.objectives]
    first = var.objectives[0]
    for depth in (3, 8):
        for eta in (0.05, 0.2):
            params = {**var.params_for(first), "max_depth": depth, "eta": eta}
            seeds.append(make_genome(everything, first, params))
    for data in warm or []:
        families = [f for f in data["families"] if f in ws.families]
        if families and data["objective"] in var.objectives:
            seeds.insert(
                0,
                make_genome(
                    families,
                    data["objective"],
                    data["params"],
                    family=data.get("family", "xgboost"),
                ),
            )
    unique = list({g.key(): g for g in seeds}.values())
    while len(unique) < size:
        genome = var.random()
        if genome.key() not in {g.key() for g in unique}:
            unique.append(genome)
    return unique[:size]


def _sample(ws, rng) -> np.ndarray | None:
    """training rows (or series) to score new candidates on, when the data is large"""
    if ws.mode == "row":
        train = ws.masks["train"]
        if train.sum() <= HALVING_ROWS:
            return None
        return train & (rng.random(len(train)) < SAMPLE_SHARE)
    rows = sum(int(m["train"].sum()) for m in ws.series_masks.values())
    if rows <= HALVING_ROWS:
        return None
    keys = np.unique(next(iter(ws.frames.values())).series)
    return keys[rng.random(len(keys)) < SAMPLE_SHARE]


def _significant(new: Entry, best: Entry) -> bool:
    """a real gain: the paired per-period difference beats its own standard error"""
    if best is None or not np.isfinite(best.fitness):
        return bool(np.isfinite(new.fitness))
    if new.fitness >= best.fitness or new.contrib is None or best.contrib is None:
        return False
    diff = best.contrib - new.contrib
    if len(diff) < 2:
        return True
    return bool(diff.sum() > diff.std(ddof=1) * np.sqrt(len(diff)))


def search(
    ws: Workspace,
    budget: Budget,
    *,
    warm: list[dict] | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_best: Callable[[dict | None], None] | None = None,
    on_generation: Callable[[dict], None] | None = None,
    on_leaderboard: Callable[[int, list[dict]], None] | None = None,
    stop: Callable[[], bool] | None = None,
    family=None,
    method: str = "genetic",
    space: dict | None = None,
    resume: dict | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    """run until a plateau, the time cap, the generation cap or `stop()`; returns the ranked
    leaderboard, the final population (the next search's warm start) and why it stopped"""
    rng = np.random.default_rng(budget.seed)
    if isinstance(family, str):
        family = get_family(family)
    legacy = family is None or (family.name == "xgboost" and not space and method == "genetic")
    var = _Variation(ws, rng) if legacy else _FamilyVariation(ws, rng, family, space)
    started = time.monotonic()
    sample = None if family and not family.space() else _sample(ws, rng)
    scored: dict[str, Entry] = {}
    prior_elapsed = (resume or {}).get("elapsed", 0)

    def checkpoint():
        if on_checkpoint:
            on_checkpoint(
                {
                    "population": [g.to_dict() for g in population],
                    "entries": [
                        {**e.to_dict(), "contrib": e.contrib, "eval_history": e.eval_history}
                        for e in scored.values()
                    ],
                    "generation": generation,
                    "quiet": quiet,
                    "best_key": best.genome.key() if best else None,
                    "rng": rng.bit_generator.state,
                    "elapsed": prior_elapsed + time.monotonic() - started,
                }
            )

    def evaluate(genome: Genome, generation: int, rows=None) -> Entry:
        cached = scored.get(genome.key())
        # a full score always serves; a sample score only serves another sample request
        if cached and (cached.note != "sample" or rows is not None):
            return cached
        try:
            result = score(ws, genome, nthread=budget.nthread, sample=rows)
            entry = Entry(
                genome,
                result.fitness,
                result.contrib,
                result.metrics,
                generation,
                "sample" if rows is not None else "",
                result.eval_history,
            )
        except (ModelError, PipelineError) as exc:
            entry = Entry(genome, float("inf"), None, {}, generation, f"failed: {exc}")
        scored[genome.key()] = entry
        checkpoint()
        return entry

    if family is not None:
        warm = [data for data in warm or [] if data.get("family", "xgboost") == family.name]
    population = (
        _seed_population(ws, var, budget.population, warm)
        if legacy
        else _family_population(var, budget.population, warm, method)
    )
    best, quiet, generation, reason = None, 0, 0, "generation cap"
    if resume:
        population = [genome_from_dict(data) for data in resume["population"]]
        for data in resume["entries"]:
            entry = Entry(
                genome_from_dict(data["genome"]),
                data["fitness"] if data["fitness"] is not None else float("inf"),
                np.array(data["contrib"]) if data["contrib"] is not None else None,
                data["metrics"],
                data["generation"],
                data["note"],
                data.get("eval_history"),
            )
            scored[entry.genome.key()] = entry
        best = scored.get(resume.get("best_key"))
        quiet, generation = resume["quiet"], resume["generation"]
        rng.bit_generator.state = resume["rng"]

    def record_progress(candidate: int):
        if on_event:
            on_event(
                {
                    "event": "progress",
                    "generation": generation,
                    "candidate": candidate,
                    "population": len(population),
                    "flat_generations": quiet,
                    "elapsed": round(prior_elapsed + time.monotonic() - started, 1),
                }
            )

    while generation < budget.max_generations:
        entries = []
        for candidate, genome in enumerate(population, 1):
            if family and scored and prior_elapsed + time.monotonic() - started >= budget.time_cap:
                reason = "time cap"
                break
            entries.append(evaluate(genome, generation, sample))
            record_progress(candidate)
            if stop and stop():
                break
        if not entries:
            break
        if sample is not None and not (stop and stop()):
            # successive halving: the best third is rescored on every training row
            entries.sort(key=Entry.rank)
            top = len(entries) // 3 or 1
            for index in range(top):
                entries[index] = evaluate(entries[index].genome, generation)
                record_progress(len(entries))
                if stop and stop():
                    break
        entries.sort(key=Entry.rank)
        full = [e for e in entries if e.note != "sample"]
        leader = full[0] if full else entries[0]
        gained = _significant(leader, best)
        if best is None or leader.rank() < best.rank():
            best = leader
            if on_best:
                history = best.eval_history
                on_best({**history, "key": best.genome.key()} if history else None)
        history = leader.eval_history
        if on_generation and history and history.get("validation"):
            on_generation(
                {
                    "generation": generation,
                    "key": leader.genome.key(),
                    "metric": history["metric"],
                    "bucket": history.get("bucket"),
                    "validation": history["validation"],
                }
            )
        for entry in scored.values():
            if entry is not best and entry.eval_history:
                # cached candidates can lead later generations; retain only their validation curve
                entry.eval_history = {
                    key: value for key, value in entry.eval_history.items() if key != "training"
                }
        if not (stop and stop()):
            quiet = 0 if gained else quiet + 1
        elapsed = prior_elapsed + time.monotonic() - started
        if on_leaderboard:
            board = sorted((e for e in scored.values() if e.note != "sample"), key=Entry.rank)
            on_leaderboard(generation, [entry.to_dict() for entry in board[:10]])
        if on_event:
            on_event(
                {
                    "event": "generation",
                    "generation": generation,
                    "best": best.fitness,
                    "metrics": best.metrics,
                    "evaluated": len(scored),
                    "gain": gained,
                    "flat_generations": quiet,
                    "elapsed": round(elapsed, 1),
                }
            )
        generation += 1
        if not legacy and (not var.space or method == "grid"):
            if reason != "time cap":
                reason = "single candidate" if not var.space else "grid complete"
            break
        if stop and stop():
            reason = "cancelled"
            break
        if quiet >= budget.plateau:
            reason = "plateau"
            break
        if elapsed >= budget.time_cap:
            reason = "time cap"
            break
        elite = [e.genome for e in entries[:ELITE]]
        children = []
        keys = {g.key() for g in elite}
        while len(elite) + len(children) < budget.population:
            a, b = (
                min(rng.choice(len(entries), TOURNAMENT), key=lambda i: entries[i].rank())
                for _ in range(2)
            )
            child = var.mutate(var.crossover(entries[a].genome, entries[b].genome))
            for _ in range(5):
                if child.key() not in keys:
                    break
                child = var.mutate(child)
            keys.add(child.key())
            children.append(child)
        population = elite + children
        checkpoint()
    board = sorted((e for e in scored.values() if e.note != "sample"), key=Entry.rank)
    if (resume or family) and board and (best is None or board[0].rank() < best.rank()):
        best = board[0]
        if on_best:
            history = best.eval_history
            on_best({**history, "key": best.genome.key()} if history else None)
    return {
        "leaderboard": [e.to_dict() for e in board[:10]],
        "best": best.genome.to_dict() if best else None,
        "population": [g.to_dict() for g in population],
        "generations": generation,
        "stopped": reason,
        "elapsed": round(prior_elapsed + time.monotonic() - started, 1),
    }
