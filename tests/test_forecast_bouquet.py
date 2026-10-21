"""screened family sampling, worker wiring and compatibility without model fits"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast import family_worker, worker
from smolsmort.forecast.families import get_family
from smolsmort.forecast.pipeline import Scored, make_genome
from smolsmort.forecast.screen import read_priors, read_top_families
from smolsmort.forecast.search import (
    Budget,
    _family_population,
    _FamilyVariation,
    _seed_population,
    _Variation,
    search,
)

PRIORS = {"a": 0.9, "b": 0.1, "c": 0.7, "d": 0.2, "e": 0.4}
TOP = {"a", "c", "e"}


def make_workspace(families=None, mode="series"):
    return SimpleNamespace(
        mode=mode,
        unit="week",
        spec=SimpleNamespace(task="regression"),
        families=families or ["lag+0", "season", "roll:4:mean", "diff:1", "measure:orders"],
        profile={"genes": {"objectives": ["squared", "absolute"]}},
        series_masks={"one": {"train": np.ones(20, dtype=bool)}},
        masks={"train": np.ones(20, dtype=bool)},
        excluded_series=[],
    )


def make_variation(ws, family, priors=None):
    rng = np.random.default_rng(17)
    return (
        _Variation(ws, rng, priors)
        if family is None
        else _FamilyVariation(ws, rng, get_family(family), {}, priors)
    )


def seed_population(ws, var, family, size=16, warm=None):
    return (
        _seed_population(ws, var, size, warm)
        if family is None
        else _family_population(var, size, warm, "genetic")
    )


@pytest.mark.parametrize("family", [None, "xgboost", "lightgbm", "ridge"])
def test_birth_frequencies_follow_each_prior(family):
    priors = {"a": 0.9, "b": 0.1}
    var = make_variation(make_workspace(list(priors)), family, priors)
    births = [var.random() for _ in range(10_000)]
    counts = Counter(name for genome in births for name in genome.families)
    frequencies = {name: counts[name] / len(births) for name in priors}
    print(f"birth frequencies {family or 'legacy'}: {frequencies}")
    assert frequencies == pytest.approx(priors, abs=0.02)
    assert any(not genome.families for genome in births)


@pytest.mark.parametrize("family", [None, "lightgbm", "ridge"])
@pytest.mark.parametrize("present", [False, True])
def test_mutation_weights_both_directions_and_keeps_one_expected_flip(family, present):
    priors = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.5}
    var = make_variation(make_workspace(list(priors)), family, priors)
    original = set(priors) if present else set()
    genome = var.make_genome(
        original, var.objectives[0], var.params_for(var.objectives[0]), family or "xgboost"
    )
    counts = Counter(
        name for _ in range(4_000) for name in original ^ set(var.mutate(genome).families)
    )
    assert sum(counts.values()) / 4_000 == pytest.approx(1, abs=0.04)
    assert counts["a"] / 4_000 == pytest.approx(0.05 if present else 0.45, abs=0.025)
    assert counts["b"] / 4_000 == pytest.approx(0.45 if present else 0.05, abs=0.025)


# captured from a1466e2 before editing search: seeds, births, crossover and mutation
@pytest.mark.parametrize(
    ("mode", "family", "expected"),
    [
        ("series", None, "f712266522fe00247f8e3c17d874509be68ea263420eb39717d3fb212049d16b"),
        ("series", "lightgbm", "019635982264c7aa8cd6f0078b85a46438a52712fdbeb24ade13cf4e10d9d54e"),
        ("series", "ridge", "3e2d7d2abd99b552d8ac70786d62685bd3a7b5a7d0770e53c1bcb91c1b054efd"),
        ("row", None, "e06a1000436afc7a1479856f418a67c8cbc7a325e23309e6a25e39cdb3cab075"),
        ("row", "lightgbm", "243e5da70575bea66b1caf968eefb6224ad0071e1a36a7ac4834ab8c30861fd8"),
        ("row", "ridge", "251df78a866865f54afdfa4f130c0b7c8cb36bc9167b2222b7bbfa8366d9111f"),
    ],
)
def test_missing_screen_preserves_original_genomes(tmp_path, mode, family, expected):
    ws = make_workspace(mode=mode)
    priors = read_priors(tmp_path)
    assert priors is None
    var = make_variation(ws, family, priors)
    seeds = seed_population(ws, var, family)
    births = [var.random() for _ in range(30)]
    children = [
        var.mutate(var.crossover(a, b)) for a, b in zip(births[::2], births[1::2], strict=True)
    ]
    data = json.dumps([g.to_dict() for g in seeds + births + children], sort_keys=True)
    assert hashlib.sha256(data.encode()).hexdigest() == expected


@pytest.mark.parametrize("family", [None, "lightgbm", "ridge"])
def test_row_mode_ignores_priors(family):
    ws = make_workspace(mode="row")
    old = make_variation(ws, family)
    screened = make_variation(ws, family, dict.fromkeys(ws.families, 0.1))
    assert [old.random() for _ in range(30)] == [screened.random() for _ in range(30)]


@pytest.mark.parametrize("family", [None, "xgboost", "lightgbm", "ridge"])
@pytest.mark.parametrize("size", [1, 6])
def test_top_seed_survives_a_full_warm_population(family, size):
    ws = make_workspace(list(PRIORS))
    var = make_variation(ws, family, PRIORS)
    warm = [
        var.make_genome(["b"], var.objectives[0], {"eta": n / 100}, family or "xgboost").to_dict()
        for n in range(10)
    ]
    population = seed_population(ws, var, family, size, warm)
    assert len(population) == size
    assert set(population[0].families) == TOP
    if size > 1:
        assert population[1].to_dict() == warm[0 if family else -1]


@pytest.mark.parametrize(
    "probabilities", [[0.9, 0.8], [0.9, 0.4, 0.3, 0.2], [0.9, 0.8, 0.6, 0.5, 0.1]]
)
def test_top_seed_uses_threshold_with_a_minimum_of_three_available_families(probabilities):
    priors = dict(zip("abcde", probabilities, strict=False))
    var = make_variation(make_workspace(list(priors)), None, priors)
    expected = list(priors)[: max(3, sum(p >= 0.5 for p in probabilities))]
    assert var.birth_families == expected


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "ridge"])
@pytest.mark.parametrize("method", ["grid", "defaults"])
def test_grid_and_broad_defaults_use_the_top_set(monkeypatch, family, method):
    seen = []

    def score(_ws, genome, **kwargs):
        seen.append(genome)
        return Scored(1.0, np.array([1.0]), {})

    monkeypatch.setattr("smolsmort.forecast.search.score", score)
    search(
        make_workspace(list(PRIORS)),
        Budget(population=4, max_generations=1),
        family=family,
        priors=PRIORS,
        method="grid" if method == "grid" else "genetic",
        defaults_only=method == "defaults",
    )
    assert seen and all(set(genome.families) == TOP for genome in seen)
    if method == "defaults":
        assert len(seen) == 1


def test_raw_series_family_keeps_no_feature_genes():
    ws = make_workspace(list(PRIORS))
    var = make_variation(ws, "snaive", PRIORS)
    assert var.random().families == ()
    assert _family_population(var, 16, [], "genetic")[0].families == ()


def test_winner_and_live_board_count_families_from_top_set(monkeypatch):
    def score(_ws, genome, **kwargs):
        return Scored(float("b" not in genome.families), np.array([1.0]), {})

    monkeypatch.setattr("smolsmort.forecast.search.score", score)
    boards = []
    result = search(
        make_workspace(list(PRIORS)),
        Budget(population=2, max_generations=1),
        priors=PRIORS,
        warm=[make_genome(["b"], "squared", {}).to_dict()],
        on_leaderboard=lambda generation, board: boards.append(board),
    )
    assert result["best"]["families"] == ["b"]
    assert result["leaderboard"] == boards[0]
    assert result["leaderboard"][0]["screened_families"] == 0
    assert result["leaderboard"][1]["screened_families"] == 3


@pytest.mark.parametrize("family", [None, "lightgbm", "ridge"])
def test_checkpoint_restores_priors_and_exact_search_sequence(monkeypatch, family):
    monkeypatch.setattr(
        "smolsmort.forecast.search.score",
        lambda _ws, genome, **kwargs: Scored(float(len(genome.families)), np.array([1.0]), {}),
    )
    ws = make_workspace(list(PRIORS))
    budget = Budget(population=8, max_generations=3)
    expected = search(ws, budget, family=family, priors=PRIORS)
    checkpoints = []
    search(
        ws,
        Budget(population=8, max_generations=1),
        family=family,
        priors=PRIORS,
        on_checkpoint=lambda state: checkpoints.append(deepcopy(state)),
    )
    saved = json.loads(json.dumps(checkpoints[-1], default=worker._plain))
    assert saved["priors"] == PRIORS
    actual = search(ws, budget, family=family, resume=saved)
    for key in ("leaderboard", "best", "population", "generations", "stopped"):
        assert actual[key] == expected[key]


def test_old_checkpoint_entries_get_screen_counts_without_rescoring(monkeypatch):
    monkeypatch.setattr(
        "smolsmort.forecast.search.score",
        lambda _ws, genome, **kwargs: Scored(1.0, np.array([1.0]), {}),
    )
    ws = make_workspace()
    checkpoints = []
    before = search(
        ws,
        Budget(max_generations=1),
        on_checkpoint=lambda state: checkpoints.append(deepcopy(state)),
    )
    saved = checkpoints[-1]
    assert "priors" not in saved
    assert all("screened_families" not in entry for entry in saved["entries"])

    def refuse_score(*args, **kwargs):
        raise AssertionError("cached entries should not be scored again")

    monkeypatch.setattr("smolsmort.forecast.search.score", refuse_score)
    after = search(
        ws, Budget(max_generations=0), resume=saved, priors=dict.fromkeys(ws.families, 0.9)
    )
    assert after["best"] == before["best"]
    for entry in after["leaderboard"]:
        assert entry["screened_families"] == len(
            set(entry["genome"]["families"]) & set(ws.families)
        )


def write_request(folder, mode, version=2):
    request = {
        "spec": {
            "source": "synthetic.csv",
            "mode": mode,
            "task": "regression",
            "columns": [
                {"name": "day", "role": "time" if mode == "series" else "anchor"},
                {"name": "value", "role": "target", "aggregation": "sum"},
            ],
        },
        "prepared": str(folder),
        "summary": {},
        "nthread": 1,
    }
    if version == 2:
        request.update(version=2, families=[{"name": "xgboost"}], time_budget_s=60)
    (folder / "request.json").write_text(json.dumps(request))


@pytest.mark.parametrize("mode", ["series", "row"])
@pytest.mark.parametrize("has_screen", [False, True])
def test_family_worker_reads_parent_screen_and_passes_priors(
    tmp_path, monkeypatch, mode, has_screen
):
    write_request(tmp_path, mode)
    ws = make_workspace(mode=mode)
    ranking = [{"family": name, "score": prior, "p_on": prior} for name, prior in PRIORS.items()]
    if has_screen:
        (tmp_path / "screen.json").write_text(json.dumps(ranking))
    target = tmp_path / "families" / "xgboost"
    target.mkdir(parents=True)
    (target / "screen.json").write_text(json.dumps([{"family": "wrong", "p_on": 1.0}]))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(family_worker, "load_workspace", lambda *args, **kwargs: ws)
    monkeypatch.setattr(get_family("xgboost"), "available", lambda: (True, ""))
    captured = []

    def capture_search(_ws, budget, **kwargs):
        captured.append(kwargs["priors"])
        (tmp_path / "cancel").touch()
        return {
            "leaderboard": [],
            "population": [],
            "best": None,
            "elapsed": 0,
            "stopped": "cancelled",
        }

    monkeypatch.setattr(family_worker, "search", capture_search)
    assert family_worker.run(tmp_path, "xgboost") == 0, (target / "status.json").read_text()
    assert captured == [PRIORS if has_screen and mode == "series" else None]


@pytest.mark.parametrize("mode", ["series", "row"])
@pytest.mark.parametrize("has_screen", [False, True])
def test_v1_worker_passes_saved_or_new_screen_priors(tmp_path, monkeypatch, mode, has_screen):
    write_request(tmp_path, mode, version=1)
    ws = make_workspace(mode=mode)
    ranking = [{"family": name, "score": prior, "p_on": prior} for name, prior in PRIORS.items()]
    if has_screen:
        (tmp_path / "screen.json").write_text(json.dumps(ranking))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(worker, "load_workspace", lambda *args, **kwargs: ws)

    def screen(*args, **kwargs):
        assert not has_screen and mode == "series"
        return ranking

    monkeypatch.setattr(worker, "screen_families", screen)
    captured = []

    def capture_search(_ws, budget, **kwargs):
        captured.append(kwargs["priors"])
        return {
            "leaderboard": [],
            "population": [],
            "best": None,
            "elapsed": 0,
            "stopped": "cancelled",
        }

    monkeypatch.setattr(worker, "search", capture_search)
    assert worker.run(tmp_path) == 0, (tmp_path / "status.json").read_text()
    assert captured == [PRIORS if mode == "series" else None]


def test_top_families_skip_wildcards_raised_to_the_same_prior(tmp_path):
    ranking = [
        {"family": "a", "score": 0.9, "p_on": 0.9, "wildcard": False},
        {"family": "c", "score": 0.6, "p_on": 0.6, "wildcard": False},
        {"family": "w", "score": 0.01, "p_on": 0.5, "wildcard": True},
        {"family": "e", "score": 0.3, "p_on": 0.35, "wildcard": False},
        {"family": "b", "score": 0.0, "p_on": 0.1, "wildcard": False},
    ]
    (tmp_path / "screen.json").write_text(json.dumps(ranking))
    top = read_top_families(tmp_path)
    assert top == ["a", "c", "e"]
    priors = read_priors(tmp_path)
    var = _Variation(make_workspace(list(priors)), np.random.default_rng(17), priors, top)
    assert var.birth_families == top
    seeds = _seed_population(make_workspace(list(priors)), var, 4, None)
    assert set(seeds[0].families) == set(top)
