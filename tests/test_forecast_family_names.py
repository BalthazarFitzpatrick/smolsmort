"""source-aware family names preserve saved selections, matrices and seeded searches"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("duckdb")

from smolsmort.forecast.candidate_worker import estimate_cost
from smolsmort.forecast.features import family_aliases, family_name, row_features, split_family_name
from smolsmort.forecast.pipeline import finish, genome_from_dict, load_workspace, make_genome, score
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.screen import read_priors, read_top_families, screen_families
from smolsmort.forecast.search import Budget, _map_saved_names, search
from smolsmort.forecast.spec import ANCHOR, Column, PrepSpec

GOLDEN = Path(__file__).parent / "golden"


def build_weekly_workspace(folder):
    path = folder / "weekly.csv"
    rng = np.random.default_rng(41)
    signal = 40 + 12 * np.sin(2 * np.pi * np.arange(166) / 52) + rng.normal(0, 2, 166)
    with path.open("w", newline="") as source:
        writer = csv.writer(source)
        writer.writerow(
            ["need_by_date", "product_group", "unit", "lead_days", "components", "site", "project"]
        )
        for group in ("a", "b"):
            for week in range(156):
                for site, share in (("north", 0.6), ("south", 0.4)):
                    writer.writerow(
                        [
                            str(np.datetime64("2020-01-06") + np.timedelta64(week * 7, "D")),
                            group,
                            signal[week] * share,
                            3 + (week % 5),
                            signal[week + 6] * share,
                            site,
                            "first" if week % 3 else "second",
                        ]
                    )
    spec = PrepSpec(
        str(path),
        "series",
        "regression",
        (
            Column("need_by_date", "time"),
            Column("product_group", "dimension"),
            Column("unit", "target", "sum"),
            Column("lead_days", "measure", "mean"),
            Column("components", "measure", "sum"),
            Column("site", "input"),
            Column("project", "input"),
        ),
        step="week",
        horizon=8,
    )
    prepared = prepare(spec, folder / "cache")
    return load_workspace(spec, prepared.folder, prepared.summary)


@pytest.fixture(scope="module")
def weekly_workspace(tmp_path_factory):
    return build_weekly_workspace(tmp_path_factory.mktemp("family-names"))


@pytest.fixture(scope="module")
def legacy():
    # captured from 3af2411 before the rename, using build_weekly_workspace and seed 7
    return json.loads((GOLDEN / "family_names_legacy.json").read_text())


def mapped_genome(ws, data):
    aliases = family_aliases(ws.spec, data["families"])
    return {**data, "families": [aliases.get(name, name) for name in data["families"]]}


def assert_same_search(ws, actual, expected):
    assert actual["best"] == mapped_genome(ws, expected["best"])
    assert actual["population"] == [mapped_genome(ws, data) for data in expected["population"]]
    assert actual["generations"] == expected["generations"]
    assert actual["stopped"] == expected["stopped"]
    assert len(actual["leaderboard"]) == len(expected["leaderboard"])
    for found, saved in zip(actual["leaderboard"], expected["leaderboard"], strict=True):
        assert found["genome"] == mapped_genome(ws, saved["genome"])
        assert found["fitness"] == pytest.approx(saved["fitness"], rel=1e-12)
        assert found["generation"] == saved["generation"]


def test_every_weekly_family_names_its_spec_sources(weekly_workspace):
    ws = weekly_workspace
    inventory = json.loads((GOLDEN / "family_names_weekly.json").read_text())
    assert ws.families == inventory["families"]
    aliases = family_aliases(ws.spec)
    assert inventory["legacy_roll_aliases"] == [
        aliases[f"roll:{window}"] for window in (4, 8, 13, 26, 52)
    ]
    target_ops = {"lag", "season", "roll", "diff", "slope", "momentum", "cycle_ratio", "ewm"}
    for name in ws.families:
        column, clock, op = split_family_name(name)
        kind = op.partition(":")[0].partition("+")[0]
        if kind == "cal":
            assert (column, clock) == (None, "need_by_date")
        elif kind == "dim":
            assert (column, clock) == ("product_group", None)
        elif kind in target_ops:
            assert (column, clock) == ("unit", "need_by_date")
        elif kind in {"measure", "lead"}:
            assert column in ws.spec.named("measure") and clock == "need_by_date"
        else:
            assert kind in {"mix", "spread"}
            assert column in ws.spec.named("input") and clock == "need_by_date"
        assert family_name(op, column, clock) == name
    assert {"components@need_by_date:lead", "unit@need_by_date:season"} <= set(ws.families)


@pytest.mark.parametrize("column,clock", [("x", "date"), ("site:level@x%", "date:@%"), ("ä", "日")])
def test_name_delimiters_round_trip(column, clock):
    assert split_family_name(family_name("roll:13:q10", column, clock)) == (
        column,
        clock,
        "roll:13:q10",
    )
    assert split_family_name(family_name("dim", column)) == (column, None, "dim")
    assert split_family_name(family_name("cal:week", clock=clock)) == (None, clock, "cal:week")


def test_matrices_equal_the_pre_rename_baseline_in_order(weekly_workspace):
    ws = weekly_workspace
    with np.load(GOLDEN / "family_names_legacy.npz") as saved:
        for bucket, frame in ws.frames.items():
            prefix = str(bucket[1])
            aliases = family_aliases(ws.spec, saved[prefix + "_families"])
            assert frame.features.names == saved[prefix + "_names"].tolist()
            assert frame.features.types == saved[prefix + "_types"].tolist()
            assert frame.features.families == [
                aliases[name] for name in saved[prefix + "_families"]
            ]
            np.testing.assert_array_equal(frame.features.x, saved[prefix + "_x"])


def test_fixed_seed_search_matches_pre_rename_genomes_and_fitness(weekly_workspace, legacy):
    actual = search(
        weekly_workspace, Budget(population=6, max_generations=3, plateau=5, seed=7), family="ridge"
    )
    assert_same_search(weekly_workspace, actual, legacy["result"])


def test_old_recipe_refits_the_same_columns_and_forecast(weekly_workspace, tmp_path):
    ws = weekly_workspace
    old = make_genome(
        [
            "lag+0",
            "season",
            "roll:13",
            "measure:lead_days",
            "lead:components",
            "mix:site",
            "dim:product_group",
        ],
        "squared",
        {"alpha": 1.0},
        "ridge",
    ).to_dict()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"genome": old}))
    saved = json.loads(path.read_text())["genome"]
    renamed = genome_from_dict(saved, spec=ws.spec)
    assert renamed.to_dict() == mapped_genome(ws, saved)
    for frame in ws.frames.values():
        before = frame.features.pick(saved["families"])
        after = frame.features.pick(renamed.families)
        assert before[1:] == after[1:]
        np.testing.assert_array_equal(before[0], after[0])
    old_fit = finish(ws, genome_from_dict(saved))
    new_fit = finish(ws, renamed)
    assert old_fit["model_error"] == new_fit["model_error"]
    for name, values in old_fit["forecast"].items():
        np.testing.assert_array_equal(values, new_fit["forecast"][name])
    assert estimate_cost(ws, genome_from_dict(saved)) == estimate_cost(ws, renamed)


def test_old_checkpoint_resumes_without_rescoring_cached_genomes(weekly_workspace, legacy):
    ws = weekly_workspace
    saved = deepcopy(legacy["checkpoint"])
    cached = {genome_from_dict(entry["genome"], spec=ws.spec).key() for entry in saved["entries"]}

    def check_score(genome, **kwargs):
        assert genome.key() not in cached
        return score(ws, genome, **kwargs)

    budget = Budget(population=6, max_generations=4, plateau=5, seed=7)
    actual = search(ws, budget, family="ridge", resume=saved, evaluate_score=check_score)
    expected = search(ws, budget, family="ridge")
    assert_same_search(ws, actual, expected)
    assert saved == legacy["checkpoint"]


def test_exhausted_old_checkpoint_retains_the_cached_winner(weekly_workspace, legacy):
    ws = weekly_workspace
    actual = search(ws, Budget(max_generations=0), family="ridge", resume=legacy["checkpoint"])
    assert actual["best"] == mapped_genome(ws, legacy["result"]["best"])


def test_checkpoint_migrates_cost_hint_keys_and_keeps_cached_measurements(weekly_workspace, legacy):
    ws = weekly_workspace
    saved = deepcopy(legacy["checkpoint"])
    saved["candidate_times"] = deepcopy(saved["entries"][:1])
    saved["cost_hints"] = {
        genome_from_dict(data).key(): index + 0.5 for index, data in enumerate(saved["population"])
    }
    priors = {entry["family"]: entry["p_on"] for entry in legacy["screen"]}
    _, migrated, mapped_priors, mapped_top = _map_saved_names(
        ws, None, saved, priors, ["lag+0", "season"]
    )
    assert migrated["cost_hints"] == {
        genome_from_dict(data, spec=ws.spec).key(): index + 0.5
        for index, data in enumerate(saved["population"])
    }
    assert migrated["best_key"] == genome_from_dict(legacy["result"]["best"], spec=ws.spec).key()
    for field in ("entries", "candidate_times"):
        for old, new in zip(saved[field], migrated[field], strict=True):
            assert new["key"] == genome_from_dict(old["genome"], spec=ws.spec).key()
            assert {key: value for key, value in old.items() if key not in {"key", "genome"}} == {
                key: value for key, value in new.items() if key not in {"key", "genome"}
            }
    aliases = family_aliases(ws.spec)
    assert mapped_priors == {aliases[name]: value for name, value in priors.items()}
    assert mapped_top == [aliases["lag+0"], aliases["season"]]


def test_old_warm_start_selects_the_same_seed(weekly_workspace, legacy):
    ws = weekly_workspace
    old = legacy["result"]["population"]
    budget = Budget(population=6, max_generations=1, seed=7)
    actual = search(ws, budget, family="ridge", warm=old)
    expected = search(ws, budget, family="ridge", warm=[mapped_genome(ws, data) for data in old])
    assert_same_search(ws, actual, expected)


def test_old_screen_and_new_screen_have_identical_priors_and_top_set(
    weekly_workspace, legacy, tmp_path
):
    ws = weekly_workspace
    aliases = family_aliases(ws.spec)
    ranking = legacy["screen"]
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(ranking))
    old_priors = read_priors(tmp_path, ws.spec)
    old_top = read_top_families(tmp_path, ws.spec)
    assert old_priors == {aliases[entry["family"]]: entry["p_on"] for entry in ranking}
    assert old_top == [aliases[name] for name in read_top_families(tmp_path)]
    actual = screen_families(ws, seed=7)
    assert [entry["family"] for entry in actual] == [aliases[entry["family"]] for entry in ranking]
    np.testing.assert_allclose(
        [[entry[key] for key in ("score", "rank_corr", "gain", "p_on")] for entry in actual],
        [[entry[key] for key in ("score", "rank_corr", "gain", "p_on")] for entry in ranking],
        rtol=1e-12,
    )
    path.write_text(json.dumps(actual))
    assert read_priors(tmp_path, ws.spec) == old_priors
    assert read_top_families(tmp_path, ws.spec) == old_top
    budget = Budget(population=4, max_generations=1, seed=7)
    old = search(
        ws,
        budget,
        family="ridge",
        priors={entry["family"]: entry["p_on"] for entry in ranking},
        top_families=["lag+0", "roll:13:mean", "measure:components"],
    )
    new = search(
        ws,
        budget,
        family="ridge",
        priors=old_priors,
        top_families=[aliases[name] for name in ["lag+0", "roll:13:mean", "measure:components"]],
    )
    assert_same_search(ws, old, new)


def test_aliases_follow_renamed_spec_columns_and_detected_periods(weekly_workspace):
    spec = weekly_workspace.spec
    renamed = replace(
        spec,
        columns=tuple(replace(column, name=f"source:{column.name}") for column in spec.columns),
    )
    aliases = family_aliases(renamed, ["lag:367", "roll:367", "roll:367:q10"])
    assert aliases["lag:367"] == "source%3Aunit@source%3Aneed_by_date:lag:367"
    assert aliases["roll:367"] == "source%3Aunit@source%3Aneed_by_date:roll:367"
    assert aliases["cal:week"] == "source%3Aneed_by_date:cal:week"


def test_reserved_source_names_cannot_alias_another_columns_family():
    spec = PrepSpec(
        "rows.csv",
        "row",
        "regression",
        (
            Column("date", "anchor"),
            Column("target", "target"),
            Column("dim", "measure"),
            Column("measure", "dimension"),
        ),
    )
    table = {
        ANCHOR: np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]"),
        "dim": np.arange(4.0),
        "measure": np.array(["a", "a", "b", "b"]),
    }
    features = row_features(spec, table, np.ones(4, dtype=bool))
    assert features.pick(["measure:dim"])[1] == ["dim"]
    assert features.pick(["dim:measure"])[1] == ["measure"]
    assert features.pick(["%64im:measure"])[1] == ["dim"]
    assert features.pick(["%6Deasure:dim"])[1] == ["measure"]
    assert split_family_name("dim:measure") == (None, None, "dim:measure")
    assert split_family_name("measure:dim") == (None, None, "measure:dim")
    for name in features.families:
        parsed = split_family_name(name)
        assert parsed[0] in {"dim", "measure", None}
        assert parsed[1] in {"date", None}
        assert genome_from_dict(
            make_genome([name], "squared", {}).to_dict(), spec=spec
        ).families == (name,)


def test_row_families_use_the_anchor_only_for_calendar():
    spec = PrepSpec(
        "rows.csv",
        "row",
        "regression",
        (
            Column("created", "anchor"),
            Column("target", "target"),
            Column("lead_days", "measure"),
            Column("site", "dimension"),
        ),
    )
    table = {
        ANCHOR: np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]"),
        "lead_days": np.arange(4.0),
        "site": np.array(["a", "a", "b", "b"]),
    }
    features = row_features(spec, table, np.ones(4, dtype=bool))
    assert set(features.families) == {
        "lead_days:measure",
        "lead_days:log",
        "site:dim",
        "site:freq",
        *(f"created:cal:{part}" for part in ("week", "month", "quarter", "weekday", "trend")),
    }
    for name in features.families:
        column, clock, op = split_family_name(name)
        if op.partition(":")[0] == "cal":
            assert column is None and clock == "created"
        else:
            assert column in {"lead_days", "site"} and clock is None
