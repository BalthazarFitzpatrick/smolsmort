"""family dispatch preserves the xgboost recipes and fitted predictions"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast.families import available_families, base, get_family, register
from smolsmort.forecast.model import fit_model, predict
from smolsmort.forecast.pipeline import (
    Genome,
    PipelineError,
    _series_fit,
    genome_from_dict,
    make_genome,
)
from smolsmort.forecast.search import (
    DEFAULTS,
    OBJECTIVE_DEFAULTS,
    OBJECTIVE_SPACE,
    SPACE,
    _seed_population,
    _Variation,
)


@pytest.fixture
def other_family(monkeypatch):
    monkeypatch.setattr(base, "_REGISTRY", dict(base._REGISTRY))
    family = SimpleNamespace(
        name="test-family", pip_extra="test-extra", available=lambda: (True, "")
    )
    register(family)
    return family


def test_registry_lists_xgboost_as_available():
    pytest.importorskip("xgboost")
    family = get_family("xgboost")
    assert family in available_families()
    assert family.available() == (True, "")
    assert family.needs == "lag_features"
    assert family.pip_extra == "forecast"


def test_importing_families_does_not_import_xgboost():
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['xgboost'] = None; "
            "from smolsmort.forecast.families import get_family; "
            "family = get_family('xgboost'); "
            "assert not family.available()[0]; "
            "assert 'uv sync --extra forecast' in family.available()[1]",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_xgboost_genome_keeps_the_known_golden_key():
    golden = json.loads((Path(__file__).parent / "golden/forecast_leaderboard.json").read_text())
    old_dict = golden["regression"][0]["genome"]
    old_key = hashlib.sha1(json.dumps(old_dict, sort_keys=True).encode()).hexdigest()[:12]
    genome = genome_from_dict(old_dict)
    assert genome.family == "xgboost"
    assert genome.to_dict() == old_dict
    assert genome.key() == old_key == "a0bafa1a7c3e"


@pytest.mark.parametrize("explicit", [False, True])
def test_xgboost_genome_roundtrip(explicit):
    data = {"families": ["lag+0"], "objective": "squared", "params": {"eta": 0.1}}
    if explicit:
        data["family"] = "xgboost"
    genome = genome_from_dict(data)
    assert genome == genome_from_dict(genome.to_dict())
    assert "family" not in genome.to_dict()


def test_registered_family_roundtrip_and_distinct_key(other_family):
    genome = make_genome(["lag+0"], "squared", {}, family=other_family.name)
    assert genome.to_dict()["family"] == other_family.name
    assert genome_from_dict(genome.to_dict()) == genome
    assert genome_from_dict(genome.to_dict(), family="xgboost").family == "xgboost"
    assert genome.key() != make_genome(["lag+0"], "squared", {}).key()


def test_search_preserves_family_in_variation_and_warm_starts(other_family):
    ws = SimpleNamespace(families=["lag+0"], profile={"genes": {"objectives": ["squared"]}})
    variation = _Variation(ws, np.random.default_rng(0))
    genome = make_genome(ws.families, "squared", DEFAULTS, family=other_family.name)
    assert variation.mutate(genome).family == other_family.name
    assert variation.crossover(genome, genome).family == other_family.name
    assert _seed_population(ws, variation, 1, [genome.to_dict()]) == [genome]


def test_unknown_family_names_the_family_and_install_command():
    with pytest.raises(PipelineError, match="unknown family 'missing'.*uv sync --extra missing"):
        make_genome(["lag+0"], "squared", {}, family="missing")


def test_unavailable_family_names_its_install_extra(other_family):
    other_family.available = lambda: (False, "uv sync --extra test-extra (missing dependency)")
    assert other_family not in available_families()
    genome = Genome(("lag+0",), "squared", (), family=other_family.name)
    assert genome_from_dict(genome.to_dict()) == genome
    with pytest.raises(PipelineError, match="test-family.*uv sync --extra test-extra"):
        _series_fit(None, genome, None, None, 1)


def test_genome_construction_and_search_never_probe_availability(other_family):
    def refuse_probe():
        raise AssertionError("recipe operations must not probe optional libraries")

    other_family.available = refuse_probe
    genome = make_genome(["lag+0"], "squared", DEFAULTS, family=other_family.name)
    assert genome_from_dict(genome.to_dict()) == genome
    ws = SimpleNamespace(families=["lag+0"], profile={"genes": {"objectives": ["squared"]}})
    variation = _Variation(ws, np.random.default_rng(0))
    assert variation.mutate(genome).family == other_family.name
    assert variation.crossover(genome, genome).family == other_family.name
    assert _seed_population(ws, variation, 1, [genome.to_dict()]) == [genome]


def test_missing_xgboost_is_reported_only_at_fit_with_one_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    genome = make_genome(["lag+0"], "squared", {})
    assert genome_from_dict(genome.to_dict()) == genome
    with pytest.raises(PipelineError, match="family 'xgboost' is unavailable") as error:
        _series_fit(None, genome, None, None, 1)
    assert str(error.value).count("uv sync --extra forecast") == 1


def test_registry_refuses_duplicate_names():
    with pytest.raises(ValueError, match="xgboost.*already registered"):
        register(get_family("xgboost"))


def test_xgboost_param_list_matches_search_spaces():
    family = get_family("xgboost")
    specs, defaults = dict(SPACE), dict(DEFAULTS)
    for objective, space in OBJECTIVE_SPACE.items():
        specs.update(space)
        defaults.update(OBJECTIVE_DEFAULTS[objective])
    params = family.space()
    assert [param.name for param in params] == list(specs)
    for param in params:
        spec = specs[param.name]
        if isinstance(spec[0], str):
            assert param.kind == "choice"
            assert param.choices == spec
            assert param.lo is None and param.hi is None
        else:
            lo, hi, kind = spec
            assert (param.lo, param.hi) == (lo, hi)
            assert param.kind == ("float" if kind == "lin" else kind)
            assert param.choices == ()
        assert param.default == defaults[param.name]
    assert family.defaults() == defaults


def test_xgboost_wrapper_predictions_and_history_match_bit_for_bit():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    y = np.exp(x[:, 0]) + 1
    options = {
        "objective": "squared",
        "y": y,
        "params": {"max_depth": 3},
        "max_rounds": 30,
        "patience": 3,
        "seed": 4,
        "nthread": 1,
    }
    direct = fit_model(x, **options)
    wrapped = get_family("xgboost").fit(x, **options)
    np.testing.assert_array_equal(wrapped.predict(x), predict(direct, x))
    assert wrapped.eval_history == direct.eval_history
    assert wrapped.eval_history is not None
    assert wrapped.fitted.rounds == direct.rounds


def test_series_fit_dispatches_sorted_rows_to_the_selected_family(other_family):
    captured = {}
    fitted = SimpleNamespace(eval_history=None, predict=lambda x: x[:, 0])

    def fit(x, **options):
        captured.update(x=x, **options)
        return fitted

    other_family.fit = fit
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    frame = SimpleNamespace(
        features=SimpleNamespace(pick=lambda families: (x, ["lag_1"], ["q"])),
        step=np.array([4, 1, 3, 2]),
        y=np.array([40, 10, 30, 20]),
    )
    genome = make_genome(["lag+0"], "squared", {"eta": 0.1}, family=other_family.name)
    actual, design = _series_fit(None, genome, frame, np.ones(4, dtype=bool), 2)
    assert actual is fitted and design is x
    np.testing.assert_array_equal(captured.pop("x"), x[[1, 3, 2, 0]])
    np.testing.assert_array_equal(captured.pop("y"), [10, 20, 30, 40])
    assert captured == {
        "objective": "squared",
        "feature_types": ["q"],
        "params": {"eta": 0.1},
        "nthread": 2,
    }
