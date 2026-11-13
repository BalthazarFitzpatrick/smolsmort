"""the effort setting rescales costly ranges, validates in requests and keeps old runs unchanged"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from smolsmort.forecast.effort import COSTLY, EFFORTS, clip_default, scale_range
from smolsmort.forecast.families.base import Param
from smolsmort.forecast.requests import parse_request
from smolsmort.forecast.search import Budget, _FamilyVariation, _Variation


def test_normal_is_unchanged_and_other_names_are_untouched():
    assert scale_range("xgboost", "max_depth", 3, 10, "int", "normal") == (3, 10)
    for effort in EFFORTS:
        assert scale_range("xgboost", "eta", 0.03, 0.3, "log", effort) == (0.03, 0.3)
        assert scale_range("lightgbm", "boosting", None, None, "choice", effort) == (None, None)


def test_low_and_high_for_int_ranges():
    assert scale_range("xgboost", "max_depth", 3, 10, "int", "low") == (3, 6)
    assert scale_range("xgboost", "max_depth", 3, 10, "int", "high") == (3, 12)
    assert scale_range("lightgbm", "num_leaves", 8, 256, "int", "low") == (8, 107)
    assert scale_range("lightgbm", "num_leaves", 8, 256, "int", "high") == (8, 384)
    assert scale_range("lightgbm", "num_leaves", 8, 400, "int", "high") == (8, 512)


def test_low_never_drops_below_lo_plus_one_and_high_never_shrinks():
    assert scale_range("xgboost", "max_depth", 3, 4, "int", "low") == (3, 4)
    assert scale_range("catboost", "depth", 4, 20, "int", "high") == (4, 20)


def test_log_kind_scales_in_log_space():
    lo, hi = scale_range("extratrees", "n_estimators", 100, 1000, "log", "low")
    assert lo == 100 and hi == pytest.approx(100 * 10**0.4)
    assert scale_range("extratrees", "n_estimators", 100, 600, "log", "high") == (100, 900)
    assert scale_range("extratrees", "n_estimators", 100, 900, "log", "high") == (100, 1000)


def test_ceilings_cover_the_named_params():
    assert COSTLY["lstm"] == {"hidden": 256, "layers": 3, "lookback": 156}
    assert COSTLY["mlp"] == {"n_layers": 6, "width": 1024}


def test_default_is_clipped_into_the_range():
    assert clip_default(6, 3, 5, "int") == 5
    assert clip_default(2, 3, 5, "int") == 3
    assert clip_default(4.0, 3, 5, "int") == 4
    assert clip_default("x", 3, 5, "int") == "x"


@pytest.mark.parametrize("effort", EFFORTS)
def test_request_accepts_known_efforts(effort):
    parsed = parse_request(make_request(effort=effort))
    assert parsed.get("effort") == (None if effort == "normal" else effort)


@pytest.mark.parametrize("value", ["", "max", "High", None, 1, True])
def test_request_rejects_unknown_efforts(value):
    with pytest.raises(ValueError, match="effort must be low, normal or high"):
        parse_request(make_request(effort=value))


def make_request(**changes):
    return {"version": 2, "time_budget_s": 1200, "families": [{"name": "snaive"}], **changes}


def digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def test_old_request_digest_is_stable_with_absent_or_normal_effort():
    old = parse_request(make_request(max_concurrent_families=4))
    assert "effort" not in old
    explicit = parse_request(make_request(max_concurrent_families=4, effort="normal"))
    assert digest(explicit) == digest(old)
    assert digest(parse_request(make_request(max_concurrent_families=4, effort="low"))) != digest(
        old
    )


class FakeFamily:
    name = "lightgbm"
    needs = "lag_features"

    def space(self):
        return [
            Param("num_leaves", "int", 8, 256, default=31),
            Param("max_depth", "int", 3, 12, default=10),
            Param("learning_rate", "log", 0.02, 0.3, default=0.1),
            Param("boosting", "choice", choices=("gbdt", "dart"), default="gbdt"),
        ]

    def objectives(self, task):
        return ("squared",)


def make_ws():
    return SimpleNamespace(
        mode=None,
        families=["lag_1"],
        spec=SimpleNamespace(task="regression"),
        profile={"genes": {"objectives": ["squared"]}},
    )


def test_budget_effort_defaults_to_normal():
    assert Budget().effort == "normal"


def test_old_budget_gives_identical_ranges_and_draws():
    rng = lambda: np.random.default_rng(3)  # noqa: E731 - tiny local factory
    plain = _FamilyVariation(make_ws(), rng(), FakeFamily(), None)
    normal = _FamilyVariation(make_ws(), rng(), FakeFamily(), None, effort="normal")
    assert plain.space == normal.space and plain.defaults == normal.defaults
    assert plain.space["num_leaves"] == (8, 256, "int")
    legacy = _Variation(make_ws(), rng())
    assert legacy.space["max_depth"] == (3, 10, "int")
    assert legacy.defaults["max_depth"] == 6


def test_low_legacy_xgboost_space_and_default():
    var = _Variation(make_ws(), np.random.default_rng(0), effort="low")
    assert var.space["max_depth"] == (3, 6, "int")
    assert var.defaults["max_depth"] == 6
    high = _Variation(make_ws(), np.random.default_rng(0), effort="high")
    assert high.space["max_depth"] == (3, 12, "int")
    assert high.space["eta"] == (0.03, 0.3, "log")


@pytest.mark.parametrize("effort", EFFORTS)
def test_family_draws_and_nudges_stay_inside_scaled_ranges(effort):
    var = _FamilyVariation(make_ws(), np.random.default_rng(1), FakeFamily(), None, effort=effort)
    lo, hi, _kind = var.space["num_leaves"]
    dlo, dhi, _kind = var.space["max_depth"]
    assert dlo <= var.defaults["max_depth"] <= dhi
    for _ in range(200):
        genome = var.random()
        mutated = var.mutate(genome)
        for g in (genome, mutated):
            params = dict(g.params)
            assert lo <= params["num_leaves"] <= hi
            assert dlo <= params["max_depth"] <= dhi
    if effort == "low":
        assert var.space["max_depth"] == (3, 7, "int")
        assert var.defaults["max_depth"] == 7
