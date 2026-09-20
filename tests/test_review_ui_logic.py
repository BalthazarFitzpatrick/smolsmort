"""smolsmort/review_ui/logic.py: clamping and menu-option shaping, no http and no torch required
for the parts that don't need a real model."""

from __future__ import annotations

import pytest

from smolsmort.review_ui.logic import RequestError, menu_options, resolve_hyperparams

torch = pytest.importorskip("torch")


def test_menu_options_lists_presets_optimizers_and_growing_sizes():
    options = menu_options("heatmap")
    assert options["optimizers"] == ["adamw", "sgd"]
    assert [p["name"] for p in options["presets"]]
    assert options["sizes"]["small"] < options["sizes"]["medium"] < options["sizes"]["large"]


def test_menu_options_refuses_a_backend_with_no_menu():
    with pytest.raises(RequestError, match="xgboost"):
        menu_options("xgboost")


def test_menu_options_refuses_an_unknown_backend():
    with pytest.raises(RequestError, match="known"):
        menu_options("nope")


def test_resolve_clamps_an_out_of_range_learning_rate():
    resolved = resolve_hyperparams("heatmap", {"learning_rate": 10.0})
    assert resolved["learning_rate"] == pytest.approx(1e-1)


def test_resolve_clamps_momentum_and_weight_decay():
    resolved = resolve_hyperparams("heatmap", {"momentum": 5.0, "weight_decay": -1.0})
    assert resolved["momentum"] == pytest.approx(0.999)
    assert resolved["weight_decay"] == pytest.approx(0.0)


def test_resolve_refuses_an_unknown_optimizer():
    with pytest.raises(RequestError, match="adamw, sgd"):
        resolve_hyperparams("heatmap", {"optimizer": "nope"})


def test_resolve_refuses_an_unknown_size():
    with pytest.raises(RequestError, match="small, medium, large, custom"):
        resolve_hyperparams("heatmap", {"size": "huge"})


def test_resolve_reports_the_real_parameter_count_for_a_fixed_size():
    resolved = resolve_hyperparams("heatmap", {"size": "small"})
    from smolsmort.review.hyperparams import param_count

    assert resolved["parameters"] == param_count("heatmap", size="small")


def test_resolve_honours_a_custom_channel_count():
    resolved = resolve_hyperparams("heatmap", {"size": "custom", "custom_channels": 40})
    assert resolved["custom_channels"] == 40
    from smolsmort.review.hyperparams import param_count

    assert resolved["parameters"] == param_count("heatmap", size="custom", custom_channels=40)


def test_resolve_defaults_match_the_old_popups_own_defaults():
    """the popup this replaces (snapshot's #train-config-popup) defaulted a blank rate/seed to
    3e-4 and 0 - this menu never leaves them blank, but its own defaults must agree"""
    resolved = resolve_hyperparams("heatmap", {})
    assert resolved["learning_rate"] == pytest.approx(3e-4)
    assert resolved["seed"] == 0
