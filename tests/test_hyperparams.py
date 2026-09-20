"""smolsmort/review/hyperparams.py: the train tab's presets and live model-size math."""

from __future__ import annotations

import pytest

from smolsmort.review.hyperparams import (
    HyperparamError,
    box_widths_for,
    heatmap_channels_for,
    param_count,
    preset,
)

torch = pytest.importorskip("torch")


def test_a_known_preset_round_trips():
    balanced = preset("balanced")
    assert balanced.optimizer == "adamw"
    assert balanced.learning_rate == pytest.approx(3e-4)


def test_an_unknown_preset_names_the_known_ones():
    with pytest.raises(HyperparamError, match="balanced"):
        preset("nope")


def test_heatmap_channels_grow_small_to_large():
    small, medium, large = (heatmap_channels_for(s) for s in ("small", "medium", "large"))
    assert small < medium < large


def test_custom_heatmap_channels_are_clamped():
    assert heatmap_channels_for("custom", custom_channels=1000) == 64
    assert heatmap_channels_for("custom", custom_channels=1) == 8


def test_custom_heatmap_channels_without_a_value_refuses():
    with pytest.raises(HyperparamError, match="custom_channels"):
        heatmap_channels_for("custom")


def test_box_widths_grow_small_to_large():
    small, medium, large = (sum(box_widths_for(s)) for s in ("small", "medium", "large"))
    assert small < medium < large


def test_custom_box_scale_is_clamped():
    assert box_widths_for("custom", custom_scale=0.1) == box_widths_for("custom", custom_scale=0.5)
    assert box_widths_for("custom", custom_scale=10) == box_widths_for("custom", custom_scale=2.0)


def test_param_count_grows_small_to_large_for_both_backends():
    for backend in ("heatmap", "box"):
        small = param_count(backend, size="small")
        medium = param_count(backend, size="medium")
        large = param_count(backend, size="large")
        assert small < medium < large, backend


def test_param_count_matches_the_real_model_it_describes():
    from smolsmort.detect.model import build_model, count_parameters

    assert param_count("heatmap", size="medium") == count_parameters(build_model(channels=24))


def test_an_unknown_backend_refuses():
    with pytest.raises(HyperparamError, match="heatmap, box"):
        param_count("nope", size="medium")


def test_listing_presets_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys; from smolsmort.review import hyperparams as h; h.preset('balanced'); print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
