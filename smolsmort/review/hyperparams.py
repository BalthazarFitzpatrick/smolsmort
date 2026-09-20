"""hyperparameter presets and live model-size math for the train tab's dropdown.

WHY A SEPARATE MODULE: `smolsmort.optim` is the mechanics (name + numbers -> a torch optimiser);
this is the judgement layer on top - what to suggest, and how big a model each size option builds -
so the two can change independently. Nothing here imports torch at module load, same reason
`smolsmort.backends` stays lazy: naming a preset must never cost a torch import.
"""

from __future__ import annotations

from dataclasses import dataclass

# heatmap channels / box widths base-scale, by preset name. "custom" has no fixed value - the
# caller supplies its own channels/scale, clamped by CHANNEL_RANGE / WIDTH_SCALE_RANGE below.
HEATMAP_CHANNELS = {"small": 16, "medium": 24, "large": 32}
BOX_WIDTHS = {
    "small": (12, 24, 48, 72, 96),
    "medium": (16, 32, 64, 96, 128),
    "large": (24, 48, 96, 144, 192),
}
SIZE_NAMES = ("small", "medium", "large", "custom")
CHANNEL_RANGE = (8, 64)  # clamp for a custom heatmap channel count
WIDTH_SCALE_RANGE = (0.5, 2.0)  # clamp for a custom box width multiplier, applied to "medium"


class HyperparamError(Exception):
    pass


@dataclass(frozen=True)
class Preset:
    name: str
    why: str
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    size: str = "medium"


# COMMON CNN STARTING POINTS, not measured on this project's own data - label them as such in the
# UI. AdamW's momentum is its beta1; SGD pairs a higher lr with nesterov momentum, the classic combo.
PRESETS = (
    Preset("balanced", "a sensible default for most runs", "adamw", 3e-4, 0.9, 1e-4, "medium"),
    Preset("fast start", "few frames, wants to move quickly", "adamw", 1e-3, 0.9, 1e-4, "small"),
    Preset("careful", "fine-tuning, avoids overshooting", "adamw", 1e-4, 0.9, 1e-3, "medium"),
    Preset("sgd classic", "textbook sgd with nesterov momentum", "sgd", 1e-2, 0.9, 5e-4, "medium"),
)


def preset(name: str) -> Preset:
    for candidate in PRESETS:
        if candidate.name == name:
            return candidate
    known = ", ".join(p.name for p in PRESETS)
    raise HyperparamError(f"no preset called {name!r} - known: {known}")


def heatmap_channels_for(size: str, *, custom_channels: int | None = None) -> int:
    """the channel count a size option builds, for the fixed-size (heatmap) backend"""
    if size == "custom":
        if custom_channels is None:
            raise HyperparamError("custom size needs custom_channels")
        low, high = CHANNEL_RANGE
        return max(low, min(high, int(custom_channels)))
    if size not in HEATMAP_CHANNELS:
        raise HyperparamError(f"no size called {size!r} - known: {', '.join(SIZE_NAMES)}")
    return HEATMAP_CHANNELS[size]


def box_widths_for(
    size: str, *, custom_scale: float | None = None
) -> tuple[int, int, int, int, int]:
    """the widths tuple a size option builds, for the size-aware (box) backend"""
    if size == "custom":
        if custom_scale is None:
            raise HyperparamError("custom size needs custom_scale")
        low, high = WIDTH_SCALE_RANGE
        scale = max(low, min(high, float(custom_scale)))
        return tuple(max(4, round(w * scale)) for w in BOX_WIDTHS["medium"])
    if size not in BOX_WIDTHS:
        raise HyperparamError(f"no size called {size!r} - known: {', '.join(SIZE_NAMES)}")
    return BOX_WIDTHS[size]


def param_count(
    backend_name: str,
    *,
    size: str,
    custom_channels: int | None = None,
    custom_scale: float | None = None,
) -> int:
    """the actual parameter count `build_model` would produce for this backend and size option -
    computed from the real model, never a hardcoded figure, so it tracks the code that builds it"""
    from smolsmort.detect.model import count_parameters

    if backend_name == "heatmap":
        from smolsmort.detect.model import build_model

        channels = heatmap_channels_for(size, custom_channels=custom_channels)
        return count_parameters(build_model(channels=channels))
    if backend_name == "box":
        from smolsmort.boxes.model import build_model

        widths = box_widths_for(size, custom_scale=custom_scale)
        return count_parameters(build_model(widths=widths))
    raise HyperparamError(f"no backend called {backend_name!r} known to hyperparams - heatmap, box")
