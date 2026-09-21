"""pure request/response shaping for the hyperparams dropdown - no http, no torch import at load.

kept apart from server.py so the clamping and preset-listing logic can be unit tested without ever
starting a socket, the same split snapshot/review/routes.py drew between validation and the handler.
"""

from __future__ import annotations

from dataclasses import asdict

from smolsmort import backends
from smolsmort.optim import OPTIMIZERS
from smolsmort.review.hyperparams import PRESETS, SIZE_NAMES, HyperparamError, param_count

LEARNING_RATE_RANGE = (1e-6, 1e-1)  # a rate outside this either does nothing or diverges at once
MOMENTUM_RANGE = (0.0, 0.999)
WEIGHT_DECAY_RANGE = (0.0, 1e-1)


class RequestError(Exception):
    """a bad request - the server turns this into a 400, never a 500"""


def _clamp(value, low, high):
    return max(low, min(high, value))


def menu_options(backend_name: str) -> dict:
    """what the dropdown needs to draw itself for one backend: presets, optimiser names, and each
    fixed size option's live parameter count - so "large" always shows the true number, not a
    guess written into the page"""
    if backend_name not in backends.names():
        raise RequestError(
            f"no backend called {backend_name!r} - known: {', '.join(backends.names())}"
        )
    if backend_name not in ("heatmap", "box"):
        raise RequestError(f"{backend_name!r} has no hyperparams menu wired up yet")
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_available = False
    else:
        torch_available = True
    sizes = {
        size: (param_count(backend_name, size=size) if torch_available else None)
        for size in ("small", "medium", "large")
    }
    return {
        "backend": backend_name,
        "optimizers": list(OPTIMIZERS),
        "sizes": sizes,
        "size_names": list(SIZE_NAMES),
        "presets": [asdict(p) for p in PRESETS],
    }


def resolve_hyperparams(backend_name: str, payload: dict) -> dict:
    """clamp and complete one submitted config, and report the model size it would build.

    THE SAME CLAMPING RULE AS THE OLD POPUP (snapshot/review/routes.py's train-start): out-of-range
    numbers are corrected rather than refused outright, because a slightly-too-high rate typed by
    hand is a typo, not an attempt to break anything - the one exception is a name that does not
    exist (an optimiser or size nobody defined), which is refused.
    """
    optimizer = payload.get("optimizer", "adamw")
    if optimizer not in OPTIMIZERS:
        raise RequestError(f"no optimizer called {optimizer!r} - known: {', '.join(OPTIMIZERS)}")
    size = payload.get("size", "medium")
    if size not in SIZE_NAMES:
        raise RequestError(f"no size called {size!r} - known: {', '.join(SIZE_NAMES)}")

    learning_rate = _clamp(float(payload.get("learning_rate", 3e-4)), *LEARNING_RATE_RANGE)
    momentum = _clamp(float(payload.get("momentum", 0.9)), *MOMENTUM_RANGE)
    weight_decay = _clamp(float(payload.get("weight_decay", 0.0)), *WEIGHT_DECAY_RANGE)
    seed = int(payload.get("seed") or 0)

    kwargs = {}
    if size == "custom":
        if backend_name == "heatmap":
            kwargs["custom_channels"] = int(payload.get("custom_channels", 24))
        else:
            kwargs["custom_scale"] = float(payload.get("custom_scale", 1.0))
    try:
        params = param_count(backend_name, size=size, **kwargs)
    except HyperparamError as exc:
        raise RequestError(str(exc)) from exc

    return {
        "backend": backend_name,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "seed": seed,
        "size": size,
        "parameters": params,
        **kwargs,
    }
