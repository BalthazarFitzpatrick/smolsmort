"""the one optimiser switch both trainers (detect, boxes) build from.

WHY SHARED: each trainer used to hardcode its own optimiser (Adam for detect, AdamW for boxes) with
no momentum or decay exposed anywhere - a run could not be pointed at a different point on the
lr/momentum/decay surface without editing code. This is the single place that turns a name plus
three numbers into a torch optimiser, so a menu offering "adamw" or "sgd" means the same thing to
either backend.
"""

from __future__ import annotations

OPTIMIZERS = ("adamw", "sgd")


class OptimizerError(Exception):
    pass


def build_optimizer(
    parameters, *, optimizer: str, learning_rate: float, momentum: float, weight_decay: float
):
    """adamw: momentum is beta1, the rest of adam's defaults (beta2, eps) are torch's own.
    sgd: momentum is classic sgd momentum, with nesterov on - the standard pairing for a small cnn.
    """
    torch = __import__("torch")
    if optimizer == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, betas=(momentum, 0.999), weight_decay=weight_decay
        )
    if optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=momentum > 0,
        )
    raise OptimizerError(f"no optimizer called {optimizer!r} - known: {', '.join(OPTIMIZERS)}")
