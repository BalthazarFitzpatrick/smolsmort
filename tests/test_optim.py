"""smolsmort/optim.py: the one optimiser switch both trainers build from."""

from __future__ import annotations

import pytest

from smolsmort.optim import OptimizerError, build_optimizer

torch = pytest.importorskip("torch")


def _params():
    return [torch.nn.Parameter(torch.zeros(2))]


def test_adamw_carries_momentum_as_beta1_and_the_given_decay():
    optimiser = build_optimizer(
        _params(), optimizer="adamw", learning_rate=1e-3, momentum=0.8, weight_decay=1e-4
    )
    assert isinstance(optimiser, torch.optim.AdamW)
    group = optimiser.param_groups[0]
    assert group["lr"] == 1e-3
    assert group["betas"][0] == pytest.approx(0.8)
    assert group["weight_decay"] == 1e-4


def test_sgd_carries_momentum_and_turns_on_nesterov():
    optimiser = build_optimizer(
        _params(), optimizer="sgd", learning_rate=1e-2, momentum=0.9, weight_decay=5e-4
    )
    assert isinstance(optimiser, torch.optim.SGD)
    group = optimiser.param_groups[0]
    assert group["momentum"] == 0.9
    assert group["nesterov"] is True


def test_sgd_with_zero_momentum_does_not_ask_for_nesterov():
    """nesterov needs momentum > 0 - torch itself refuses nesterov=True with momentum=0"""
    optimiser = build_optimizer(
        _params(), optimizer="sgd", learning_rate=1e-2, momentum=0.0, weight_decay=0.0
    )
    assert optimiser.param_groups[0]["nesterov"] is False


def test_an_unknown_optimizer_names_the_known_ones():
    with pytest.raises(OptimizerError, match="adamw, sgd"):
        build_optimizer(
            _params(), optimizer="nope", learning_rate=1e-3, momentum=0.9, weight_decay=0.0
        )
