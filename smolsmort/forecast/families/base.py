"""family contracts and a registry that keeps optional libraries lazy"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["float", "int", "log", "choice"]
    lo: float | int | None = None
    hi: float | int | None = None
    choices: tuple[object, ...] = ()
    default: object = None


class Fitted(Protocol):
    @property
    def eval_history(self) -> dict | None: ...

    def predict(self, x) -> np.ndarray: ...


class Family(Protocol):
    name: str
    label: str
    needs: Literal["lag_features", "raw_series"]
    pip_extra: str

    def available(self) -> tuple[bool, str]: ...

    def space(self) -> list[Param]: ...

    def defaults(self) -> dict: ...

    def objectives(self, task: str) -> tuple[str, ...]: ...

    def cost(self, shape: tuple[int, int], params: dict) -> float: ...

    def fit(
        self,
        x,
        *,
        objective: str,
        y=None,
        feature_types: list[str] | None = None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        **options,
    ) -> Fitted: ...


def probe_import(module: str, pip_extra: str) -> tuple[bool, str]:
    """probe only when asked; shared libraries can fail even when the package is installed"""
    try:
        import_module(module)
    except (ImportError, OSError) as exc:
        return False, f"uv sync --extra {pip_extra} ({exc})"
    return True, ""


_REGISTRY: dict[str, Family] = {}


def register(family: Family) -> None:
    if family.name in _REGISTRY:
        raise ValueError(f"family {family.name!r} is already registered")
    _REGISTRY[family.name] = family


def get_family(name: str) -> Family:
    return _REGISTRY[name]


def available_families() -> list[Family]:
    return [family for family in _REGISTRY.values() if family.available()[0]]
