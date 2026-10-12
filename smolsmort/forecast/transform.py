"""reversible target scales; origin values always stay on the raw scale"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TransformError(ValueError):
    pass


@dataclass(frozen=True)
class TargetTransform:
    name: str

    def validate_target(self, y):
        values = np.asarray(y, dtype=float)
        if self.name != "none" and np.any(values < 0):
            raise TransformError(f"{self.name} transform refuses negative targets")
        return values

    def _log_base(self, base):
        if base is None:
            raise TransformError(f"{self.name} transform requires origin targets")
        values = self.validate_target(base)
        if not np.isfinite(values).all():
            raise TransformError(f"{self.name} transform requires finite origin targets")
        return np.log1p(values)

    def fwd(self, y, base=None):
        if self.name == "none":
            return y
        values = np.log1p(self.validate_target(y))
        return values - self._log_base(base) if self.name == "logdiff" else values

    def inv(self, z, base=None):
        if self.name == "none":
            return z
        values = np.asarray(z, dtype=float)
        if self.name == "logdiff":
            values = values + self._log_base(base)
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = np.expm1(values)
        if not np.isfinite(prediction).all():
            raise TransformError(f"{self.name} transform returned nonfinite raw predictions")
        return prediction


TRANSFORMS = {name: TargetTransform(name) for name in ("none", "log1p", "logdiff")}


def get_transform(name: str = "none") -> TargetTransform:
    try:
        return TRANSFORMS[name]
    except (KeyError, TypeError) as exc:
        raise TransformError("transform must be none, log1p or logdiff") from exc
