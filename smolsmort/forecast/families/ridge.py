"""numpy ridge with training-only missing-value and categorical transforms"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from smolsmort.forecast.families.base import Param
from smolsmort.forecast.families.preprocess import Preprocessor
from smolsmort.forecast.model import ModelError


@dataclass
class RidgeFitted:
    prep: Preprocessor
    coefficients: np.ndarray
    intercept: float
    eval_history: dict | None = None

    def __getattr__(self, name: str):
        # preprocessing state (means, scales, medians, ...) reads through to the shared prep
        if name == "prep":
            raise AttributeError(name)
        return getattr(self.prep, name)

    def transform(self, x) -> np.ndarray:
        return self.prep.transform(x)

    def predict(self, x) -> np.ndarray:
        predictions = self.transform(x) @ self.coefficients + self.intercept
        if not np.isfinite(predictions).all():
            raise ModelError("ridge returned non-finite predictions")
        return predictions


class RidgeFamily:
    name = "ridge"
    label = "ridge"
    needs = "lag_features"
    pip_extra = ""
    objective_aliases = {"squared": "squared"}

    def available(self) -> tuple[bool, str]:
        return True, ""

    def space(self) -> list[Param]:
        return [Param("alpha", "log", 1e-4, 1e4, default=1.0)]

    def defaults(self) -> dict:
        return {param.name: param.default for param in self.space()}

    def objectives(self, task: str) -> tuple[str, ...]:
        return ("squared",) if task == "regression" else ()

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        rows, columns = shape
        return float(rows * columns**2 + columns**3)

    def fit(
        self,
        x,
        *,
        objective: str = "squared",
        y=None,
        feature_types: list[str] | None = None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        **options,
    ) -> RidgeFitted:
        if objective != "squared":
            raise ModelError(f"no ridge objective called {objective!r}")
        alpha = float((params or {}).get("alpha", 1.0))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ModelError("ridge alpha must be positive and finite")
        values = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if values.ndim != 2 or target.shape != (len(values),):
            raise ModelError("ridge needs a feature matrix and one target per row")
        types = ["q"] * values.shape[1] if feature_types is None else feature_types
        if len(types) != values.shape[1] or any(kind not in {"c", "q"} for kind in types):
            raise ModelError("ridge needs one c or q type marker per feature")
        usable = np.isfinite(target)
        values, target = values[usable], target[usable]
        if not len(target):
            raise ModelError("ridge needs at least one usable target")
        if np.isinf(values).any():
            raise ModelError("ridge features must be finite or nan")
        prep = Preprocessor.fit(values, types, scale=True, label="ridge")
        fitted = RidgeFitted(prep, np.empty(0), 0.0)
        design = prep.transform(values)
        center, target_mean = design.mean(axis=0), float(target.mean())
        centered = design - center
        # centering solves the unpenalised intercept outside the ridge system
        with threadpool_limits(nthread):
            gram = centered.T @ centered
            gram.flat[:: gram.shape[0] + 1] += alpha
            try:
                fitted.coefficients = np.linalg.solve(gram, centered.T @ (target - target_mean))
            except np.linalg.LinAlgError as exc:
                raise ModelError("ridge could not solve the regularised system") from exc
            fitted.intercept = target_mean - float(center @ fitted.coefficients)
        residual = fitted.predict(values) - target
        loss = float(residual @ residual + alpha * (fitted.coefficients @ fitted.coefficients))
        fitted.eval_history = {"metric": "penalized_squared_error", "training": [loss]}
        return fitted
