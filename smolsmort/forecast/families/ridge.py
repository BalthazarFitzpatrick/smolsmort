"""numpy ridge with training-only missing-value and categorical transforms"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.families.base import Param
from smolsmort.forecast.model import ModelError


@dataclass
class RidgeFitted:
    columns: int
    numeric: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    missing: np.ndarray
    categories: tuple[tuple[int, np.ndarray], ...]
    coefficients: np.ndarray
    intercept: float
    eval_history: dict | None = None

    def transform(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.columns:
            raise ModelError("ridge needs a matrix with the training column count")
        if np.isinf(values).any():
            raise ModelError("ridge features must be finite or nan")
        numeric = values[:, self.numeric]
        imputed = np.where(np.isnan(numeric), self.medians, numeric)
        blocks = [(imputed - self.means) / self.scales]
        blocks.append(np.isnan(values[:, self.missing]).astype(float))
        for column, levels in self.categories:
            encoded = values[:, column, None] == levels
            unknown = ~encoded.any(axis=1, keepdims=True)
            blocks.extend((encoded, unknown))
        return np.column_stack(blocks)

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
        numeric = np.array([i for i, kind in enumerate(types) if kind == "q"], dtype=int)
        raw_numeric = values[:, numeric]
        # entirely missing columns use zero; they still carry a missing indicator
        medians = np.array(
            [
                np.median(column[~np.isnan(column)]) if np.isfinite(column).any() else 0.0
                for column in raw_numeric.T
            ]
        )
        imputed = np.where(np.isnan(raw_numeric), medians, raw_numeric)
        means, scales = imputed.mean(axis=0), imputed.std(axis=0)
        scales[scales == 0] = 1.0
        categories = tuple(
            (i, np.unique(values[~np.isnan(values[:, i]), i]))
            for i, kind in enumerate(types)
            if kind == "c"
        )
        fitted = RidgeFitted(
            values.shape[1],
            numeric,
            medians,
            means,
            scales,
            np.flatnonzero(np.isnan(values).any(axis=0)),
            categories,
            np.empty(0),
            0.0,
        )
        design = fitted.transform(values)
        center, target_mean = design.mean(axis=0), float(target.mean())
        centered = design - center
        # centering solves the unpenalised intercept outside the ridge system
        gram = centered.T @ centered
        gram.flat[:: gram.shape[0] + 1] += alpha
        try:
            fitted.coefficients = np.linalg.solve(gram, centered.T @ (target - target_mean))
        except np.linalg.LinAlgError as exc:
            raise ModelError("ridge could not solve the regularised system") from exc
        fitted.intercept = target_mean - float(center @ fitted.coefficients)
        return fitted
