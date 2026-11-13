"""scikit-learn extra trees with training-only imputation and one-hot categories"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.families.base import Param, probe_import
from smolsmort.forecast.families.preprocess import Preprocessor
from smolsmort.forecast.model import ModelError

CRITERIA = {"squared_error": "squared_error", "absolute_error": "absolute_error"}


@dataclass
class ExtratreesFitted:
    preprocessor: Preprocessor
    model: object
    eval_history: dict | None = None

    def predict(self, x) -> np.ndarray:
        predictions = np.asarray(self.model.predict(self.preprocessor.transform(x)), dtype=float)
        if not np.isfinite(predictions).all():
            raise ModelError("extratrees returned non-finite predictions")
        return predictions


class ExtratreesFamily:
    name = "extratrees"
    label = "extra trees"
    needs = "lag_features"
    pip_extra = "trees"
    objective_aliases = {"squared_error": "squared", "absolute_error": "absolute"}

    def available(self) -> tuple[bool, str]:
        return probe_import("sklearn", self.pip_extra)

    def space(self) -> list[Param]:
        return [
            Param("n_estimators", "int", 50, 600, default=200),
            Param("max_depth", "int", 3, 24, default=12),
            Param("min_samples_leaf", "int", 1, 40, default=3),
            Param("max_features", "float", 0.3, 1.0, default=0.8),
        ]

    def defaults(self) -> dict:
        return {param.name: param.default for param in self.space()}

    def objectives(self, task: str) -> tuple[str, ...]:
        return ("squared_error", "absolute_error") if task == "regression" else ()

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        """relative work hint; absolute_error is several times slower than this"""
        rows, columns = shape
        trees = params.get("n_estimators", 200)
        depth = params.get("max_depth", 12)
        return float(rows * columns * trees * depth)

    def fit(
        self,
        x,
        *,
        objective: str = "squared_error",
        y=None,
        feature_types: list[str] | None = None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        **options,
    ) -> ExtratreesFitted:
        if objective not in CRITERIA:
            raise ModelError(f"no extratrees objective called {objective!r}")
        merged = {**self.defaults(), **(params or {})}
        try:
            trees = int(round(float(merged["n_estimators"])))
            depth = int(round(float(merged["max_depth"])))
            leaf = int(round(float(merged["min_samples_leaf"])))
            features = float(merged["max_features"])
        except (TypeError, ValueError) as exc:
            raise ModelError("extratrees params must be numbers") from exc
        if trees < 1 or depth < 1 or leaf < 1 or not 0 < features <= 1:
            raise ModelError("extratrees params are out of range")
        values = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if values.ndim != 2 or target.shape != (len(values),):
            raise ModelError("extratrees needs a feature matrix and one target per row")
        types = ["q"] * values.shape[1] if feature_types is None else feature_types
        usable = np.isfinite(target)
        values, target = values[usable], target[usable]
        if not len(target):
            raise ModelError("extratrees needs at least one usable target")
        if np.isinf(values).any():
            raise ModelError("extratrees features must be finite or nan")
        from sklearn.ensemble import ExtraTreesRegressor

        pre = Preprocessor.fit(values, types)
        design = pre.transform(values)
        model = ExtraTreesRegressor(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=features,
            criterion=CRITERIA[objective],
            n_jobs=nthread,
            random_state=seed,
        )
        model.fit(design, target)
        residual = model.predict(design) - target
        history = {"metric": "train_mse", "training": [float(residual @ residual / len(target))]}
        return ExtratreesFitted(pre, model, history)
