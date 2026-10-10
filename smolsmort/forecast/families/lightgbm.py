"""lightgbm fits on native missing values and translated categorical columns"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.families.base import Param, probe_import
from smolsmort.forecast.model import ModelError

OBJECTIVES = ("regression_l1", "regression", "poisson", "tweedie")


@dataclass
class LightgbmFitted:
    booster: object
    rounds: int
    nthread: int
    eval_history: dict | None = None

    def predict(self, x) -> np.ndarray:
        values = np.asarray(
            self.booster.predict(np.asarray(x, dtype=np.float32), num_threads=self.nthread)
        )
        if not np.isfinite(values).all():
            raise ModelError("lightgbm returned non-finite predictions")
        return values


class LightgbmFamily:
    name = "lightgbm"
    label = "lightgbm"
    needs = "lag_features"
    pip_extra = "lightgbm"

    def available(self) -> tuple[bool, str]:
        return probe_import("lightgbm", self.pip_extra)

    def space(self) -> list[Param]:
        return [
            Param("num_leaves", "log", 7, 255, default=31),
            Param("learning_rate", "log", 0.005, 0.3, default=0.05),
            Param("min_child_samples", "int", 5, 100, default=20),
            Param("feature_fraction", "float", 0.5, 1.0, default=1.0),
            Param("bagging_fraction", "float", 0.5, 1.0, default=1.0),
            Param("lambda_l2", "log", 1e-8, 100.0, default=1e-8),
            Param("max_depth", "int", -1, 16, default=-1),
            Param("objective", "choice", choices=OBJECTIVES, default="regression_l1"),
        ]

    def defaults(self) -> dict:
        return {param.name: param.default for param in self.space()}

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        """relative work hint from rows, columns and leaves"""
        rows, columns = shape
        return float(rows * columns * params.get("num_leaves", 31))

    def fit(
        self,
        x,
        *,
        objective: str = "regression_l1",
        y=None,
        feature_types: list[str] | None = None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        inner: float = 0.15,
        max_rounds: int = 2000,
        patience: int = 50,
    ) -> LightgbmFitted:
        if objective not in OBJECTIVES:
            raise ModelError(f"no lightgbm objective called {objective!r}")
        import lightgbm as lgb

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        usable = np.isfinite(y)
        if objective in {"poisson", "tweedie"} and np.any(y[usable] < 0):
            raise ModelError(f"the {objective} objective needs a non-negative target")
        rows = np.flatnonzero(usable)
        if len(rows) < 20:
            raise ModelError(f"only {len(rows)} rows carry a usable target - too few to fit")
        categories = [i for i, kind in enumerate(feature_types or []) if kind == "c"]
        config = {
            **self.defaults(),
            **(params or {}),
            "objective": objective,
            "deterministic": True,
            "force_col_wise": True,
            "seed": seed,
            "num_threads": nthread,
            "verbosity": -1,
        }
        # lightgbm ignores bagging_fraction unless bagging is enabled
        config["bagging_freq"] = 1 if config["bagging_fraction"] < 1 else 0

        def pick(index, reference=None):
            return lgb.Dataset(
                x[index], label=y[index], categorical_feature=categories, reference=reference
            )

        cut = int(len(rows) * (1 - inner))
        head, tail = rows[:cut], rows[cut:]
        rounds, history = max_rounds, None
        if patience and len(head) >= 10 and len(tail) >= 10:
            results = {}
            train = pick(head)
            probe = lgb.train(
                config,
                train,
                num_boost_round=max_rounds,
                valid_sets=[train, pick(tail, train)],
                valid_names=["training", "validation"],
                callbacks=[
                    lgb.early_stopping(patience, verbose=False),
                    lgb.record_evaluation(results),
                ],
            )
            rounds = probe.best_iteration or max_rounds
            metric = next(iter(results.get("validation", {})), None)
            training = results.get("training", {}).get(metric, [])
            validation = results.get("validation", {}).get(metric, [])
            if (
                training
                and len(training) == len(validation)
                and np.isfinite([training, validation]).all()
            ):
                history = {"metric": metric, "training": training, "validation": validation}
        booster = lgb.train(config, pick(rows), num_boost_round=rounds)
        return LightgbmFitted(booster, rounds, nthread, history)
