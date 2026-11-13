"""catboost fits on native missing values and integer-coded categorical columns"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.families.base import Param, probe_import
from smolsmort.forecast.model import ModelError, RoundErrors

OBJECTIVES = ("RMSE", "MAE", "Poisson", "Tweedie")
# catboost needs the tweedie power spelled into the loss name
LOSS_NAMES = {
    "RMSE": "RMSE",
    "MAE": "MAE",
    "Poisson": "Poisson",
    "Tweedie": "Tweedie:variance_power=1.5",
}
LOG_LINK = {"Poisson", "Tweedie"}
INTEGER_PARAMS = ("depth",)
# the profile suggests objectives by generic name; this maps catboost's names onto them
OBJECTIVE_ALIASES = {
    "RMSE": "squared",
    "MAE": "absolute",
    "Poisson": "poisson",
    "Tweedie": "tweedie",
}


def build_pool(cb, x, y, categories):
    """wrap a float matrix as a pool; categorical columns become integer codes [nan is -1]"""
    if not categories:
        return cb.Pool(x, label=y)
    import pandas as pd

    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    for index in categories:
        column = frame.iloc[:, index].to_numpy()
        frame.isetitem(index, np.where(np.isfinite(column), column, -1).astype(np.int64))
    return cb.Pool(frame, label=y, cat_features=categories)


@dataclass
class CatboostFitted:
    model: object
    rounds: int
    categories: list[int]
    link: str
    eval_history: dict | None = None

    def predict(self, x) -> np.ndarray:
        import catboost as cb

        x = np.asarray(x, dtype=np.float64)
        pool = build_pool(cb, x, None, self.categories)
        values = np.asarray(self.model.predict(pool, prediction_type=self.link), dtype=float)
        if not np.isfinite(values).all():
            raise ModelError("catboost returned non-finite predictions")
        return values


class CatboostFamily:
    objective_aliases = OBJECTIVE_ALIASES
    name = "catboost"
    label = "catboost"
    needs = "lag_features"
    pip_extra = "catboost"

    def available(self) -> tuple[bool, str]:
        return probe_import("catboost", self.pip_extra)

    def space(self) -> list[Param]:
        return [
            Param("depth", "int", 4, 10, default=6),
            Param("learning_rate", "log", 0.01, 0.3, default=0.05),
            Param("l2_leaf_reg", "log", 0.1, 30.0, default=3.0),
        ]

    def defaults(self) -> dict:
        return {param.name: param.default for param in self.space()}

    def objectives(self, task: str) -> tuple[str, ...]:
        # regression only; the search filters these through the profile's generic names
        return OBJECTIVES if task == "regression" else ()

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        """relative work hint from rows, columns and leaves per tree"""
        rows, columns = shape
        return float(rows * columns * 2 ** int(round(float(params.get("depth", 6)))))

    def fit(
        self,
        x,
        *,
        objective: str = "MAE",
        y=None,
        feature_types: list[str] | None = None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        inner: float = 0.15,
        max_rounds: int = 2000,
        patience: int = 50,
        round_errors: bool = True,
        metric_transform: str = "none",
        metric_base=None,
    ) -> CatboostFitted:
        if objective not in OBJECTIVES:
            raise ModelError(f"no catboost objective called {objective!r}")
        import catboost as cb

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        usable = np.isfinite(y)
        if objective in LOG_LINK and np.any(y[usable] < 0):
            raise ModelError(f"the {objective} objective needs a non-negative target")
        rows = np.flatnonzero(usable)
        if len(rows) < 20:
            raise ModelError(f"only {len(rows)} rows carry a usable target - too few to fit")
        categories = [i for i, kind in enumerate(feature_types or []) if kind == "c"]
        config = {**self.defaults(), **(params or {})}
        # a log-scale search draws floats; catboost refuses non-integer counts
        for name in INTEGER_PARAMS:
            config[name] = int(round(float(config[name])))
        link = "Exponent" if objective in LOG_LINK else "RawFormulaVal"
        base = {
            **config,
            "loss_function": LOSS_NAMES[objective],
            "random_seed": seed,
            "thread_count": nthread,
            "allow_writing_files": False,
            "verbose": 0,
        }

        cut = int(len(rows) * (1 - inner))
        head, tail = rows[:cut], rows[cut:]
        rounds, history = max_rounds, None
        if patience and len(head) >= 10 and len(tail) >= 10:
            train = build_pool(cb, x[head], y[head], categories)
            check = build_pool(cb, x[tail], y[tail], categories)
            # keep every trained round so the curves line up with the history
            probe = cb.CatBoostRegressor(
                **base,
                iterations=max_rounds,
                od_type="Iter",
                od_wait=patience,
                use_best_model=False,
            )
            probe.fit(train, eval_set=check)
            rounds = int(probe.get_best_iteration()) + 1
            evals = probe.get_evals_result()
            metric = LOSS_NAMES[objective].split(":")[0]
            training = list(evals.get("learn", {}).get(metric, []))
            validation = list(evals.get("validation", {}).get(metric, []))
            if (
                training
                and len(training) == len(validation)
                and np.isfinite([training, validation]).all()
            ):
                history = {"metric": metric, "training": training, "validation": validation}
                if round_errors:
                    errors = RoundErrors(
                        y[tail],
                        metric_transform,
                        None if metric_base is None else metric_base[tail],
                    )
                    for staged in probe.staged_predict(check, prediction_type=link, eval_period=1):
                        errors.add(staged)
                    curves = errors.curves(len(validation))
                    if curves:
                        history["curves"] = curves
        model = cb.CatBoostRegressor(**base, iterations=rounds)
        model.fit(build_pool(cb, x[rows], y[rows], categories))
        return CatboostFitted(model, rounds, categories, link, history)
