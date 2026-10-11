"""the existing xgboost implementation behind the series family contract"""

from __future__ import annotations

from dataclasses import dataclass

from smolsmort.forecast import model
from smolsmort.forecast.families.base import Param, probe_import


@dataclass
class XgboostFitted:
    fitted: model.Fitted

    @property
    def eval_history(self) -> dict | None:
        return self.fitted.eval_history

    def predict(self, x):
        return model.predict(self.fitted, x)


class XgboostFamily:
    name = "xgboost"
    label = "xgboost"
    needs = "lag_features"
    pip_extra = "forecast"

    def available(self) -> tuple[bool, str]:
        return probe_import("xgboost", self.pip_extra)

    def space(self) -> list[Param]:
        from smolsmort.forecast.search import DEFAULTS, OBJECTIVE_DEFAULTS, OBJECTIVE_SPACE, SPACE

        specs = dict(SPACE)
        defaults = dict(DEFAULTS)
        for objective, space in OBJECTIVE_SPACE.items():
            specs.update(space)
            defaults.update(OBJECTIVE_DEFAULTS[objective])
        params = []
        for name, spec in specs.items():
            if isinstance(spec[0], str):
                params.append(Param(name, "choice", choices=tuple(spec), default=defaults[name]))
            else:
                lo, hi, kind = spec
                params.append(
                    Param(name, "float" if kind == "lin" else kind, lo, hi, default=defaults[name])
                )
        return params

    def defaults(self) -> dict:
        return {param.name: param.default for param in self.space()}

    def objectives(self, task: str) -> tuple[str, ...]:
        return tuple(
            name
            for name in model.OBJECTIVES
            if (name in model.CLASSIFIERS) == (task == "classification")
        )

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        """relative work hint from rows, columns and depth; not a runtime estimate"""
        rows, columns = shape
        return float(rows * columns * params.get("max_depth", 6))

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
    ) -> XgboostFitted:
        return XgboostFitted(
            model.fit_model(
                x,
                objective=objective,
                y=y,
                feature_types=feature_types,
                params=params,
                seed=seed,
                nthread=nthread,
                **options,
            )
        )
