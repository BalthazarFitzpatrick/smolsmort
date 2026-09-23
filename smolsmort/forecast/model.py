"""xgboost fits for the forecast topics: one call per objective, rounds found on the end of train"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBJECTIVES = {
    "squared": {"objective": "reg:squarederror"},
    "absolute": {"objective": "reg:absoluteerror"},
    "tweedie": {"objective": "reg:tweedie"},
    "poisson": {"objective": "count:poisson"},
    # aft with the logistic distribution diverged to predictions of 1e17+ on test data
    # (2026-09-23), so only normal and extreme are ever offered
    "aft": {"objective": "survival:aft", "aft_loss_distribution": "normal"},
    "logistic": {"objective": "binary:logistic"},
    "softprob": {"objective": "multi:softprob"},
}
CLASSIFIERS = {"logistic", "softprob"}
# these model a non-negative quantity; a negative target is refused rather than silently clipped
NONNEGATIVE = {"tweedie", "poisson", "aft"}


class ModelError(ValueError):
    pass


@dataclass
class Fitted:
    booster: object
    objective: str
    rounds: int
    feature_types: list[str] | None
    classes: list[str] | None = None


def _xgb():
    import xgboost

    return xgboost


def _matrix(x, feature_types):
    xgb = _xgb()
    categorical = bool(feature_types) and "c" in feature_types
    return xgb.DMatrix(
        np.asarray(x, dtype=np.float32),
        feature_types=feature_types,
        enable_categorical=categorical,
        missing=np.nan,
    )


def _labelled(x, feature_types, objective, y, lower, upper):
    dm = _matrix(x, feature_types)
    if objective == "aft":
        dm.set_float_info("label_lower_bound", lower)
        dm.set_float_info("label_upper_bound", upper)
    else:
        dm.set_label(y)
    return dm


def _targets(objective, y, lower, upper, labels, classes):
    """the label arrays for one objective, plus the rows that can be learned from"""
    if objective == "aft":
        if lower is None or upper is None:
            raise ModelError("the aft objective needs lower and upper bounds")
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        usable = np.isfinite(lower) & (lower > 0)
        return None, lower, upper, usable, classes
    if objective in CLASSIFIERS:
        if labels is None:
            raise ModelError(f"the {objective} objective needs labels")
        labels = np.asarray(labels, dtype=object)
        usable = np.array([v is not None and v == v and v != "" for v in labels])
        classes = classes or sorted({str(v) for v in labels[usable]})
        index = {c: i for i, c in enumerate(classes)}
        codes = np.array(
            [index.get(str(v), -1) if ok else -1 for v, ok in zip(labels, usable, strict=True)]
        )
        return codes.astype(np.float32), None, None, usable & (codes >= 0), classes
    y = np.asarray(y, dtype=np.float64)
    usable = np.isfinite(y)
    if objective in NONNEGATIVE and np.any(y[usable] < 0):
        raise ModelError(f"the {objective} objective needs a non-negative target")
    return y, None, None, usable, classes


def fit_model(
    x,
    *,
    objective: str,
    y=None,
    lower=None,
    upper=None,
    labels=None,
    classes: list[str] | None = None,
    feature_types: list[str] | None = None,
    params: dict | None = None,
    inner: float = 0.15,
    max_rounds: int = 2000,
    patience: int = 50,
    seed: int = 0,
    nthread: int = 1,
) -> Fitted:
    """fit one model on rows sorted oldest first; the newest `inner` share picks the rounds, then
    everything is refit with them. nthread stays 1 unless the caller runs in the search worker"""
    if objective not in OBJECTIVES:
        raise ModelError(f"no objective called {objective!r} - known: {', '.join(OBJECTIVES)}")
    xgb = _xgb()
    x = np.asarray(x, dtype=np.float32)
    y, lower, upper, usable, classes = _targets(objective, y, lower, upper, labels, classes)
    if usable.sum() < 20:
        raise ModelError(f"only {int(usable.sum())} rows carry a usable target - too few to fit")
    rows = np.flatnonzero(usable)
    cut = int(len(rows) * (1 - inner))
    head, tail = rows[:cut], rows[cut:]
    config = {**OBJECTIVES[objective], "seed": seed, "nthread": nthread, **(params or {})}
    if objective in CLASSIFIERS and len(classes) > 2:
        config = {**config, **OBJECTIVES["softprob"], "num_class": len(classes)}
    elif objective in CLASSIFIERS:
        config = {**config, **OBJECTIVES["logistic"]}

    def pick(index):
        return _labelled(
            x[index],
            feature_types,
            objective,
            None if y is None else y[index],
            None if lower is None else lower[index],
            None if upper is None else upper[index],
        )

    rounds = max_rounds
    if len(tail) >= 10 and len(head) >= 10:
        probe = xgb.train(
            config,
            pick(head),
            max_rounds,
            evals=[(pick(tail), "inner")],
            early_stopping_rounds=patience,
            verbose_eval=False,
        )
        rounds = probe.best_iteration + 1
    booster = xgb.train(config, pick(rows), rounds, verbose_eval=False)
    return Fitted(booster, objective, rounds, feature_types, classes)


def predict(fitted: Fitted, x) -> np.ndarray:
    """values for regression and aft (in the target's own unit), class probabilities otherwise"""
    out = fitted.booster.predict(_matrix(x, fitted.feature_types))
    if fitted.objective in CLASSIFIERS and out.ndim == 1:
        return np.column_stack([1 - out, out])
    return out
