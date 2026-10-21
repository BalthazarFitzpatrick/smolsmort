"""one candidate recipe, scored in phases: `val` fits on train and predicts validation (every
candidate), `test` refits on train + val for the frozen winner only, `final` fits on everything and
predicts the rows nobody knows the answer to yet"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast.families import get_family
from smolsmort.forecast.features import (
    buckets_for,
    entity_features,
    family_aliases,
    family_sort_key,
    split_family_name,
    timeseries_features,
)
from smolsmort.forecast.importance import combine_importance, measure_importance
from smolsmort.forecast.model import CLASSIFIERS, NONNEGATIVE, fit_model, predict
from smolsmort.forecast.profile import build_profile
from smolsmort.forecast.spec import ANCHOR, LOWER, ROW, SERIES, STEP, UNIT_DAYS, UPPER, PrepSpec
from smolsmort.forecast.splits import entity_cuts, labels_as_of, timeseries_cuts
from smolsmort.forecast.tables import is_text, read_table
from smolsmort.forecast.transform import get_transform


class PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class Genome:
    families: tuple[str, ...]
    objective: str
    params: tuple[tuple[str, object], ...]
    family: str = "xgboost"
    transform: str = "none"

    def __post_init__(self):
        get_transform(self.transform)
        try:
            get_family(self.family)
        except KeyError as exc:
            raise PipelineError(
                f"unknown family {self.family!r}; install its provider with "
                f"uv sync --extra {self.family} and register it"
            ) from exc

    def to_dict(self) -> dict:
        data = {
            "families": list(self.families),
            "objective": self.objective,
            "params": dict(self.params),
        }
        if self.family != "xgboost":
            data["family"] = self.family
        if self.transform != "none":
            data["transform"] = self.transform
        return data

    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def make_genome(
    families, objective: str, params: dict, family: str = "xgboost", transform: str = "none"
) -> Genome:
    return Genome(
        tuple(sorted(set(families), key=family_sort_key)),
        objective,
        tuple(sorted(params.items())),
        family,
        transform,
    )


def genome_from_dict(
    data: dict, family: str | None = None, *, spec: PrepSpec | None = None
) -> Genome:
    aliases = family_aliases(spec, data["families"]) if spec is not None else {}
    return make_genome(
        [aliases.get(name, name) for name in data["families"]],
        data["objective"],
        data["params"],
        family=data.get("family", "xgboost") if family is None else family,
        transform=data.get("transform", "none"),
    )


@dataclass
class Scored:
    fitness: float
    contrib: np.ndarray
    metrics: dict
    eval_history: dict | None = None
    prediction: np.ndarray | None = None


# ---------------------------------------------------------------- workspace


@dataclass
class Workspace:
    spec: PrepSpec
    folder: Path
    summary: dict
    profile: dict
    families: list[str]
    unit: str
    # entity mode
    table: dict | None = None
    features: object = None
    masks: dict | None = None
    bounds: dict | None = None
    predict_rows: np.ndarray | None = None
    # time series mode
    frames: dict | None = None
    series_masks: dict | None = None
    test_steps: int = 0
    panel: dict | None = None
    transform: str = "none"
    excluded_series: list[dict] = field(default_factory=list)
    series_folds: tuple[tuple[np.datetime64, np.datetime64], ...] = ()

    @property
    def mode(self) -> str:
        return self.spec.mode


def load_workspace(
    spec: PrepSpec, folder: Path, summary: dict, *, transform: str = "none"
) -> Workspace:
    """everything a search needs, read and built once: tables, cuts, profile, candidate features"""
    folder = Path(folder)
    target_transform = get_transform(transform)
    if spec.mode == "entity":
        if transform != "none":
            raise PipelineError("target transforms require time series regression")
        return _entity_workspace(spec, folder, summary)
    if spec.task == "classification":
        raise PipelineError("classification requires entity mode; time series runs use regression")
    ws = _timeseries_workspace(spec, folder, summary)
    ws.transform = transform
    if transform != "none":
        target_transform.validate_target(ws.panel[spec.named("target")[0]])
    return ws


def _entity_workspace(spec, folder, summary) -> Workspace:
    table = read_table(folder / "rows.parquet", order_by=f"{ANCHOR}, {ROW}")
    as_of = dt.date.fromisoformat(summary["as_of"])
    censored = spec.censor is not None
    cuts = entity_cuts(table[ANCHOR], as_of, maturity_weeks=26 if censored else 0)
    train, val, test = cuts.masks(table[ANCHOR])
    unit_days = UNIT_DAYS[spec.censor.unit] if censored else 7.0
    bounds = {"final": (table[LOWER], table[UPPER])}
    for phase, cut in (("val", cuts.val_start), ("test", cuts.test_start)):
        if censored:
            bounds[phase] = labels_as_of(table[ANCHOR], table[LOWER], table[UPPER], cut, unit_days)
        else:
            bounds[phase] = (table[LOWER], table[UPPER])
    profiled = dict(table)
    profiled[LOWER], profiled[UPPER] = bounds["val"]
    profile = build_profile(spec, profiled, train)
    features = entity_features(spec, table, train)
    predict_ids = read_table(folder / "predict.parquet")[ROW]
    ws = Workspace(spec, folder, summary, profile, [], spec.censor.unit if censored else "week")
    ws.table, ws.features = table, features
    ws.masks = {"train": train, "val": val, "test": test}
    ws.bounds = bounds
    ws.predict_rows = np.isin(table[ROW], predict_ids)
    ws.families = sorted(
        set(features.families) - _excluded_families(profile, features.families), key=family_sort_key
    )
    return ws


def _timeseries_workspace(spec, folder, summary) -> Workspace:
    panel = read_table(folder / "panel.parquet", order_by=f"{SERIES}, {STEP}")
    scaffold = read_table(folder / "scaffold.parquet")
    inputs = read_table(folder / "inputs.parquet") if spec.named("input") else None
    scaffold, excluded = _filter_scaffold(spec, panel, scaffold)
    steps = np.unique(panel[STEP])
    cuts = timeseries_cuts(len(steps), spec.horizon)
    val_start, test_start = steps[cuts.val_start], steps[cuts.test_start]
    train = panel[STEP] < val_start
    profile = build_profile(spec, panel, train)
    genes = profile["genes"]
    unit = summary["step"]
    frames, masks = {}, {}
    for bucket in buckets_for(spec.horizon):
        frame = timeseries_features(
            spec,
            panel,
            scaffold,
            bucket,
            unit,
            train,
            period=genes.get("period"),
            measure_lags=genes.get("measure_lags"),
            inputs=inputs,
        )
        step = frame.step
        known = ~frame.future & np.isfinite(frame.y)
        frames[bucket] = frame
        masks[bucket] = {
            "train": known & (step < val_start.astype("datetime64[D]")),
            "val": known
            & (step >= val_start.astype("datetime64[D]"))
            & (step < test_start.astype("datetime64[D]")),
            "test": known & (step >= test_start.astype("datetime64[D]")),
            "future": frame.future,
        }
    ws = Workspace(spec, folder, summary, profile, [], unit)
    ws.frames, ws.series_masks = frames, masks
    ws.series_folds = tuple(
        (steps[start].astype("datetime64[D]"), steps[end].astype("datetime64[D]"))
        for start, end in cuts.folds
    )
    ws.panel = panel
    ws.excluded_series = excluded
    ws.test_steps = int(cuts.steps - cuts.test_start)
    families = set().union(*(set(f.features.families) for f in frames.values()))
    ws.families = sorted(families - _excluded_families(profile, families), key=family_sort_key)
    return ws


def _filter_scaffold(spec, panel, scaffold) -> tuple[dict, list[dict]]:
    """keep future rows only for series with a finite target at the panel origin"""
    target = panel[spec.named("target")[0]]
    origin = panel[STEP].max()
    dims = spec.named("dimension")
    excluded = []
    for key in np.unique(panel[SERIES]):
        mine = np.flatnonzero(panel[SERIES] == key)
        observed = mine[np.isfinite(target[mine])]
        last_seen = panel[STEP][observed].max() if len(observed) else None
        if last_seen is not None and last_seen == origin:
            continue
        excluded.append(
            {
                "series": str(key),
                "dims": {name: panel[name][mine[0]] for name in dims},
                "last_seen": str(last_seen.astype("datetime64[D]"))
                if last_seen is not None
                else None,
            }
        )
    if not excluded:
        return scaffold, excluded
    keep = ~np.isin(scaffold[SERIES], [entry["series"] for entry in excluded])
    return {name: values[keep] for name, values in scaffold.items()}, excluded


def _excluded_families(profile, families) -> set[str]:
    leaks = set(profile["genes"].get("exclude", []))
    return {
        name
        for name in families
        if (parts := split_family_name(name))[0] in leaks and parts[2] in {"measure", "log", "lead"}
    }


# ---------------------------------------------------------------- scoring


def score(
    ws: Workspace, genome: Genome, *, nthread: int = 1, sample: np.ndarray | None = None
) -> Scored:
    """validation fitness for one genome; `sample` restricts the training rows (successive halving)"""
    histories = []
    if ws.mode == "entity":
        pred, truth, periods = _entity_phase(ws, genome, "val", nthread, sample, histories)
    else:
        pred, truth, periods = _timeseries_phase(ws, genome, "val", nthread, sample, histories)
    result = _fitness(ws, pred, truth, periods)
    result.prediction = pred
    if ws.mode == "timeseries" and get_family(genome.family).needs == "raw_series":
        from smolsmort.forecast.families.stats import season_length

        result.metrics["season_length"] = season_length(ws.profile)
    result.eval_history = histories[0] if histories else None
    return result


def _fitness(ws, pred, truth, periods) -> Scored:
    folds = getattr(ws, "series_folds", ())
    if not folds:
        return _period_fitness(ws, pred, truth, periods)
    scores = []
    for start, end in folds:
        rows = (periods >= start.astype(int)) & (periods < end.astype(int))
        if rows.any():
            scores.append(_period_fitness(ws, pred[rows], truth[rows], periods[rows]))
    if not scores:
        return _period_fitness(ws, pred, truth, periods)
    fold_scores = [s.fitness for s in scores]
    fitness = float(np.mean(fold_scores))
    # equal fold weights keep paired contributions aligned with the ranked fitness
    contrib = np.concatenate([s.contrib / len(scores) for s in scores])
    return Scored(
        fitness,
        contrib,
        {
            "wape": fitness,
            "mae": ev.mae(truth, pred),
            "fold_scores": fold_scores,
            "fold_count": len(scores),
        },
    )


def _period_fitness(ws, pred, truth, periods) -> Scored:
    classification = ws.spec.task == "classification"
    if classification:
        codes, probs = truth, pred
        losses = -np.log(np.clip(probs[np.arange(len(codes)), codes], 1e-12, 1))
        per_row, denom = losses, len(losses)
        metrics = {"log_loss": float(losses.mean()), "brier": ev.brier(codes, probs)}
    else:
        per_row, denom = np.abs(truth - pred), max(float(np.abs(truth).sum()), 1e-12)
        metrics = {"wape": ev.wape(truth, pred), "mae": ev.mae(truth, pred)}
    unique, inverse = np.unique(periods, return_inverse=True)
    contrib = np.bincount(inverse, weights=per_row, minlength=len(unique)) / denom
    return Scored(float(contrib.sum()), contrib, metrics)


def _entity_design(ws, genome):
    x, _, types = ws.features.pick(set(genome.families))
    if not x.shape[1]:
        raise PipelineError("a genome with no feature families cannot be fitted")
    return x, types


def _entity_fit(ws, genome, rows, phase, nthread):
    x, types = _entity_design(ws, genome)
    lower, upper = ws.bounds[phase]
    target = ws.table[ws.spec.named("target")[0]]
    params = dict(genome.params)
    if genome.objective in CLASSIFIERS:
        return fit_model(
            x[rows],
            objective=genome.objective,
            labels=target[rows],
            feature_types=types,
            params=params,
            nthread=nthread,
        ), x
    if genome.objective == "aft":
        keep = rows & np.isfinite(lower) & (lower > 0)
        return fit_model(
            x[keep],
            objective="aft",
            lower=lower[keep],
            upper=upper[keep],
            feature_types=types,
            params=params,
            nthread=nthread,
        ), x
    known = rows & np.isfinite(lower) & (lower == upper)
    return fit_model(
        x[known],
        objective=genome.objective,
        y=lower[known],
        feature_types=types,
        params=params,
        nthread=nthread,
    ), x


def _entity_phase(ws, genome, phase, nthread, sample=None, histories=None):
    """(predictions, truth, period keys) on the phase's evaluation rows"""
    masks = ws.masks
    fit_rows = masks["train"] if phase == "val" else masks["train"] | masks["val"]
    if sample is not None:
        fit_rows = fit_rows & sample
    eval_rows = masks["val"] if phase == "val" else masks["test"]
    target = ws.table[ws.spec.named("target")[0]]
    fitted, x = _entity_fit(ws, genome, fit_rows, phase, nthread)
    if histories is not None:
        histories.append(fitted.eval_history)
    if ws.spec.task == "classification":
        known = eval_rows & np.array([v is not None and v == v and v != "" for v in target])
        classes = fitted.classes
        index = {c: i for i, c in enumerate(classes)}
        keep = known & np.array([str(v) in index for v in target])
        codes = np.array([index[str(v)] for v in target[keep]])
        return predict(fitted, x[keep]), codes, _weeks(ws.table[ANCHOR][keep])
    keep = eval_rows & np.isfinite(target)
    return predict(fitted, x[keep]), target[keep], _weeks(ws.table[ANCHOR][keep])


def _weeks(days) -> np.ndarray:
    return (np.asarray(days).astype("datetime64[D]").astype(int) - 4) // 7


def _timeseries_phase(ws, genome, phase, nthread, sample=None, histories=None):
    preds, truths, periods = [], [], []
    for bucket, frame in ws.frames.items():
        masks = ws.series_masks[bucket]
        eval_rows = masks["val"] if phase == "val" else masks["test"]
        if phase == "val":

            def predict_fold(fit_rows, fold_rows, frame=frame, bucket=bucket):
                if sample is not None:
                    fit_rows = fit_rows & np.isin(frame.series, sample)
                return _timeseries_predict(
                    ws, genome, frame, fit_rows, fold_rows, bucket, nthread, histories
                )

            prediction = _timeseries_validation(ws, bucket, predict_fold)
        else:
            fit_rows = masks["train"] | masks["val"]
            prediction = _timeseries_predict(
                ws, genome, frame, fit_rows, eval_rows, bucket, nthread, histories
            )
        preds.append(prediction)
        truths.append(frame.y[eval_rows])
        periods.append(frame.step[eval_rows].astype(int))
    return np.concatenate(preds), np.concatenate(truths), np.concatenate(periods)


def _validation_masks(ws, bucket):
    masks, frame = ws.series_masks[bucket], ws.frames[bucket]
    folds = getattr(ws, "series_folds", ())
    if not folds:
        yield masks["train"], masks["val"]
        return
    known = masks["train"] | masks["val"]
    for start, end in folds:
        yield (
            known & (frame.step < start),
            masks["val"] & (frame.step >= start) & (frame.step < end),
        )


def _timeseries_validation(ws, bucket, predict_fold):
    masks = ws.series_masks[bucket]
    prediction = np.empty(int(masks["val"].sum()))
    for fit_rows, eval_rows in _validation_masks(ws, bucket):
        assert not np.any((fit_rows | eval_rows) & masks["test"]), (
            "test targets are sealed during search"
        )
        if eval_rows.any():
            prediction[eval_rows[masks["val"]]] = predict_fold(fit_rows, eval_rows)
    return prediction


def _timeseries_fit(ws, genome, frame, rows, nthread):
    family = get_family(genome.family)
    available, reason = family.available()
    if not available:
        raise PipelineError(f"family {genome.family!r} is unavailable; {reason}")
    if genome.transform == "logdiff" and genome.objective in NONNEGATIVE:
        raise PipelineError(
            f"logdiff transform requires a signed-target objective, got {genome.objective!r}"
        )
    x, _, types = frame.features.pick(set(genome.families))
    if not x.shape[1]:
        raise PipelineError("a genome with no feature families cannot be fitted")
    # oldest first, so the early-stopping slice is the newest steps rather than the last series
    order = np.flatnonzero(rows)[np.argsort(frame.step[rows], kind="stable")]
    base = frame.base[order] if genome.transform == "logdiff" else None
    fitted = family.fit(
        x[order],
        objective=genome.objective,
        y=get_transform(genome.transform).fwd(frame.y[order], base),
        feature_types=types,
        params=dict(genome.params),
        nthread=nthread,
    )
    return fitted, x


def _timeseries_raw_fit(ws, genome, frame, rows, bucket, nthread):
    from smolsmort.forecast.families.stats import RawSeriesTask, season_length

    # bound raw target reads before transforms or statistical providers can inspect them
    keep = (
        ws.panel[STEP] <= frame.step[rows].max()
        if rows.any()
        else np.zeros(len(ws.panel[STEP]), dtype=bool)
    )
    panel = {name: values[keep] for name, values in ws.panel.items()}
    target = ws.spec.named("target")[0]
    if genome.transform != "none":
        transform = get_transform(genome.transform)
        transform.validate_target(panel[target])
        if genome.family != "snaive":
            # logdiff stats fit levels; the cutoff offset is applied after forecasting
            panel = {**panel, target: np.log1p(panel[target])}
    task = RawSeriesTask(
        panel,
        target,
        ws.unit,
        frame,
        rows,
        bucket,
        season_length(ws.profile),
    )
    fitted = get_family(genome.family).fit(
        task, objective=genome.objective, params=dict(genome.params), nthread=nthread
    )
    return fitted, task


def _timeseries_predict(
    ws, genome, frame, fit_rows, eval_rows, bucket, nthread, histories=None, importance=None
):
    transform = get_transform(genome.transform)
    if get_family(genome.family).needs == "raw_series":
        fitted, task = _timeseries_raw_fit(ws, genome, frame, eval_rows, bucket, nthread)
        prediction = fitted.predict(task)
        if genome.transform != "none":
            base = _raw_origin_base(ws, frame, eval_rows)
            if genome.family == "snaive":
                prediction = transform.fwd(prediction, base)
            elif genome.transform == "logdiff":
                prediction = prediction - np.log1p(base)
            prediction = transform.inv(prediction, base)
    else:
        fitted, x = _timeseries_fit(ws, genome, frame, fit_rows, nthread)
        if importance is not None:
            _, names, _ = frame.features.pick(set(genome.families))
            importance.append(
                measure_importance(genome.family, fitted, names, x[fit_rows & np.isfinite(frame.y)])
            )
        base = frame.base[eval_rows] if genome.transform == "logdiff" else None
        prediction = transform.inv(fitted.predict(x[eval_rows]), base)
    # one actual fit, never a blend of curves from different horizon buckets
    if histories is not None and not histories:
        history = fitted.eval_history
        histories.append({**history, "bucket": list(bucket)} if history else None)
    return prediction


def _raw_origin_base(ws, frame, rows):
    """the last observed target at each statistical forecast's cutoff"""
    base = np.full(int(rows.sum()), np.nan)
    keys, origins = frame.series[rows], frame.origin[rows]
    target = ws.spec.named("target")[0]
    for key in np.unique(keys):
        mine = (ws.panel[SERIES] == key) & (ws.panel[STEP] <= origins[keys == key].max())
        steps, values = ws.panel[STEP][mine], ws.panel[target][mine]
        known = np.isfinite(values)
        steps, values = steps[known], values[known]
        order = np.argsort(steps)
        selected = keys == key
        positions = np.searchsorted(steps[order], origins[selected], side="right") - 1
        valid = positions >= 0
        indices = np.flatnonzero(selected)[valid]
        base[indices] = values[order][positions[valid]]
    return base


def fit_importance(ws, genome, *, nthread=1):
    """fit a family outside the ensemble on known rows, without evaluating its test split"""
    if get_family(genome.family).needs != "lag_features":
        return []
    scores = []
    for bucket, frame in ws.frames.items():
        masks = ws.series_masks[bucket]
        rows = masks["train"] | masks["val"] | masks["test"]
        fitted, x = _timeseries_fit(ws, genome, frame, rows, nthread)
        _, names, _ = frame.features.pick(set(genome.families))
        scores.append(
            measure_importance(genome.family, fitted, names, x[rows & np.isfinite(frame.y)])
        )
    return scores


# ---------------------------------------------------------------- the frozen winner


def finish(
    ws: Workspace,
    genome: Genome,
    *,
    nthread: int = 1,
    members=None,
    validation=None,
    prediction_cache=None,
    importance_cache=None,
) -> dict:
    """band from validation, the verdict from the untouched test split, and the forecast itself"""
    if ws.mode == "entity":
        result = _finish_entities(ws, genome, nthread)
    else:
        result = _finish_timeseries(
            ws, genome, nthread, members, validation, prediction_cache, importance_cache
        )
    return {**result, "transform": genome.transform}


def _finish_entities(ws, genome, nthread) -> dict:
    spec = ws.spec
    classification = spec.task == "classification"
    val_pred, val_truth, _ = _entity_phase(ws, genome, "val", nthread)
    test_pred, test_truth, _ = _entity_phase(ws, genome, "test", nthread)
    target = ws.table[spec.named("target")[0]]
    if classification:
        prior = _class_prior(ws, target)
        model_error = ev.log_loss(test_truth, test_pred)
        baseline_error = ev.log_loss(test_truth, np.tile(prior, (len(test_truth), 1)))
        verdict = ev.judge(
            model_error=model_error,
            baseline_error=baseline_error,
            band_coverage=None,
            test_rows=len(test_truth),
            needs_band=False,
            metric="log_loss",
        )
        band = None
    else:
        band = ev.fit_band(val_truth, val_pred)
        nonnegative = (
            genome.objective in NONNEGATIVE
            or ws.profile["columns"][spec.named("target")[0]].get("min", -1) >= 0
        )
        lo, hi = ev.apply_band(band, test_pred, nonnegative)
        baseline = _entity_baseline(ws, target)
        model_error = ev.wape(test_truth, test_pred)
        baseline_error = ev.wape(test_truth, baseline)
        lower_t, upper_t = ws.bounds["test"]
        seen = (ws.masks["train"] | ws.masks["val"]) & np.isfinite(lower_t) & (lower_t == upper_t)
        verdict = ev.judge(
            model_error=model_error,
            baseline_error=baseline_error,
            band_coverage=ev.coverage(test_truth, lo, hi),
            test_rows=len(test_truth),
            band_width=_median_width(lo, hi),
            spread=_spread(lower_t[seen]),
        )
    everything = np.ones(len(target), dtype=bool)
    fitted, x = _entity_fit(ws, genome, everything, "final", nthread)
    _, names, _ = ws.features.pick(set(genome.families))
    importance = combine_importance("xgboost", [measure_importance("xgboost", fitted, names)])
    rows = ws.predict_rows
    out = {"row": ws.table[ROW][rows], "anchor": ws.table[ANCHOR][rows]}
    if classification:
        probs = predict(fitted, x[rows])
        out["class"] = np.array(fitted.classes, dtype=object)[probs.argmax(axis=1)]
        out["probability"] = probs.max(axis=1)
    else:
        pred = predict(fitted, x[rows])
        lo, hi = ev.apply_band(band, pred, nonnegative=True)
        out.update(prediction=pred, lower=lo, upper=hi)
        if spec.censor:
            days = UNIT_DAYS[spec.censor.unit]
            anchor = ws.table[ANCHOR][rows].astype("datetime64[D]")
            out["predicted_date"] = anchor + (pred * days).round().astype("timedelta64[D]")
    return {
        "verdict": verdict,
        "band": None if band is None else band.__dict__,
        "test": {
            "prediction": test_pred.tolist() if not classification else None,
            "rows": len(test_truth),
        },
        "validation": {"truth": val_truth, "prediction": val_pred},
        "forecast": out,
        "model_error": model_error,
        "baseline_error": baseline_error,
        "importance": importance,
    }


def _class_prior(ws, target) -> np.ndarray:
    train = ws.masks["train"] | ws.masks["val"]
    values = [str(v) for v in target[train] if v is not None and v == v and v != ""]
    classes = sorted(set(values))
    counts = np.array([values.count(c) for c in classes], dtype=float)
    return counts / counts.sum()


def _entity_baseline(ws, target) -> np.ndarray:
    """the group median of the dimension that best predicts validation - the naive answer"""
    masks, spec = ws.masks, ws.spec
    lower, upper = ws.bounds["val"]
    known_val = masks["train"] & np.isfinite(lower) & (lower == upper)
    val_rows = masks["val"] & np.isfinite(target)
    candidates = [d for d in spec.inputs() if is_text(ws.table[d])] or [None]
    best, best_error = None, np.inf
    for dim in candidates:
        keys = ws.table[dim] if dim else np.zeros(len(target), dtype=object)
        guess = ev.group_median(keys[known_val], lower[known_val], keys[val_rows])
        error = ev.wape(target[val_rows], guess)
        if error < best_error:
            best, best_error = dim, error
    lower_t, upper_t = ws.bounds["test"]
    known_test = (masks["train"] | masks["val"]) & np.isfinite(lower_t) & (lower_t == upper_t)
    test_rows = masks["test"] & np.isfinite(target)
    keys = ws.table[best] if best else np.zeros(len(target), dtype=object)
    return ev.group_median(keys[known_test], lower_t[known_test], keys[test_rows])


def _finish_timeseries(
    ws, genome, nthread, members=None, validation=None, prediction_cache=None, importance_cache=None
) -> dict:
    profile = ws.profile
    period = profile["genes"].get("period")
    bands, val_rows, test_rows, forecast = {}, [], [], []
    test_pred_all, test_truth_all, base_all = [], [], []
    test_lo, test_hi = [], []
    offset = 0
    single = members is None
    members = members or [genome]

    def predict_members(frame, fit_rows, eval_rows, bucket, collect=False):
        predictions = []
        for member in members:
            key = (member.key(), bucket, fit_rows.tobytes(), eval_rows.tobytes())
            prediction = prediction_cache.get(key) if prediction_cache is not None else None
            if prediction is None:
                options = {}
                if collect and importance_cache is not None:
                    options["importance"] = importance_cache.setdefault(member.key(), [])
                prediction = _timeseries_predict(
                    ws, member, frame, fit_rows, eval_rows, bucket, nthread, **options
                )
                if prediction_cache is not None:
                    prediction_cache[key] = prediction
            predictions.append(prediction)
        return np.mean(predictions, axis=0)

    for bucket, frame in ws.frames.items():
        masks = ws.series_masks[bucket]
        count = int(masks["val"].sum())
        val_pred = (
            np.asarray(validation)[offset : offset + count]
            if validation is not None
            else _timeseries_validation(
                ws,
                bucket,
                lambda fit_rows, eval_rows, frame=frame, bucket=bucket: predict_members(
                    frame, fit_rows, eval_rows, bucket
                ),
            )
        )
        offset += count
        band = ev.fit_band(frame.y[masks["val"]], val_pred)
        bands[f"{bucket[0]}-{bucket[1]}"] = None if band is None else band.__dict__
        val_rows.append(_timeseries_rows(frame, masks["val"], val_pred, bucket))
        test_pred = predict_members(frame, masks["train"] | masks["val"], masks["test"], bucket)
        lo, hi = ev.apply_band(band, test_pred, nonnegative=True)
        test_pred_all.append(test_pred)
        test_truth_all.append(frame.y[masks["test"]])
        test_lo.append(lo)
        test_hi.append(hi)
        base_all.append(_timeseries_baseline(frame, masks["test"], bucket, period))
        test_rows.append(_timeseries_rows(frame, masks["test"], test_pred, bucket))
        future_pred = predict_members(
            frame,
            masks["train"] | masks["val"] | masks["test"],
            masks["future"],
            bucket,
            collect=True,
        )
        lo, hi = ev.apply_band(band, future_pred, nonnegative=True)
        rows = _timeseries_rows(frame, masks["future"], future_pred, bucket)
        rows.update(lower=lo, upper=hi)
        forecast.append(rows)
    truth, pred = np.concatenate(test_truth_all), np.concatenate(test_pred_all)
    first = next(iter(ws.frames))
    seen = ws.series_masks[first]["train"] | ws.series_masks[first]["val"]
    model_error, baseline_error = ev.wape(truth, pred), ev.wape(truth, np.concatenate(base_all))
    validation_rows = _stack(val_rows)
    validation_score = _fitness(
        ws,
        validation_rows["prediction"],
        validation_rows["actual"],
        validation_rows["step"].astype(int),
    )
    verdict = ev.judge(
        model_error=model_error,
        baseline_error=baseline_error,
        band_coverage=ev.coverage(truth, np.concatenate(test_lo), np.concatenate(test_hi)),
        test_rows=len(truth),
        test_steps=ws.test_steps,
        band_width=_median_width(np.concatenate(test_lo), np.concatenate(test_hi)),
        spread=_spread(ws.frames[first].y[seen]),
        val_error=validation_score.fitness,
    )
    metadata = {}
    if get_family(genome.family).needs == "raw_series":
        from smolsmort.forecast.families.stats import season_length

        metadata["season_length"] = season_length(ws.profile)
    return {
        **metadata,
        "excluded_series": ws.excluded_series,
        "verdict": verdict,
        "bands": bands,
        "validation": validation_rows,
        "fold_scores": validation_score.metrics.get("fold_scores", [validation_score.fitness]),
        "fold_count": validation_score.metrics.get("fold_count", 1),
        "val_error": validation_score.fitness,
        "val_test_gap": model_error - validation_score.fitness,
        "test": _stack(test_rows),
        "forecast": _stack(forecast),
        "model_error": model_error,
        "baseline_error": baseline_error,
        "importance": combine_importance(genome.family, importance_cache.get(genome.key(), []))
        if single and importance_cache is not None
        else None,
    }


def _timeseries_rows(frame, mask, pred, bucket) -> dict:
    return {
        "series": frame.series[mask],
        "step": frame.step[mask],
        "bucket": np.full(int(mask.sum()), f"{bucket[0]}-{bucket[1]}", dtype=object),
        "actual": frame.y[mask],
        "prediction": pred,
    }


def _timeseries_baseline(frame, mask, bucket, period) -> np.ndarray:
    """seasonal naive when a season reaches past the bucket, else the value at the origin"""
    names = frame.features.names
    column = (
        f"lag_{period}"
        if period and period >= bucket[1] and f"lag_{period}" in names
        else f"lag_{bucket[1]}"
    )
    return frame.features.x[mask, names.index(column)].astype(float)


def _median_width(lo, hi) -> float:
    widths = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    widths = widths[np.isfinite(widths)]
    return float(np.median(widths)) if len(widths) else float("nan")


def _spread(values) -> float:
    """the width of the target's own 10-90% range - what knowing nothing already tells you"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.subtract(*np.quantile(values, [0.9, 0.1]))) if len(values) else float("nan")


def _stack(parts: list[dict]) -> dict:
    keys = set().union(*(p.keys() for p in parts)) if parts else set()
    return {k: np.concatenate([p[k] for p in parts if k in p]) for k in keys}
