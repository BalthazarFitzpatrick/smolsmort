"""one candidate recipe, scored in phases: `val` fits on train and predicts validation (every
candidate), `test` refits on train + val for the frozen winner only, `final` fits on everything and
predicts the rows nobody knows the answer to yet"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smolsmort.forecast import evaluate as ev
from smolsmort.forecast.features import buckets_for, row_features, series_features
from smolsmort.forecast.model import CLASSIFIERS, NONNEGATIVE, fit_model, predict
from smolsmort.forecast.profile import build_profile
from smolsmort.forecast.spec import ANCHOR, LOWER, ROW, SERIES, STEP, UNIT_DAYS, UPPER, PrepSpec
from smolsmort.forecast.splits import labels_as_of, row_cuts, series_cuts
from smolsmort.forecast.tables import is_text, read_table


class PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class Genome:
    families: tuple[str, ...]
    objective: str
    params: tuple[tuple[str, object], ...]

    def to_dict(self) -> dict:
        return {
            "families": list(self.families),
            "objective": self.objective,
            "params": dict(self.params),
        }

    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def make_genome(families, objective: str, params: dict) -> Genome:
    return Genome(tuple(sorted(set(families))), objective, tuple(sorted(params.items())))


def genome_from_dict(data: dict) -> Genome:
    return make_genome(data["families"], data["objective"], data["params"])


@dataclass
class Scored:
    fitness: float
    contrib: np.ndarray
    metrics: dict


# ---------------------------------------------------------------- workspace


@dataclass
class Workspace:
    spec: PrepSpec
    folder: Path
    summary: dict
    profile: dict
    families: list[str]
    unit: str
    # row mode
    table: dict | None = None
    features: object = None
    masks: dict | None = None
    bounds: dict | None = None
    predict_rows: np.ndarray | None = None
    # series mode
    frames: dict | None = None
    series_masks: dict | None = None
    test_steps: int = 0

    @property
    def mode(self) -> str:
        return self.spec.mode


def load_workspace(spec: PrepSpec, folder: Path, summary: dict) -> Workspace:
    """everything a search needs, read and built once: tables, cuts, profile, candidate features"""
    folder = Path(folder)
    if spec.mode == "row":
        return _row_workspace(spec, folder, summary)
    if spec.task == "classification":
        raise PipelineError(
            "classification runs per row only; the time-series flavour is regression"
        )
    return _series_workspace(spec, folder, summary)


def _row_workspace(spec, folder, summary) -> Workspace:
    table = read_table(folder / "rows.parquet", order_by=f"{ANCHOR}, {ROW}")
    as_of = dt.date.fromisoformat(summary["as_of"])
    censored = spec.censor is not None
    cuts = row_cuts(table[ANCHOR], as_of, maturity_weeks=26 if censored else 0)
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
    features = row_features(spec, table, train)
    predict_ids = read_table(folder / "predict.parquet")[ROW]
    ws = Workspace(spec, folder, summary, profile, [], spec.censor.unit if censored else "week")
    ws.table, ws.features = table, features
    ws.masks = {"train": train, "val": val, "test": test}
    ws.bounds = bounds
    ws.predict_rows = np.isin(table[ROW], predict_ids)
    ws.families = sorted(set(features.families) - _excluded_families(profile))
    return ws


def _series_workspace(spec, folder, summary) -> Workspace:
    panel = read_table(folder / "panel.parquet", order_by=f"{SERIES}, {STEP}")
    scaffold = read_table(folder / "scaffold.parquet")
    steps = np.unique(panel[STEP])
    cuts = series_cuts(len(steps), spec.horizon)
    val_start, test_start = steps[cuts.val_start], steps[cuts.test_start]
    train = panel[STEP] < val_start
    profile = build_profile(spec, panel, train)
    genes = profile["genes"]
    unit = summary["step"]
    frames, masks = {}, {}
    for bucket in buckets_for(spec.horizon):
        frame = series_features(
            spec,
            panel,
            scaffold,
            bucket,
            unit,
            train,
            period=genes.get("period"),
            measure_lags=genes.get("measure_lags"),
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
    ws.test_steps = int(cuts.steps - cuts.test_start)
    families = set().union(*(set(f.features.families) for f in frames.values()))
    ws.families = sorted(families - _excluded_families(profile))
    return ws


def _excluded_families(profile) -> set[str]:
    leaks = set(profile["genes"].get("exclude", []))
    return {f"{kind}:{name}" for name in leaks for kind in ("measure", "log", "lead")}


# ---------------------------------------------------------------- scoring


def score(
    ws: Workspace, genome: Genome, *, nthread: int = 1, sample: np.ndarray | None = None
) -> Scored:
    """validation fitness for one genome; `sample` restricts the training rows (successive halving)"""
    if ws.mode == "row":
        pred, truth, periods = _row_phase(ws, genome, "val", nthread, sample)
    else:
        pred, truth, periods = _series_phase(ws, genome, "val", nthread, sample)
    return _fitness(ws, pred, truth, periods)


def _fitness(ws, pred, truth, periods) -> Scored:
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


def _row_design(ws, genome):
    x, _, types = ws.features.pick(set(genome.families))
    if not x.shape[1]:
        raise PipelineError("a genome with no feature families cannot be fitted")
    return x, types


def _row_fit(ws, genome, rows, phase, nthread):
    x, types = _row_design(ws, genome)
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


def _row_phase(ws, genome, phase, nthread, sample=None):
    """(predictions, truth, period keys) on the phase's evaluation rows"""
    masks = ws.masks
    fit_rows = masks["train"] if phase == "val" else masks["train"] | masks["val"]
    if sample is not None:
        fit_rows = fit_rows & sample
    eval_rows = masks["val"] if phase == "val" else masks["test"]
    target = ws.table[ws.spec.named("target")[0]]
    fitted, x = _row_fit(ws, genome, fit_rows, phase, nthread)
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


def _series_phase(ws, genome, phase, nthread, sample=None):
    preds, truths, periods = [], [], []
    for bucket, frame in ws.frames.items():
        masks = ws.series_masks[bucket]
        fit_rows = masks["train"] if phase == "val" else masks["train"] | masks["val"]
        if sample is not None:
            fit_rows = fit_rows & np.isin(frame.series, sample)
        eval_rows = masks["val"] if phase == "val" else masks["test"]
        fitted, x = _series_fit(ws, genome, frame, fit_rows, nthread)
        preds.append(predict(fitted, x[eval_rows]))
        truths.append(frame.y[eval_rows])
        periods.append(frame.step[eval_rows].astype(int))
    return np.concatenate(preds), np.concatenate(truths), np.concatenate(periods)


def _series_fit(ws, genome, frame, rows, nthread):
    x, _, types = frame.features.pick(set(genome.families))
    if not x.shape[1]:
        raise PipelineError("a genome with no feature families cannot be fitted")
    # oldest first, so the early-stopping slice is the newest steps rather than the last series
    order = np.flatnonzero(rows)[np.argsort(frame.step[rows], kind="stable")]
    fitted = fit_model(
        x[order],
        objective=genome.objective,
        y=frame.y[order],
        feature_types=types,
        params=dict(genome.params),
        nthread=nthread,
    )
    return fitted, x


# ---------------------------------------------------------------- the frozen winner


def finish(ws: Workspace, genome: Genome, *, nthread: int = 1) -> dict:
    """band from validation, the verdict from the untouched test split, and the forecast itself"""
    if ws.mode == "row":
        return _finish_rows(ws, genome, nthread)
    return _finish_series(ws, genome, nthread)


def _finish_rows(ws, genome, nthread) -> dict:
    spec = ws.spec
    classification = spec.task == "classification"
    val_pred, val_truth, _ = _row_phase(ws, genome, "val", nthread)
    test_pred, test_truth, _ = _row_phase(ws, genome, "test", nthread)
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
        )
        band = None
    else:
        band = ev.fit_band(val_truth, val_pred)
        nonnegative = (
            genome.objective in NONNEGATIVE
            or ws.profile["columns"][spec.named("target")[0]].get("min", -1) >= 0
        )
        lo, hi = ev.apply_band(band, test_pred, nonnegative)
        baseline = _row_baseline(ws, target)
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
    fitted, x = _row_fit(ws, genome, everything, "final", nthread)
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
    }


def _class_prior(ws, target) -> np.ndarray:
    train = ws.masks["train"] | ws.masks["val"]
    values = [str(v) for v in target[train] if v is not None and v == v and v != ""]
    classes = sorted(set(values))
    counts = np.array([values.count(c) for c in classes], dtype=float)
    return counts / counts.sum()


def _row_baseline(ws, target) -> np.ndarray:
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


def _finish_series(ws, genome, nthread) -> dict:
    profile = ws.profile
    period = profile["genes"].get("period")
    bands, val_rows, test_rows, forecast = {}, [], [], []
    test_pred_all, test_truth_all, base_all = [], [], []
    test_lo, test_hi = [], []
    for bucket, frame in ws.frames.items():
        masks = ws.series_masks[bucket]
        fitted, x = _series_fit(ws, genome, frame, masks["train"], nthread)
        val_pred = predict(fitted, x[masks["val"]])
        band = ev.fit_band(frame.y[masks["val"]], val_pred)
        bands[f"{bucket[0]}-{bucket[1]}"] = None if band is None else band.__dict__
        val_rows.append(_series_rows(frame, masks["val"], val_pred, bucket))
        fitted, x = _series_fit(ws, genome, frame, masks["train"] | masks["val"], nthread)
        test_pred = predict(fitted, x[masks["test"]])
        lo, hi = ev.apply_band(band, test_pred, nonnegative=True)
        test_pred_all.append(test_pred)
        test_truth_all.append(frame.y[masks["test"]])
        test_lo.append(lo)
        test_hi.append(hi)
        base_all.append(_series_baseline(frame, masks["test"], bucket, period))
        test_rows.append(_series_rows(frame, masks["test"], test_pred, bucket))
        fitted, x = _series_fit(
            ws, genome, frame, masks["train"] | masks["val"] | masks["test"], nthread
        )
        future_pred = predict(fitted, x[masks["future"]])
        lo, hi = ev.apply_band(band, future_pred, nonnegative=True)
        rows = _series_rows(frame, masks["future"], future_pred, bucket)
        rows.update(lower=lo, upper=hi)
        forecast.append(rows)
    truth, pred = np.concatenate(test_truth_all), np.concatenate(test_pred_all)
    first = next(iter(ws.frames))
    seen = ws.series_masks[first]["train"] | ws.series_masks[first]["val"]
    model_error, baseline_error = ev.wape(truth, pred), ev.wape(truth, np.concatenate(base_all))
    verdict = ev.judge(
        model_error=model_error,
        baseline_error=baseline_error,
        band_coverage=ev.coverage(truth, np.concatenate(test_lo), np.concatenate(test_hi)),
        test_rows=len(truth),
        test_steps=ws.test_steps,
        band_width=_median_width(np.concatenate(test_lo), np.concatenate(test_hi)),
        spread=_spread(ws.frames[first].y[seen]),
    )
    return {
        "verdict": verdict,
        "bands": bands,
        "validation": _stack(val_rows),
        "test": _stack(test_rows),
        "forecast": _stack(forecast),
        "model_error": model_error,
        "baseline_error": baseline_error,
    }


def _series_rows(frame, mask, pred, bucket) -> dict:
    return {
        "series": frame.series[mask],
        "step": frame.step[mask],
        "bucket": np.full(int(mask.sum()), f"{bucket[0]}-{bucket[1]}", dtype=object),
        "actual": frame.y[mask],
        "prediction": pred,
    }


def _series_baseline(frame, mask, bucket, period) -> np.ndarray:
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
