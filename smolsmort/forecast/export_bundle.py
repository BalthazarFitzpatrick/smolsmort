"""portable run reports and saved prediction bundles"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from smolsmort.forecast import views
from smolsmort.forecast.export_wide import _format_cell, export_wide_csv, normalize_columns
from smolsmort.forecast.features import split_family_name
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.spec import STEP, spec_from_dict
from smolsmort.forecast.tables import read_table
from smolsmort.forecast.tiles import read_ensemble_rows, read_json, read_tiles, table_metrics

INCLUDE_LEVELS = ("details only", "details + predictions", "everything")


def default_folder() -> str:
    return str(Path.home() / "smolsmort-exports")


def _read_members(run_dir, selected, models, saved_recipe, result):
    finalists = read_json(run_dir / "finalists.json", [])
    if selected.get("kind") == "ensemble":
        ids = selected.get("members", [])
        chosen = [model for model in models if model["tile_id"] in ids]
        if len(chosen) != len(ids):
            raise views.ViewError("ensemble member recipe is missing")
    else:
        chosen = [selected]
    members = []
    for model in chosen:
        family = model["family"]
        finalist = next(
            (item for item in finalists if item.get("key") == model.get("recipe_key")), {}
        )
        genome = finalist.get("genome") or model.get("recipe")
        if not genome:
            genome = saved_recipe.get("genome")
        if not genome:
            raise views.ViewError(f"no saved recipe for {family}")
        target = run_dir / "families" / family
        exit_test = read_json(target / "exit_test.json", {})
        if exit_test.get("recipe_key") != model.get("recipe_key"):
            exit_test = {}
        summary = result.get("families", {}).get(family, {})
        facts = {
            key: summary[key]
            for key in (
                "state",
                "fitness",
                "model_error",
                "baseline_error",
                "val_error",
                "fold_count",
                "fold_scores",
                "horizon_mode",
                "horizon",
                "generation",
                "reason",
                "nthread",
                "holdout",
                "origin_blocks",
            )
            if key in summary
        }
        members.append(
            {
                "family": family,
                "recipe_key": model.get("recipe_key"),
                "genome": genome,
                "fit_facts": facts,
                "exit_test": exit_test,
                **{key: finalist[key] for key in ("seed", "nthread") if key in finalist},
            }
        )
    return members


def build_recipe(run_dir: Path, query: dict, *, made: datetime | None = None) -> dict:
    result = views._require_done(run_dir)
    request = views.read_request(run_dir)
    saved_recipe = read_json(run_dir / "recipe.json", {})
    selection = views._select_model(run_dir, result, query, read_only=True)
    models = read_tiles(run_dir, include_ensembles=False, read_only=True)
    ensembles = read_ensemble_rows(run_dir, read_only=True)
    model_id = selection["model"]
    selected = next((item for item in models + ensembles if item["tile_id"] == model_id), None)
    if selected is None:
        if model_id == "winner":
            selected = next(
                (item for item in models if item["family"] == selection["family"]), None
            )
        else:
            names = set(selection["members"])
            selected = {
                "tile_id": "ensemble",
                "kind": "ensemble",
                "family": "ensemble",
                "members": [item["tile_id"] for item in models if item["family"] in names],
                "recipe": {"weight": "equal"},
            }
    if selected is None:
        raise views.ViewError("no saved model recipe")
    if selected["kind"] == "ensemble" and not selected.get("members"):
        raise views.ViewError("ensemble has no saved member recipes")
    metric = query.get("metric") or "wape"
    if metric not in ("wape", "mae", "mape", "ape", "ae", "wae"):
        raise views.ViewError("unknown export metric")
    metrics = selected.get("metrics", {}).get(metric, {})
    baseline = next((item for item in models if item["family"] == "snaive"), {})
    naive = baseline.get("metrics", {}).get(metric, {}).get("test")
    test = metrics.get("test")
    ratio = test / naive if test is not None and naive else None
    if ratio is None and metric == "wape":
        ratio = selected.get("error_metrics", {}).get("test_ratio", {}).get("value")
    if model_id in ("winner", "ensemble"):
        metrics = {}
        for phase in ("validation", "test"):
            path = views._optional_prediction_path(run_dir, {"model": model_id}, phase)
            if path is not None:
                metrics[phase] = table_metrics(read_table(path)).get(metric)
        test = metrics.get("test")
        ratio = test / naive if test is not None and naive else None
        if ratio is None and metric == "wape":
            payload = views.run_metric_payload(run_dir, request, result=result, model=model_id)
            ratio = payload["error_metrics"].get("test_ratio", {}).get("value")
    return {
        "run_id": run_dir.name,
        "made": (made or datetime.now().astimezone()).isoformat(),
        "model": model_id,
        "name": "ensemble of " + ", ".join(selection["members"])
        if selected["kind"] == "ensemble"
        else selected["family"],
        "kind": "ensemble" if selected["kind"] == "ensemble" else "single",
        "selected": {
            key: selected[key]
            for key in (
                "tile_id",
                "kind",
                "family",
                "recipe_key",
                "members",
                "recipe",
                "metrics",
                "error_metrics",
            )
            if key in selected
        },
        "scores": {"metric": metric, **metrics, "ratio": ratio},
        "request": request,
        "members": _read_members(run_dir, selected, models, saved_recipe, result),
        "weighting": selected.get("recipe", {}) if selected["kind"] == "ensemble" else None,
        "holdout": result.get("holdout"),
    }


def _cell(value):
    if isinstance(value, (dict, list, bool)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows):
    return [
        "| field | value |",
        "| --- | --- |",
        *(f"| {key} | {_cell(value)} |" for key, value in rows if value is not None),
        "",
    ]


def _feature_groups(features):
    groups = {}
    for feature in features:
        operation = split_family_name(feature)[2]
        kind = operation.partition(":")[0].partition("+")[0]
        if kind == "lag":
            kind = "lags"
        elif kind in ("roll", "ewm"):
            kind = "rolling"
        elif kind in ("cal", "date"):
            kind = "calendar"
        elif kind == "lead":
            kind = "leads"
        elif kind in ("measure", "log"):
            kind = "measures"
        elif kind in ("dim", "freq"):
            kind = "dimensions"
        else:
            kind = "other"
        groups.setdefault(kind, []).append(feature)
    return groups.items()


def _feature_kind(feature):
    operation = split_family_name(feature)[2]
    if operation.startswith(("cal:", "date:")):
        return operation.replace("date:", "cal:", 1)
    return "lag+" if operation.startswith("lag+") else operation.partition(":")[0]


def _feature_definitions(recipe):
    # features.py: _timeseries_base, stacked_timeseries_features, timeseries_features, _shift
    definitions = {
        "lag": (
            "target@clock:lag:k",
            "single mode: y_(o-k); bucket mode: y_(t-k), present only if k >= b. unavailable history is missing; zeros remain zero",
        ),
        "lag+": (
            "target@clock:lag+j",
            "y_(o-j), including y_o for lag+0 and y_(o-1) for lag+1. unavailable history is missing; zeros remain zero",
        ),
        "diff": (
            "target@clock:diff:k",
            "y_o - y_(o-k); both values must be finite, otherwise missing. zeros are ordinary values",
        ),
        "season": (
            "target@clock:season",
            "single mode: y_(t-p), mean(y_(t-p-1), y_(t-p), y_(t-p+1)), plus y_(t-P) if a period P was detected; p = 52 weeks (365 days, 12 months, 4 quarters, 1 year). each read must be <= o; any missing neighbor makes the mean missing. bucket mode: y_(t-P), only if P >= b; unavailable history is missing; zeros remain zero",
        ),
        "measure": (
            "source@clock:measure",
            "time series: raw source x_(o-j) for j = 0, 1, 2, 3; entity mode: raw input x. unavailable history is missing; zeros remain zero",
        ),
        "lead": (
            "source@clock:lead",
            "with detected source lag k, single mode reads x_(o-k); bucket mode reads x_(t-k) only if k >= b. unavailable history is missing; zeros remain zero",
        ),
        "weeks_ahead": (
            "weeks_ahead",
            "h = t - o, known without target reads; always added to single-mode designs (ridge also gets one indicator per lead). no missing history or zero handling needed",
        ),
    }
    # features.py: _rolling, _rolling_mean, _slope, _ewm, _timeseries_base
    definitions.update(
        {
            "roll": (
                "target@clock:roll:k:stat",
                "over finite y in [o-k+1, o], n >= max(floor(k/2), 1) is required, else missing. mean = sum(y)/n; std = sqrt(max(mean(y*y)-mean(y)^2, 0)); min/max are extrema; median/q10/q90 are linearly interpolated quantiles at 0.5/0.1/0.9; zeros = count(y = 0)/n. zeros count as observations; legacy roll:k selects mean/std/max/zeros",
            ),
            "slope": (
                "target@clock:slope:k",
                "over [o-k+1, o], x is the original position 0..k-1 of each finite y: (n*sum(x*y)-sum(x)*sum(y))/(n*sum(x*x)-sum(x)^2). require n >= max(floor(k/2), 2) and positive denominator, else missing; gaps retain their positions, zeros remain observations",
            ),
            "momentum": (
                "target@clock:momentum:k",
                "m = floor(k/2); mean(y in [o-m+1, o]) - mean(y in [o-2*m+1, o-m]). each mean needs max(floor(m/2), 1) finite values, else missing; zeros count as observations",
            ),
            "ewm": (
                "target@clock:ewm:a",
                "first finite y initializes e; each later finite y_s sets e_s = a*y_s + (1-a)*e_previous, returned at o. missing values retain the previous level; before the first finite value the result is missing; zeros update the level",
            ),
            "cycle_ratio": (
                "target@clock:cycle_ratio",
                "y_o / y_(o-p), for detected or fallback cycle p; missing/nonfinite numerator or denominator, or zero denominator, gives missing. zero numerator with nonzero denominator gives zero",
            ),
        }
    )
    # features.py: _calendar, _timeseries_calendar, _codes, entity_features
    definitions.update(
        {
            "cal:week": (
                "clock:cal:week",
                "floor(day_of_year_zero_based/7)+1 at t; this is not iso week numbering",
            ),
            "cal:month": ("clock:cal:month", "calendar month 1..12 at t"),
            "cal:quarter": ("clock:cal:quarter", "floor((month-1)/3)+1 at t"),
            "cal:weekday": ("clock:cal:weekday", "weekday at t, monday = 0 through sunday = 6"),
            "cal:cyclic": (
                "clock:cal:cyclic",
                "sin(2*pi*c), cos(2*pi*c): c = (month-1)/12 and day_of_year_zero_based/days_in_year; daily data also uses weekday/7. weekly dates are aligned to monday",
            ),
            "cal:holiday": (
                "clock:cal:holiday",
                "holiday indicator at t, min(days until next holiday, 30), min(days since previous holiday, 30), from the configured country's calendar. weekly data uses monday only",
            ),
            "cal:dom": (
                "clock:cal:dom",
                "day of month d, indicators for first/last day of month, and floor((d-1)/7)+1, at t",
            ),
            "cal:trend": (
                "clock:cal:trend",
                "entity anchor days since earliest training anchor divided by 365.25; if training is empty, earliest anchor is used",
            ),
            "dim": (
                "source:dim",
                "sorted nonempty training categories map to integer codes 0..n-1; null, nan, empty or unseen categories are missing. series dimensions are constant; no target history or target transform is used; a literal category 0 is a category",
            ),
            "freq": (
                "source:freq",
                "training count of category / max(training row count, 1); missing/unseen categories give zero. training levels only; raw categories, no target transform",
            ),
            "log": (
                "source:log",
                "entity input log(1+x), offered only when finite training inputs exist and their minimum is nonnegative. missing x remains missing and zero maps to zero; no future target reads or target transform",
            ),
        }
    )
    # features.py: _build_input_values, timeseries_features, stacked_timeseries_features
    definitions.update(
        {
            "mix": (
                "source@clock:mix",
                "at o-j, level share = raw target total for level / raw target total across levels; top five levels by training target total get separate shares, others are summed. observed zero total gives zero shares; no source rows gives missing; skip_missing excludes (none) when configured",
            ),
            "spread": (
                "source@clock:spread",
                "at o-j, count = number of retained level rows; hhi = sum(level_share^2), with mix's raw target shares. observed zero total gives hhi = 0; no source rows gives missing; skip_missing excludes (none) when configured",
            ),
        }
    )
    kinds = sorted(
        {
            _feature_kind(feature)
            for member in recipe["members"]
            for feature in member["genome"].get("families", [])
        }
    )
    # features.py: _extend, _fill_value, fill_shutdown_weeks; pipeline.py: _timeseries_fit
    lines = [
        "## feature definitions",
        "",
        "y_t is the raw series target at week t; x_t is a raw source measure; o is the feature origin; h = t-o; k is a window or lag; b is the bucket's upper lead. other time units use steps instead of weeks.",
        "id patterns use source and clock column names; legacy ids omit those prefixes. relative lag j = 0, 1, 2, 3 for mix/spread as well as measure features.",
        "single mode uses o = t-h; bucket mode uses o = t-b, which can precede the displayed forecast origin. history features read only weeks <= o; calendar features may read the known date t. all history features use raw values before the member's target transform, including log1p.",
        "short histories are not padded before series start. after series end, sum/count extend with zero, last carries the last finite value, other aggregations stay missing. if history_fill is configured, m is the median of finite, unflagged targets in the prior median_weeks; with at least four such values, m >= min_level and y_s < below*m, replace y_s with m for features only; labels remain unchanged.",
        "calendar features use dates only, no transformed target, history minimum or zero exclusion. missing numeric feature values stay missing here; the fitted estimator handles them.",
        "",
        "| kind | id pattern | definition |",
        "| --- | --- | --- |",
    ]
    for kind in kinds:
        pattern, definition = definitions.get(
            kind, (kind, "see smolsmort/forecast/features.py: family_aliases")
        )
        lines.append(f"| {kind} | `{pattern}` | {definition} |")
    lines += [""]
    # transform.py: TargetTransform.fwd, TargetTransform.inv, TargetTransform._log_base
    transforms = {
        "none": "z_t = y_t; inverse yhat_t = zhat_t",
        "log1p": "z_t = log(1+y_t); inverse yhat_t = exp(zhat_t)-1",
        "logdiff": "z_t = log(1+y_t)-log(1+y_o); inverse yhat_t = exp(zhat_t+log(1+y_o))-1; origin target must be finite",
    }
    used = sorted({member["genome"].get("transform", "none") for member in recipe["members"]})
    lines += [
        "target transform: "
        + "; ".join(
            f"{name}: {transforms.get(name, 'see smolsmort/forecast/transform.py: get_transform')}"
            for name in used
        )
        + ". log transforms reject negative targets; zero maps to log(1) = 0 before any origin subtraction. target labels may be missing; nonfinite inverse log predictions are rejected.",
        "",
    ]
    # evaluate.py: fit_band, apply_band, fit_lead_bands, apply_lead_bands
    # pipeline.py: _finish_timeseries
    lines += [
        "bands: fit raw validation pairs separately per horizon in single mode; bucket mode shares calibration within each bucket. fewer than four finite pairs gives missing bounds. if min(y) >= 0 and median(y) > 0, f = max(0.1*median(abs(y)), 0.000001), s(p) = max(abs(p), f); otherwise s(p) = 1. residual r = (y-p)/s(p).",
        "for coverage c, at least 20 pairs use q_low = quantile(r, (1-c)/2), q_high = quantile(r, 1-(1-c)/2); 4..19 pairs use q_high = quantile(abs(r), c), q_low = -q_high. lower = p+q_low*s(p), upper = p+q_high*s(p); nonnegative mode clips lower to max(lower, 0). a series/horizon with at least 20 finite pairs overrides the pooled horizon band; otherwise it uses the pooled band. zeros are included and protected by the scale floor.",
        "lower/upper denote c = 0.8; lower_50/upper_50 denote c = 0.5. exported csv renames lower/upper to lower_80/upper_80. calibration uses validation residuals only; test/future actuals do not fit bands.",
        "",
    ]
    # evaluate.py: mean_predictions; pipeline.py: _timeseries_fit, _timeseries_predict
    lines += [
        "ensemble combination: p = sum(available member raw predictions)/count(available members), after each member's inverse target transform; each available member has equal weight. nan members are excluded per row, all missing gives missing; zeros participate. only predictions for the same saved row are combined.",
        "",
    ]
    # export_wide.py: _sum_totals, export_wide_csv; evaluate.py: wape
    lines += [
        "wide.csv totals: within each phase, total_forecast = sum(series forecasts) and total_actual = sum(series actuals) for the same target week, forecast_made_week and horizon. zeros participate; any missing summand makes that total missing. totals repeat on every series row; do not sum the repeated totals. total bounds are recalibrated by horizon from validation total residuals using the band rule above, with lower clipped at zero; they are not sums of series bounds.",
        "",
        "wape = sum(abs(y-p))/sum(abs(y)) over finite raw actual/prediction pairs. zeros contribute error but no denominator; zero denominator gives missing. no target transform is used for scoring.",
        "",
    ]
    # views.py: _horizon_metrics, _horizon_baseline; tiles.py: score_ensembles
    # export_bundle.py: build_recipe; evaluate.py: judge
    lines += [
        "results-tab ratio = sum(abs(y-p))/sum(abs(y-naive)) on the selected test rows and horizon selection, after filters/blending. missing actual/prediction pairs are excluded; any missing naive prediction on those rows or zero naive error gives missing. naive uses saved predictions, otherwise the visible seasonal target (52 weeks for weekly data), falling back to y_o. ratio < 1 beats naive; 1 ties. the bundle scores ratio compares the selected test metric with snaive's test metric, falling back to the saved wape verdict ratio when needed; it does not apply results-tab filters.",
        "",
    ]
    return lines


def render_readme(recipe: dict) -> str:
    request = recipe["request"]
    spec = request["spec"]
    columns = spec.get("columns", [])
    lines = ["# run export", ""]
    if recipe.get("files"):
        lines += ["## files", "", *(f"- `{name}`" for name in recipe["files"]), ""]
        lines += [
            "csv date and grouping columns use the spec's own names; all other columns are numeric measures.",
            "input history names the target `actual`. output tables use `forecast`, `actual` where known,",
            "`forecast_made_week` and integer `horizon` where saved. pooled lead errors have no date or groups.",
            "phase and other labels are omitted; bundle results have a clean csv header.",
            "",
        ]
    lines += _table((key, recipe[key]) for key in ("run_id", "made", "name", "kind"))
    lines += ["## scores", ""] + _table(recipe["scores"].items())
    data = [("dataset", Path(spec["source"]).name)] if spec.get("source") else []
    for label, role in (
        ("date column", "time"),
        ("target", "target"),
        ("scaffold dimensions", "dimension"),
    ):
        data.append((label, [column["name"] for column in columns if column["role"] == role]))
    data += [(key, value) for key, value in spec.items() if key != "columns" and key != "source"]
    data += [
        (key, request[key])
        for key in ("transform", "holdout", "horizon_mode", "aggregate", "seed")
        if key in request
    ]
    lines += ["## data and spec", ""] + _table(data)
    if recipe.get("input_data"):
        lines += ["## input history", ""] + _table(recipe["input_data"].items())
        if recipe.get("input_status"):
            lines += [recipe["input_status"], ""]
    if recipe.get("holdout"):
        lines += ["## splits and cut dates", ""] + _table(recipe["holdout"].items())
    lines += _feature_definitions(recipe)
    for member in recipe["members"]:
        genome = member["genome"]
        lines += [f"## member: {member['family']}", ""]
        lines += _table(
            [
                (key, member[key])
                for key in ("family", "recipe_key", "seed", "nthread")
                if key in member
            ]
            + [(key, genome[key]) for key in ("objective", "transform", "seed") if key in genome]
        )
        if "transform" not in genome:
            lines += [
                "target transform is omitted in the saved genome; smolsmort defaults to none.",
                "",
            ]
        if not any("seed" in source for source in (member, genome, request)):
            lines += ["no random seed is recorded.", ""]
        lines += ["### params", ""] + _table(genome.get("params", {}).items())
        lines += ["### features", ""] + _table(_feature_groups(genome.get("families", [])))
        lines += ["### saved fit facts", ""]
        facts = member["fit_facts"]
        lines += _table((key, value) for key, value in facts.items() if key != "origin_blocks")
        blocks = facts.get("origin_blocks", {})
        if blocks:
            lines += [
                "| phase | start | end | train rows | rows | origin count |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for phase, windows in blocks.items():
                for window in windows:
                    lines.append(
                        "| "
                        + " | ".join(
                            _cell(value)
                            for value in (
                                phase,
                                *(
                                    window.get(key, "")
                                    for key in (
                                        "start",
                                        "end",
                                        "train_rows",
                                        "rows",
                                        "origin_count",
                                    )
                                ),
                            )
                        )
                        + " |"
                    )
            lines += [""]
        exit_test = member["exit_test"]
        if exit_test:
            lines += ["### exit test", ""] + _table(
                (key, value) for key, value in exit_test.items() if key != "verdict"
            )
    if recipe["kind"] == "ensemble":
        lines += ["## ensemble", ""] + _table(recipe["weighting"].items())
        lines += [
            "members use equal weights, saved in `ensembles.json`.",
            "",
            "no model weights are stored.",
            "",
        ]
    if recipe.get("missing_tables"):
        lines += ["## tables not saved", "", ", ".join(recipe["missing_tables"]), ""]
    return "\n".join(lines)


def _table_csv(columns):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(
        [_format_cell(value) for value in row] for row in zip(*columns.values(), strict=True)
    )
    return buffer.getvalue().encode("utf-8")


def _input_csv(recipe: dict) -> bytes | None:
    """rebuild the run's original panel without modifying its saved cache"""
    request = recipe["request"]
    if request["spec"].get("mode") != "timeseries":
        recipe["input_data"] = {"mode": "entity"}
        recipe["input_status"] = "input.csv not written: entity run"
        return None
    source = Path(request["spec"]["source"])
    facts = recipe["input_data"] = {
        "dataset": source.name,
        "size_bytes": "unavailable",
        "mtime": "unavailable",
    }
    try:
        stat = source.stat()
        facts.update(
            size_bytes=stat.st_size,
            mtime=datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        )
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        saved_hash = Path(request.get("prepared", "")).name.rsplit("-", 1)[-1]
        facts["saved source hash"] = saved_hash
        facts["current source hash"] = fingerprint
        if saved_hash != fingerprint:
            reason = "dataset changed or saved source fingerprint unavailable"
        else:
            spec = spec_from_dict(request["spec"])
            with tempfile.TemporaryDirectory(prefix="smolsmort-export-") as cache:
                prepared = prepare(spec, Path(cache))
                if (
                    prepared.folder.name.rsplit("-", 1)[-1] != fingerprint
                    or hashlib.sha256(source.read_bytes()).hexdigest()[:12] != fingerprint
                ):
                    reason = "dataset changed during export"
                else:
                    panel = read_table(prepared.folder / "panel.parquet", order_by=STEP)
                    columns = {spec.named("time")[0]: panel[STEP]}
                    columns.update((name, panel[name]) for name in spec.named("dimension"))
                    columns["actual"] = panel[spec.named("target")[0]]
                    return _table_csv(columns)
    except FileNotFoundError:
        reason = "dataset file missing"
    except OSError as exc:
        reason = f"dataset could not be read ({exc.strerror})"
    recipe["input_status"] = f"input.csv not written: {reason}"
    return None


def export_bundle(run_dir: Path, query: dict) -> Path:
    include = query.get("include", "details only")
    if include not in INCLUDE_LEVELS:
        raise views.ViewError("include must be details only, details + predictions or everything")
    value = query.get("folder")
    if not isinstance(value, str) or not value.strip():
        raise views.ViewError("folder must be a nonempty absolute path")
    try:
        folder = Path(value.strip()).expanduser()
    except RuntimeError as exc:
        raise views.ViewError("folder home directory could not be resolved") from exc
    if not folder.is_absolute():
        raise views.ViewError("folder must be an absolute path")
    folder = folder.resolve()
    recipe = build_recipe(run_dir, query)
    model_query = {"model": recipe["model"]}
    files = {}
    if include != "details only":
        history = _input_csv(recipe)
        if history is not None:
            files["input.csv"] = history
        download = views.export_csv(run_dir, recipe["request"], model_query).decode("utf-8")
        if download.startswith("# verdict:"):
            download = download.partition("\n")[2]
        reader = csv.DictReader(io.StringIO(download))
        rows = list(reader)
        columns = {name: [row[name] for row in rows] for name in reader.fieldnames or []}
        files["results.csv"] = _table_csv(normalize_columns(recipe["request"], columns))
        if recipe["request"]["spec"].get("mode") == "timeseries":
            files["wide.csv"] = export_wide_csv(run_dir, recipe["request"], model_query)
    if include == "everything":
        missing = []
        for kind in ("validation", "test", "forecast", "lead_errors"):
            if recipe["model"].startswith("ensemble:"):
                path = run_dir / "tile_views" / recipe["selected"]["recipe_key"] / f"{kind}.parquet"
                path = path if path.exists() else None
            else:
                path = views._optional_prediction_path(run_dir, model_query, kind)
            if path is not None:
                files[f"{kind}.csv"] = _table_csv(
                    normalize_columns(recipe["request"], read_table(path))
                )
            else:
                missing.append(kind)
        recipe["missing_tables"] = missing
    recipe["files"] = [*files, "README.md", "recipe.json"]
    files["README.md"] = render_readme(recipe).encode("utf-8")
    files["recipe.json"] = (json.dumps(recipe, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{run_dir.name}_{recipe['made'][:10]}"
    suffix = 0
    while True:
        target = folder / (name if suffix == 0 else f"{name}_{suffix}")
        try:
            target.mkdir()
            break
        except FileExistsError:
            suffix += 1
    try:
        for filename, data in files.items():
            (target / filename).write_bytes(data)
    except OSError:
        shutil.rmtree(target)
        raise
    return target
