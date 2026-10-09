# multi-family spikes: s0

measured 2026-10-09 on macos 26.6 arm64, python 3.12.13. production code unchanged. workspace code and feature constants are from commit 696e6e2. these are single-series mechanics probes, not a tuned family search or a test-set verdict. elapsed seconds include early stopping and refitting for trees, and the entire cross-validation call for stats; imports, workspace prep, output mapping and standalone checks are outside those fit timers. jobs ran concurrently, so timings include contention.

read-only input: `/Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv`. sha256: `a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be`. scripts use production `prepare()` and `load_workspace()` with time `dteday`, target `cnt`, aggregation `sum`, mode `series`, task `regression`, no dimensions. temporary parquet files are removed on context exit. raw csv and generated outputs are not committed.

daily horizon 30: 731 panel steps; 641 raw training steps; 60 validation steps, 2012-10-03 through 2012-12-01; 30 untouched test steps. buckets are `(1,1)`, `(2,4)`, `(5,8)`, `(9,13)`, `(14,26)`, `(27,30)`. the last bucket ends at 30, not 52, because `buckets_for(30)` clips it. each daily bucket has exactly the same 60 validation targets. wape is `sum(abs(y-prediction))/sum(abs(y))`.

## a: lightgbm and xgboost

all `frame.features.families` are selected through `pick()`. rows come from `ws.series_masks[bucket]`; training rows are sorted by target step. daily bucket `(1,1)` has 640 training rows, 32 float32 features, 195 missing training cells, and no categorical columns. newest 96 training rows select rounds; the preceding 544 fit the probe. both libraries use seed 0, one thread, at most 2000 rounds, patience 50, and then refit all training rows with the selected rounds.

lightgbm: learning rate 0.05, minimum 20 rows per leaf, deterministic mode, column-wise fitting. xgboost: `max_depth=3`, `eta=0.05`, production `model.fit_model()` and `model.predict()`.

| library | objective | leaves | rounds | fit seconds | validation wape |
|---|---|---:|---:|---:|---:|
| xgboost | absolute | — | 106 | 0.1115 | 0.152661 |
| xgboost | squared | — | 64 | 0.0772 | 0.163283 |
| lightgbm | regression_l1 | 7 | 164 | 0.0915 | 0.156188 |
| lightgbm | regression_l1 | 15 | 101 | 0.1058 | 0.155336 |
| lightgbm | regression_l1 | 31 | 114 | 0.1282 | 0.154712 |
| lightgbm | regression | 7 | 82 | 0.0450 | 0.163342 |
| lightgbm | regression | 15 | 77 | 0.0740 | 0.177731 |
| lightgbm | regression | 31 | 74 | 0.0908 | 0.173204 |

best lightgbm is 0.154712, versus xgboost absolute 0.152661 on identical rows. this six-candidate lightgbm probe does not establish which family wins after tuning.

numeric nan cells need no imputation in either library. categorical handling needs translation: xgboost receives `feature_types=['c', ...]`; lightgbm receives categorical column indices through `categorical_feature`. the repo's `_codes()` emits nonnegative integer codes as float32, with unknown/missing levels represented by nan. a separate 100-row probe with codes 0/1/2/nan and a numeric column produced finite predictions in both libraries: lightgbm mae 1.07e-14, xgboost mae 8.18e-5. this proves the existing representation works; it does not make their split algorithms identical or cover arbitrary pandas object columns.

lightgbm was imported before xgboost, then xgboost and lightgbm fits ran in the same macos process, followed by categorical fits. both daily and weekly processes completed without crash or hang. only this import order, these versions and one-thread fits were exercised. keep the approved separate-family worker design.

## b: rolling statistical forecasts

for bucket `(lo,b)` and validation target `d`, the expected cutoff is `d-b` steps, inclusive. each call ends its input panel at the final validation target, excluding the test split. use `cross_validation(h=b, step_size=1, n_windows=validation_step_count, refit=mode)`. select output rows by all three keys: `unique_id == frame.series`, `ds == frame.step`, `cutoff == frame.origin`. the one-to-one join refuses missing predictions or duplicate matched rows. never score every output horizon: the call emits `n_windows*b` rows, but only its horizon-b rows match the tree rows. daily totals are 360 selected rows across six buckets, versus 4920 raw cross-validation rows per model.

manual seasonal naive repeats the last observed season: prediction for target `d` is `y[d-ceil(b/s)*s]`, where `s` is season length. the source step is at or before `d-b`. this differs from the current production `_series_baseline`, which falls back to the origin value when the period is shorter than the bucket. u5 must preserve that existing baseline's semantics for the existing `snaive` family or explicitly document a changed baseline; the tables below label the separate hand-computed seasonal naive.

### daily, season length 7

all stats scores below use the full 60-row validation mask. xgboost columns use the same masks and features from spike a; both objectives are shown so no per-bucket choice is hidden.

| bucket | seasonal naive | xgboost absolute | xgboost squared | ets true / false | theta true / false | arima true / false |
|---|---:|---:|---:|---:|---:|---:|
| 1–1 | 0.260404 | 0.152661 | 0.163283 | 0.164684 / 0.164286 | 0.164381 / 0.160629 | 0.154555 / 0.152355 |
| 2–4 | 0.260404 | 0.221176 | 0.207448 | 0.186533 / 0.186531 | 0.188276 / 0.181308 | 0.181589 / 0.184334 |
| 5–8 | 0.254855 | 0.219845 | 0.200255 | 0.208033 / 0.209798 | 0.204314 / 0.199030 | 0.198834 / 0.202745 |
| 9–13 | 0.254855 | 0.208984 | 0.215780 | 0.223314 / 0.225617 | 0.217993 / 0.215898 | 0.211136 / 0.216798 |
| 14–26 | 0.291382 | 0.249806 | 0.249312 | 0.296712 / 0.298767 | 0.255435 / 0.257883 | 0.268806 / 0.276724 |
| 27–30 | 0.326982 | 0.236295 | 0.245857 | 0.329753 / 0.334137 | 0.291129 / 0.287865 | 0.301094 / 0.302441 |

| model | bucket | true seconds | false seconds | speedup | max absolute forecast difference |
|---|---|---:|---:|---:|---:|
| ets | 1–1 | 13.050 | 0.277 | 47.2x | 38.469 |
| ets | 2–4 | 18.326 | 0.478 | 38.3x | 158.602 |
| ets | 5–8 | 23.575 | 0.371 | 63.6x | 232.530 |
| ets | 9–13 | 25.487 | 0.377 | 67.6x | 323.127 |
| ets | 14–26 | 24.667 | 0.437 | 56.4x | 562.955 |
| ets | 27–30 | 24.262 | 0.581 | 41.8x | 631.870 |
| theta | 1–1 | 12.693 | 0.273 | 46.5x | 765.684 |
| theta | 2–4 | 15.890 | 0.315 | 50.5x | 792.727 |
| theta | 5–8 | 17.879 | 0.370 | 48.4x | 793.098 |
| theta | 9–13 | 19.553 | 0.339 | 57.8x | 801.098 |
| theta | 14–26 | 17.385 | 0.406 | 42.8x | 766.749 |
| theta | 27–30 | 18.789 | 0.353 | 53.2x | 832.288 |
| arima | 1–1 | 19.367 | 0.375 | 51.7x | 523.266 |
| arima | 2–4 | 20.796 | 0.473 | 44.0x | 499.791 |
| arima | 5–8 | 17.765 | 0.271 | 65.6x | 618.708 |
| arima | 9–13 | 14.308 | 0.258 | 55.5x | 625.929 |
| arima | 14–26 | 16.082 | 0.802 | 20.1x | 733.957 |
| arima | 27–30 | 33.921 | 0.877 | 38.7x | 297.668 |

`refit=False` does not equal `refit=True`. for every daily model/bucket, refit-true forecasts at the first and last origins exactly matched standalone forecasts fitted on data through those origins: maximum absolute difference 0. the equality check also passed for successful weekly buckets.

refit-false still updates with observed history. adding 1000 to the target at 2012-11-01 left forecasts from earlier origins unchanged (maximum difference 0) and changed later forecasts by up to 183.546 for ets, 173.218 for theta, and 370.743 for arima. installed statsforecast `core.py` cross-validation code fits the first window, then calls each model's `forward(y=y_train, h=h, ...)` with the current origin's history. parameters/model selection are reused; the observations are not frozen at the initial cutoff.

arima is bounded: `max_p=2`, `max_q=2`, `max_P=1`, `max_Q=1`, `max_order=4`, `nmodels=20`, `approximation=True`; remaining options are the installed defaults. these numbers are not the cost of unrestricted autoarima. convergence warnings are captured with counts and up to three distinct messages in json, rather than printed as large logs. successful predictions do not prove optimizer convergence.

### weekly aggregation

production prep aggregates to 106 monday steps, including partial first and last weeks. horizon 30 leaves 26 raw train steps, 50 validation steps, and 30 test steps. buckets through 26 have 50 validation rows; bucket 30 loses the earliest four and has 46. initial stats histories have 26, 23, 19, 14, 1 and 1 observations. a complete same-row 52-week seasonal-naive score is unavailable because the first validation rows have no observation 52 weeks earlier. no missing rows were dropped to manufacture a score.

| model | bucket | true wape / error | false wape / error | true / false seconds | speedup | max absolute difference |
|---|---|---|---|---|---:|---:|
| ets | 1–1 | 0.114864 | 0.111252 | 2.672 / 0.026 | 101.5x | 5587.996 |
| ets | 2–4 | 0.180060 | 0.152603 | 1.973 / 0.013 | 154.6x | 8140.933 |
| ets | 5–8 | 0.268012 | 0.213819 | 1.397 / 0.012 | 112.8x | 13532.060 |
| ets | 9–13 | 0.366230 | 0.576159 | 0.901 / 0.014 | 63.2x | 59253.973 |
| ets | 14–26 | IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed | KeyError: 'fitted' | 0.002 / 0.001 | unavailable | unavailable |
| ets | 27–30 | IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed | KeyError: 'fitted' | 0.001 / 0.001 | unavailable | unavailable |
| theta | 1–1 | 0.110059 | 0.114013 | 0.557 / 0.014 | 40.5x | 3916.430 |
| theta | 2–4 | 0.180580 | 0.178487 | 0.447 / 0.011 | 39.3x | 3863.219 |
| theta | 5–8 | 0.304169 | 0.312804 | 0.409 / 0.013 | 31.7x | 4416.444 |
| theta | 9–13 | 0.487160 | 0.537197 | 0.345 / 0.012 | 27.8x | 15460.920 |
| theta | 14–26 | NotImplementedError: tiny datasets | NotImplementedError: tiny datasets | 0.001 / 0.001 | unavailable | unavailable |
| theta | 27–30 | NotImplementedError: tiny datasets | NotImplementedError: tiny datasets | 0.001 / 0.001 | unavailable | unavailable |
| arima | 1–1 | 0.116495 | 0.366437 | 4.989 / 0.176 | 28.4x | 20382.732 |
| arima | 2–4 | 0.195336 | 0.918800 | 4.541 / 0.170 | 26.7x | 50805.358 |
| arima | 5–8 | 0.327064 | 0.433574 | 5.489 / 0.234 | 23.5x | 26528.638 |
| arima | 9–13 | 0.524547 | 0.385490 | 5.164 / 0.183 | 28.3x | 26713.343 |
| arima | 14–26 | 0.781695 | 0.942150 | 6.447 / 0.045 | 142.9x | 68712.354 |
| arima | 27–30 | 0.846526 | 0.941829 | 9.442 / 0.030 | 311.7x | 74598.599 |

weekly horizon-30 xgboost: bucket 1 absolute/squared wape 0.177341/0.182976; bucket 4 0.227972/0.223101. train rows are 25/22, so production skips early stopping and uses 2000 rounds. buckets 8/13/26/30 have 18/13/0/0 train rows and fail the production minimum of 20. lightgbm bucket 1 with the same minimum-leaf setting makes a constant fit: absolute 0.432309, squared 0.365477. this split cannot support the intended whole-family weekly comparison.

supplemental horizon 8 leaves 67 raw train steps, 26 validation steps (2012-04-09 through 2012-10-01), and 13 test steps. initial stats histories are 67/64/60; all rows have a 52-week seasonal-naive source. this is a separate split, never compared with the daily or weekly-horizon-30 validation scores.

| bucket | seasonal naive | xgboost absolute / squared | ets true / false | theta true / false | arima true / false |
|---|---:|---:|---:|---:|---:|
| 1–1 | 0.342045 | 0.443813 / 0.256808 | 0.062197 / 0.067137 | 0.059343 / 0.059764 | 0.061048 / 0.060139 |
| 2–4 | 0.342045 | 0.430938 / 0.437793 | 0.069942 / 0.105806 | 0.061915 / 0.061076 | 0.073296 / 0.066601 |
| 5–8 | 0.342045 | 0.404254 / 0.360412 | 0.119945 / 0.117657 | 0.093281 / 0.106002 | 0.112788 / 0.114249 |

| model | bucket | true seconds | false seconds | speedup | max absolute forecast difference |
|---|---|---:|---:|---:|---:|
| ets | 1–1 | 4.098 | 0.121 | 33.8x | 3650.808 |
| ets | 2–4 | 3.298 | 0.109 | 30.4x | 11291.730 |
| ets | 5–8 | 2.681 | 0.098 | 27.2x | 15037.441 |
| theta | 1–1 | 0.657 | 0.029 | 22.6x | 278.086 |
| theta | 2–4 | 0.578 | 0.024 | 23.8x | 359.260 |
| theta | 5–8 | 0.539 | 0.018 | 29.7x | 2306.712 |
| arima | 1–1 | 2.339 | 0.091 | 25.7x | 836.895 |
| arima | 2–4 | 2.040 | 0.083 | 24.6x | 2833.408 |
| arima | 5–8 | 1.976 | 0.101 | 19.5x | 5614.685 |

all three models accept season length 52 in the supplemental run, but fewer than two full seasonal cycles precede each first origin. inspecting the first ets fit gives `method='ETS(A,Ad,N)'`, `m=1`: it selected a nonseasonal model despite the requested 52. this establishes api feasibility, not reliable estimation of yearly seasonality.

### api and season source for u5

input columns: `unique_id` (repo series key), `ds` (datetime64 ns), `y` (raw numeric target); frequency `D` or `W-MON`. output columns: `unique_id`, `ds`, `cutoff`, `y`, and `AutoETS`, `AutoTheta` or `AutoARIMA`. bucket identity comes from the call, not an output column. assign it after the exact join and keep frame row order. do not assume rows are emitted in pipeline order or that the last row alone identifies the desired horizon.

the production source for season is `ws.profile['genes']['period']` (also `ws.profile['period']`), learned on training only. measured daily period is 6; weekly period is null, both horizon 30 and 8. these tables intentionally force 7 and 52 as requested. using the unchanged profile would therefore produce different forecasts. u5 should consume a positive profile period, falling back to 1 when absent, and persist the effective season length with the recipe. f1 must resolve the profile's period detection before describing daily 7 or weekly 52 as automatic defaults. no profile or feature code was changed in s0.

## c: wheels, size and imports

binary-only `uv pip install --dry-run --only-binary :all:` succeeded for all six combinations. this checks the full resolved dependency closure, including native coreforecast, scipy, statsmodels and pyarrow, without allowing source builds.

| platform | python | result | resolved packages | versions |
|---|---|---|---:|---|
| x86_64-pc-windows-msvc | 3.11 | passed | 30 | statsforecast 2.1.1; lightgbm 4.7.0; numba 0.68.0; llvmlite 0.50.0 |
| x86_64-pc-windows-msvc | 3.12 | passed | 30 | statsforecast 2.1.1; lightgbm 4.7.0; numba 0.68.0; llvmlite 0.50.0 |
| x86_64-pc-windows-msvc | 3.13 | passed | 30 | statsforecast 2.1.1; lightgbm 4.7.0; numba 0.68.0; llvmlite 0.50.0 |
| aarch64-apple-darwin | 3.11 | passed | 29 | statsforecast 2.1.1; lightgbm 4.7.0; numba 0.68.0; llvmlite 0.50.0 |
| aarch64-apple-darwin | 3.12 | passed | 29 | statsforecast 2.1.1; lightgbm 4.7.0; numba 0.68.0; llvmlite 0.50.0 |
| aarch64-apple-darwin | 3.13 | passed | 29 | statsforecast 2.1.1; lightgbm 4.7.0; numba 0.68.0; llvmlite 0.50.0 |

python 3.11 resolves numpy 2.4.6 and scipy 1.17.1; 3.12/3.13 resolve numpy 2.5.3 and scipy 1.18.1. windows adds colorama. these are resolution results at measurement time, not windows runtime tests. no windows host was available.

statsforecast 2.1.1 declares no numba/llvmlite dependency; its installed requirements are cloudpickle, coreforecast, fugue, numpy, pandas, scipy, statsmodels, threadpoolctl, tqdm and utilsforecast. the approved plan's numba compilation risk does not apply to this measured version; native wheel/runtime risk remains.

`du -sk` of the throwaway venv before adding test tools: 385284 kib (376.254 mib) for numpy, duckdb, xgboost, lightgbm, statsforecast, pandas and dependencies. explicitly adding numba/llvmlite gives 552740 kib (539.785 mib), an extra 167456 kib (163.531 mib). this is combined installed disk use, not the download size or incremental size of each forecasting extra. ruff/pytest/ui-base/torch were added afterward solely for repo validation; those are excluded from these sizes.

fresh-process import times, one run per library; os caches were not flushed. python process startup is excluded. this is cold python-module state, not cold disk.

| library | version | import seconds |
|---|---|---:|
| numpy | 2.5.3 | 0.129781 |
| duckdb | 1.5.6 | 1.261158 |
| xgboost | 3.4.1 | 0.700199 |
| lightgbm | 4.7.0 | 1.136129 |
| statsforecast | 2.1.1 | 0.515751 |
| pandas | 2.3.3 | 0.220112 |
| numba | 0.68.0 | 0.097031 |
| llvmlite | 0.50.0 | 0.000170 |

llvmlite's top-level import is only a namespace import; it does not load llvm. imports for libraries with shared dependencies are measured in separate processes, so these values overlap and must not be summed as a worker startup estimate.

## decisions and cost estimates

u3: keep numpy float32 and native nan handling. translate categorical type markers to lightgbm column indices; do not pass xgboost's type strings through unchanged. newest 15 percent of training steps choose rounds, then refit all train rows; use the same minimum head/tail guard as `fit_model()` for thin data. sort by step before the split; with multiple series, hold out all rows of the newest steps together. separate worker processes remain the design.

u5: use rolling-origin cross-validation with `h=b`, stride 1, exact origin joins and `refit=False` as the fixed-parameter, observed-history protocol. it works and is much cheaper; do not claim it reproduces refit-true forecasts. use this same mode for scoring and subsequent phases. retain true-mode measurements as the documented accuracy/cost comparison, without adding an unrequested ui control. check earliest origin history before fitting and expose missing-history/model failures; never rank silently dropped rows alongside complete candidates. use the train-only profile period and persist the effective value. cap autoarima search as above for the initial cost hint.

models-page cost hints, measured elapsed seconds for one candidate on one bike series:

| model/protocol | daily six-bucket total | weekly horizon-8 three-bucket total |
|---|---:|---:|
| ets, refit=true | 129.368 | 10.077 |
| ets, refit=false | 2.521 | 0.328 |
| theta, refit=true | 102.189 | 1.774 |
| theta, refit=false | 2.055 | 0.072 |
| arima, refit=true | 122.239 | 6.356 |
| arima, refit=false | 3.055 | 0.275 |
| xgboost absolute | 0.763 | see per-bucket json |
| xgboost squared | 0.473 | see per-bucket json |

lightgbm measured only bucket 1: 0.045–0.128 seconds per candidate on daily data. a six-bucket estimate formed by multiplying that range by six is 0.270–0.769 seconds; it is an extrapolation, not a measured six-bucket run. start estimates from measured bucket/window/series counts, multiply by requested candidates, and show units and assumptions. scaling to more series or longer history needs calibration, especially for arima. parallel-family wall time is the largest family estimate plus startup/coordination, subject to cpu contention. parameter grids can change rounds and fit cost; these timings cannot guarantee a time cap.

changes to the approved plan, recorded here without editing `.reference/plan.md`:

- retain `refit=False`, now with proof that history rolls and an explicit warning that parameters/forecasts differ from refit-true
- name the exact three-key mapping and inclusive origin; score only matched horizon-b rows
- replace the numba dependency/compilation assumption for statsforecast 2.1.1 with its actual native dependency closure
- make weekly history sufficiency a u5 acceptance case; horizon 30 on this two-year input cannot yield a full seasonal-52 comparison
- f1 must reconcile detected period 6/null with the intended 7/52 scenarios; unchanged-profile u5 results will differ from these forced-season results
- do not prescribe which family must win weekly; this rolling horizon-8 split gives stats much lower errors than the manual seasonal naive
- preserve the existing production seasonal-naive semantics unless a separate approved change replaces them
- windows runtime safety is still unproved; wheel availability alone does not close that risk

updated unit contracts:

```text
unit: u3 lightgbm
expects: bucket frame, all selected float32 feature columns, train/val masks, type markers
does: translates category indices, selects rounds on newest training steps, refits train
outputs: finite predictions in frame order, rounds, fit history
verifies: daily bucket 1 accepts 195 nan cells; 0/1/2/nan category fixture fits; validation matches the same 60-row mask as xgboost

unit: u5 stats and snaive
expects: raw panel, per-bucket masks, positive train-only profile period or fallback 1
does: scores origin d-b with stride 1 and fixed-parameter forward updates, checking history
outputs: predictions joined by series/target/origin and tagged with bucket in frame order
verifies: first/last true-mode forecasts equal standalone fits; perturbing a future observation leaves earlier-origin false-mode forecasts unchanged; insufficient-history candidates fail explicitly; flat-series baseline remains equal
```

## reproduction

run from the specified worktree. each command is independent; no command installs into a repo venv. variables below keep every writable cache outside the repo. dependency versions above identify the measured resolution; unpinned installs may choose newer releases later. stats scripts enforce a 480-second posix timer; use `--bucket` and then `--max-windows` if a slower host needs a smaller probe, and report the changed rows.

```sh
cd /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/worktrees/mf-spikes
export UV_CACHE_DIR=/private/tmp/multifamily-uv-cache
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=/private/tmp/multifamily-numba-cache
export PYTHONPATH=/Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/worktrees/mf-spikes
uv venv /private/tmp/multifamily-spike-venv --python 3.12
uv pip install --python /private/tmp/multifamily-spike-venv/bin/python numpy duckdb xgboost lightgbm statsforecast pandas
du -sk /private/tmp/multifamily-spike-venv
uv pip install --python /private/tmp/multifamily-spike-venv/bin/python numba llvmlite
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/platform_wheels.py
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/platform_wheels.py --imports-only
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/tree_fits.py --unit day --output /private/tmp/multifamily-tree-day.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/tree_fits.py --unit week --output /private/tmp/multifamily-tree-week.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/tree_fits.py --unit week --horizon 8 --output /private/tmp/multifamily-tree-week8.json
```

wheel script runs all six equivalents of this command, replacing platform and python version:

```sh
uv pip install --dry-run --only-binary :all: --python-platform x86_64-pc-windows-msvc --python-version 3.11 --target /private/tmp/multifamily-wheels/target --no-config statsforecast lightgbm numba llvmlite
```

stats commands (one model per process, bounded arima settings are in the script):

```sh
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit day --horizon 30 --model ets --output /private/tmp/multifamily-stats-day-ets.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit day --horizon 30 --model theta --output /private/tmp/multifamily-stats-day-theta.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit day --horizon 30 --model arima --output /private/tmp/multifamily-stats-day-arima.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit week --horizon 30 --model ets --output /private/tmp/multifamily-stats-week-ets.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit week --horizon 30 --model theta --output /private/tmp/multifamily-stats-week-theta.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit week --horizon 30 --model arima --output /private/tmp/multifamily-stats-week-arima.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit week --horizon 8 --model ets --output /private/tmp/multifamily-stats-week8-ets.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit week --horizon 8 --model theta --output /private/tmp/multifamily-stats-week8-theta.json
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit week --horizon 8 --model arima --output /private/tmp/multifamily-stats-week8-arima.json
```

the update perturbation check runs automatically for bucket 1. inspect the first weekly ets fit with:

```sh
/private/tmp/multifamily-spike-venv/bin/python scripts/spikes/stats_rolling.py --source /Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv --unit week --horizon 8 --model ets --inspect-initial --output /private/tmp/multifamily-stats-inspect-week8-ets.json
```

api checks and measurements are written to json outside the repo; the committed tables above retain the findings after large temporary files are removed.

format and validate through the external environment:

```sh
uv pip install --python /private/tmp/multifamily-spike-venv/bin/python ruff pytest pytest-xdist pillow 'ui-base @ git+https://github.com/BalthazarFitzpatrick/smortui.git@cde0f6b14a2ef2f68eacf10b9d76f50e61e4e0ee' torch
export UV_PROJECT_ENVIRONMENT=/private/tmp/multifamily-spike-venv
export UV_NO_SYNC=1
export RUFF_CACHE_DIR=/private/tmp/multifamily-ruff-cache
uv run ruff check . --fix
uv run ruff format .
uv run pytest -q -n auto -p no:cacheprovider --basetemp=/private/tmp/multifamily-pytest-parallel --durations=15
```

validation: `ruff check . --fix` and `ruff format .` passed. full suite with `-n auto`: 660 passed, 4 skipped, 1 xfailed in 307.86 seconds; browser/coreml extras are outside this spike environment. output retained at `/private/tmp/multifamily-pytest-output.txt`. product changes, family modules, dependency manifests, feature changes and ui work are outside s0.
