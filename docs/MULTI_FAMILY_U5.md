# u5: statistical families and snaive

measured 2026-10-09 on macos, python 3.13.13, statsforecast 2.1.1. target os: windows, mac, linux. local runtime checks cover macos. no windows runtime was available.

raw-series families use `cross_validation(h=b, step_size=1, n_windows=evaluation_step_count, refit=False)` per bucket. input ends at the phase's last evaluated target. predictions join exactly on series, target step and inclusive origin, preserving frame order. missing, duplicate or nonfinite matches fail the candidate. library history errors become named `PipelineError` failures, which the existing search records.

future evaluation extends the input through the last future target with placeholder values. all evaluated origins stay at or before the last observed step; placeholders never enter an origin's training history. snaive calls production `_series_baseline`, including its origin-value fallback for buckets longer than the season. statistical families take no tunable parameters in u5; arima uses the spike's fixed caps.

effective season length comes from the train-only profile, falling back to 1. scores include it in metrics; finished results and saved recipes include `season_length`. the existing xgboost fit implementation, recipe keys and golden leaderboard are unchanged.

## bike daily validation

read-only source: `/Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv`. production prep: `dteday`, `cnt`, sum, daily, horizon 30, no dimensions. 731 panel steps; 60 validation rows per bucket; 360 selected rows per model. test rows were excluded from every validation call.

after merging f1 (`5d2ff9e`), the profile detects period **7**. all values below match the spike's forced-season-7 `refit=False` table to six decimals. these are validation measurements, not test verdicts or tuned-search winners. the earlier base (`649eeb8`) detected 6: theta and arima differed, while ets matched to six decimals.

| bucket | ets | theta | arima |
|---|---:|---:|---:|
| 1–1 | 0.164286 | 0.160629 | 0.152355 |
| 2–4 | 0.186531 | 0.181308 | 0.184334 |
| 5–8 | 0.209798 | 0.199030 | 0.202745 |
| 9–13 | 0.225617 | 0.215898 | 0.216798 |
| 14–26 | 0.298767 | 0.257883 | 0.276724 |
| 27–30 | 0.334137 | 0.287865 | 0.302441 |

arima emitted convergence warnings in every bucket. fitted results retain distinct warning messages. finite predictions do not establish optimizer convergence. raw measurements: `/private/tmp/multifamily-u5-bike.json`.

## verification

`tests/test_forecast_family_stats.py` covers every bucket's validation join against independently indexed statsforecast output; first and last fixed-parameter forward updates; future-observation perturbations; test and future dispatch; missing and duplicate matches; insufficient history; library history exceptions; search continuation with a viable snaive candidate; baseline parity; flat snaive; lazy imports; recipe season persistence; bounded arima.

fit tests use `pytest.importorskip("statsforecast")`. snaive, registry, profile fallback and lazy-import checks run without the stats extra. full-suite outputs: `/private/tmp/multifamily-u5-tests.txt`.

full suites at `5532a34`, including development `5d2ff9e`:

- no optional extras, with ci's torch dev dependency: 547 passed, 34 skipped, 1 xfailed in 163.54 seconds
- repo environment, `uv sync --extra forecast --extra stats`: 728 passed, 4 skipped, 1 xfailed in 363.34 seconds
- initial bare `-e . pytest` install: existing vision tests failed without torch; ci uses `uv sync --dev`, which installs torch; no product or test code changed to bypass that requirement

development `5d2ff9e` was merged into the feature branch at `5532a34`, without conflicts. `uv lock` produced no change. the bike table above was rerun on that merged base.

## limits

the existing `features._unit_days` dates month, quarter and year origins with fixed day counts. statsforecast uses calendar cutoffs, so those units can fail the required exact origin join. u5 preserves that frame contract and reports missing matches instead of scoring shifted origins. day and week origins use exact day counts.

browser and coreml extras are outside these local environments. windows runtime remains untested. the statistical families are raw-target models; target transforms and family-specific search orchestration belong to later approved units.
