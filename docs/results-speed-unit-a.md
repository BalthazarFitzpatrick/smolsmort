# results speed unit a

run: `20261104-140018-573593`, base: `e66732e`, branch: `fix/results-speed`

## root cause and fix

`tiles.read_tiles` unconditionally called `score_ensembles`. `tab._run_tiles`
also called the scoring writer `ensemble_rows` instead of reading `ensembles.json`.
seven families produce 120 subsets, formerly aligned separately for validation
and test: 240 joins per request.

readers now reuse all saved rows while every family validation/test parquet is
older than or equal to the score file. missing, invalid, or stale files are
scored once and replaced atomically. only the validation-ranked top five become
tiles. legacy folders without family parquets save an empty ensemble list.

scoring joins numpy key records once per split and shared-key schema, then
indexes those joins for each subset. subset intersections, first-member row
order, last duplicate values, actual checks, metrics, ids, and ranking stay
intact. different schemas get separate joins rather than silently dropping keys.

worker completion and incremental set updates write the top five prediction
tables with per-series 80% and 50% lead bands. views return saved tables before
resolving members. the existing on-demand fallback remains for older runs.

## timings

seconds, direct function calls without a server, same real run and interpreter
extras. original source loaded before changes for the baseline and golden.

| call | before | after |
| --- | ---: | ---: |
| `read_tiles` | 6.714251 | 0.156697 |
| `/api/forecast-view`, plain | 6.995542 | 0.393465 |
| `/api/forecast-view`, `breakdown=client_group` | 6.744318 | 0.250830 |
| `/api/forecast-view`, `model=ensemble` | 6.658164 | 0.202383 |
| `/api/forecast-run-tiles` | 6.598500 | 0.064916 |

all saved-score paths meet 0.5 s. another after measurement was
0.120394 / 0.367217 / 0.255567 / 0.208949 / 0.065445 s respectively.
these are local function timings, not browser or network latency.

reproduce with `uv run --extra forecast --extra lightgbm --extra stats --extra holidays scripts/measure_results_speed.py <run-folder>`.
optional second argument saves the compact golden; optional third argument loads
the original `tiles.py` source for baseline measurement. golden contains only
120 ids and validation/test wape, mae, mape. real-run golden test skips when the
local run is absent; synthetic join/cache/band tests always run.

## caller regression review

- `read_tiles`: `views._select_tile`, `views.result_details` (therefore `views.view`, tables and exports), `tab._run_tiles`, `runs.run_state`, `tiles.ensemble_rows`; payloads unchanged, `include_ensembles=False` still skips scoring and cache writes
- `score_ensembles`: `tiles.ensemble_rows`; all subsets retained, golden equality within `1e-12`
- `align_tables`: `plan_sets.pick_ensemble`, `views._tile_prediction_path`, new `tiles.write_ensemble_views`; same column arrays, shared-key joins, duplicate and row ordering checked
- `ensemble_rows`: `worker.run`, `plan_sets.update_set`, cache miss in `read_ensemble_rows`; still explicitly computes fresh run-time scores
- `plan_sets.update_set`: `build_refit_run`, worker parent-set updates, plan refresh; same state names, added saved views
- `views._tile_prediction_path`: `run_metric_payload`, `_prediction_path`, `_optional_prediction_path`, recursive member resolution; saved paths bypass construction, legacy fallback retained
- `tab._run_tiles`: `/api/forecast-run-tiles`; same top-five tiles and full ensemble table
- `worker.run`: worker entry point and existing in-process worker tests; completion saves scores before prediction views

## verification

per-file pytest commands, sequential, using:
`uv run --with playwright --with pytest --extra forecast --extra lightgbm --extra stats --extra holidays pytest -q -p no:cacheprovider <file>`.

| file under `tests/` | result |
| --- | --- |
| `test_forecast_run_tiles_data.py` | 19 passed |
| `test_forecast_tiles.py` | 26 passed |
| `test_forecast_plan.py` | 40 passed |
| `test_forecast_single_views.py` | 7 passed |
| `test_forecast_runs_v2.py` | 27 passed |
| `test_forecast_worker_threads.py` | 4 passed |
| `test_forecast_metric_payloads.py` | 8 passed |
| `test_forecast_routes_v2.py` | 99 passed, 2 known base failures |
| `test_forecast_routes.py` | 29 passed, 4 known base failures |

routes v2 failures: `test_v1_route_passes_the_same_budget_and_warm_arguments`,
`test_v2_route_passes_all_coordinator_controls`; expected kwargs omit existing
aggregate/horizon/holdout defaults. routes failures:
`test_view_timeseries_mode_has_a_band_within_bounds`,
`test_view_series_breakdown_gives_one_pair_per_level`,
`test_export_carries_the_ratio_on_every_row` (`KeyError: None` in unchanged
`_horizon_baseline`), `test_entity_mode_view_has_load_line_and_mc_band`
(missing load series). all six match the supplied base-failure list; base
tests were not rerun in another checkout.

focused coverage: saved-score reads reject any scorer call; missing/stale
cache refreshes exactly once; tiles api reuses the same full score list;
legacy missing tables; original real-run golden; subset coverage, shared keys,
duplicate order; two joins for eleven subsets; only five views persisted;
per-series bands; saved paths reject table construction; real worker/refit
completion persists ensemble views. an initial two-row band fixture failed
because the existing fitter needs four residuals; expanded the fixture to
exercise both series overrides without changing that rule.

`uv run ruff check . --fix`, `uv run ruff format .`, and `git diff --check`
also passed. no full suite or concurrent pytest commands.

## scope

unit a only. no ui drawing changes, server restart, push, or merge. existing
finished runs retain legacy prediction-table fallback until artifacts are
written by a run update. insufficient validation residuals keep the existing
band rule (`fit_band` returns no band under four residuals).
