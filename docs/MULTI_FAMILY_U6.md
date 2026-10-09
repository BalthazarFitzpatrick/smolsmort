# u6: request v2 and per-family workers

target os: windows, mac, linux. local runtime checks use macos and python 3.13.13.

unversioned requests retain the v1 budget and xgboost search. v2 requests carry `spec`, `prepared`, `summary`, `nthread`, `transform`, `families`, `time_budget_s`, `ensemble`, `warm_from` and `recipe`. only `transform: "none"` is supported. family methods are `genetic` and `grid`; names must be unique and safe folder names.

```json
{
  "version": 2,
  "transform": "none",
  "families": [
    {"name": "xgboost", "method": "grid", "space": {"max_depth": [3, 6], "eta": [0.05, 0.1]}},
    {"name": "snaive", "method": "genetic"}
  ],
  "time_budget_s": 120,
  "ensemble": {"enabled": false, "top": 3}
}
```

the example omits the prepared-data fields. `runs.start_run` accepts these settings as keyword arguments and writes the complete request. space overrides are lists of parameter values. grid searches evaluate their cartesian product for each offered objective; parameters without overrides retain their defaults. genetic searches draw from the provider's typed space, with overrides restricting the selected parameters. xgboost keeps its profile-filtered objectives, feature variation and original random draw order when no overrides are supplied. providers can expose `objectives(task)`; older providers without that method default to squared error.

`worker.py` launches every chosen family through `sys.executable -m smolsmort.forecast.family_worker`. each family receives the full search duration and `max(1, nthread // family_count)` threads. workers load their workspace once and write status, events, leaderboard, generation curves, evaluation history, winner, population and a checkpoint inside `families/<name>/`. empty spaces bypass sampling and evaluate exactly one full-data candidate. missing libraries and failed fits stay local to the family.

checkpoints atomically retain completed scores, contributions, histories, population, generation, plateau counter, random state and consumed search time. restarting the same worker reuses cached scores and the remaining search budget. an interrupted candidate can be fitted again. completed family folders are skipped. warm starts read only the matching family's population; xgboost also accepts an older v1 population.

the launcher polls per-family status into the top-level `families` map, merges leaderboards with a `family` field, and selects the lowest finite validation fitness from completed families. only that winner goes through the existing `finish` path. the saved v2 recipe carries the winning family and `transform: "none"`.

cancellation uses a file marker. the launcher terminates remaining child processes, waits, then kills if needed. process polling uses `subprocess`; windows reattachment queries the process exit code through the stdlib's `ctypes`. neither cancellation nor child control requires posix signals.

## verification

`tests/test_forecast_runs_v2.py` covers v1 parsing, invalid v2 requests, single-candidate statistical searches, provider-specific spaces/objectives, grid deadlines, parallel worker pids, merged output and final result, a simulated missing library, a real killed/restarted worker, completed-folder skipping, deterministic cache reuse and the torch refusal. existing search, route and system tests cover the v1 path.

baseline verification executes the original search from `56a531f` and the new search on the same synthetic panel and seed, comparing serialized leaderboard bytes. it also compares the checked-in golden file's bytes with that commit. full-suite output from the no-extras ci environment and the forecast/stats environment is retained at `/private/tmp/multifamily-u6-tests.txt`.

## limits

deadlines are checked between candidates. a candidate already fitting can finish after the search deadline; final refitting and worker startup also add wall time. resume charges time recorded in the last checkpoint and repeats an interrupted fit. fewer threads than families still means at least one thread per process.

the allocator, ensemble fitting and extra verdicts belong to u7. request fields for the ensemble are persisted but do not trigger an ensemble in u6. new routes and per-family event aggregation belong to u8. windows runtime and browser checks were not performed locally.
