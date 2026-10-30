"""a seeded autoregressive panel makes the cost of stale near-week inputs measurable"""

from copy import deepcopy

import numpy as np
from test_forecast_single import build_workspace

from smolsmort.forecast import pipeline
from smolsmort.forecast.evaluate import wape
from smolsmort.forecast.spec import SERIES, STEP


def test_fresh_origins_improve_one_week_forecasts(monkeypatch):
    rng = np.random.default_rng(37)
    weeks, series = 260, 13
    positions = np.arange(weeks)
    seasonal = 4 * np.sin(positions * 2 * np.pi / 52)
    values = []
    for _ in range(series):
        state = np.zeros(weeks)
        noise = rng.normal(0, 5, weeks)
        for step in range(1, weeks):
            state[step] = 0.8 * state[step - 1] + noise[step]
        values.append(100 + seasonal + state)
    workspace = build_workspace(
        monkeypatch, weeks=weeks, series=series, values=np.concatenate(values)
    )
    genome = pipeline.make_genome(["y@day:lag+0"], "squared", {"alpha": 1.0}, "ridge")
    scored = pipeline.score(workspace, genome, nthread=1)
    frame = workspace.frames[(1, 8)]
    validation = workspace.series_masks[(1, 8)]["val"]
    truth, leads = frame.y[validation], frame.weeks_ahead[validation]
    fresh_one = wape(truth[leads == 1], scored.prediction[leads == 1])
    fresh_eight = wape(truth[leads == 8], scored.prediction[leads == 8])

    stale = deepcopy(workspace)
    stale_frame = stale.frames[(1, 8)]
    lag_column = stale_frame.features.names.index("lag_0")
    stale_frame.features.x[:, lag_column] = np.nan
    for key in np.unique(frame.series):
        panel_rows = workspace.panel[SERIES] == key
        history = dict(
            zip(workspace.panel[STEP][panel_rows], workspace.panel["y"][panel_rows], strict=True)
        )
        rows = np.flatnonzero(stale_frame.series == key)
        # the rejected single-bucket interpretation reads target minus eight for every lead
        stale_frame.features.x[rows, lag_column] = [
            history.get(step - np.timedelta64(56, "D"), np.nan) for step in stale_frame.step[rows]
        ]
    stale_scored = pipeline.score(stale, genome, nthread=1)
    stale_one = wape(truth[leads == 1], stale_scored.prediction[leads == 1])

    assert fresh_one < 0.85 * fresh_eight, (fresh_one, fresh_eight)
    assert fresh_one < 0.85 * stale_one, (fresh_one, stale_one)
