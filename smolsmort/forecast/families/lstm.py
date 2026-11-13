"""global lstm family: one torch model over all eligible series, fed raw weekly history"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from smolsmort.forecast.families.base import Param, load_torch, probe_import
from smolsmort.forecast.families.stats import RawSeriesTask
from smolsmort.forecast.spec import SERIES, STEP

MAX_EPOCHS = 100
PATIENCE = 8
BATCH = 64
VAL_SHARE = 0.2
SCALE_FLOOR = 1e-6
# the model is refit on every fourth origin and reused in between, like stats.py
REFIT_EVERY = 4

SPACE = [
    Param("hidden", "int", 8, 128, default=32),
    Param("layers", "int", 1, 2, default=1),
    Param("lookback", "int", 13, 104, default=26),
    Param("dropout", "float", 0.0, 0.4, default=0.1),
    Param("learning_rate", "log", 1e-4, 1e-2, default=1e-3),
]


@dataclass
class LstmFitted:
    prediction: np.ndarray
    season_length: int
    eval_history: dict | None = None
    warnings: tuple[str, ...] = ()

    def predict(self, task: RawSeriesTask) -> np.ndarray:
        return self.prediction.copy()


def resolve_params(params: dict | None) -> dict:
    """merge over defaults; integer genes arrive as floats"""
    merged = {p.name: p.default for p in SPACE} | dict(params or {})
    return {
        "hidden": max(1, round(merged["hidden"])),
        "layers": min(2, max(1, round(merged["layers"]))),
        "lookback": max(2, round(merged["lookback"])),
        "dropout": float(merged["dropout"]),
        "learning_rate": float(merged["learning_rate"]),
    }


def compute_leads(task: RawSeriesTask, rows: np.ndarray) -> np.ndarray:
    """weeks_ahead when the frame carries it, else whole units between origin and step"""
    frame = task.frame
    if frame.weeks_ahead is not None:
        return np.asarray(frame.weeks_ahead[rows], dtype=int)
    step = np.asarray(frame.step[rows]).astype("datetime64[D]")
    origin = np.asarray(frame.origin[rows]).astype("datetime64[D]")
    if task.unit in ("day", "week"):
        days = (step - origin).astype(int)
        return days // (7 if task.unit == "week" else 1)
    months = step.astype("datetime64[M]").astype(int) - origin.astype("datetime64[M]").astype(int)
    return months // {"month": 1, "quarter": 3, "year": 12}[task.unit]


def scale_of(values: np.ndarray) -> float:
    return max(float(np.mean(np.abs(values))), SCALE_FLOOR)


def make_windows(values: np.ndarray, lookback: int, horizon: int, last: int):
    """windows whose targets all sit at or before index `last`; short inputs are edge padded"""
    min_end = min(lookback, 26) - 1
    ends = np.arange(min_end, last - horizon + 1)
    if len(ends) == 0:
        return np.empty((0, lookback)), np.empty((0, horizon)), ends
    padded = np.concatenate([np.full(lookback, values[0]), values])
    inputs = np.lib.stride_tricks.sliding_window_view(padded, lookback)[ends + 1]
    targets = np.lib.stride_tricks.sliding_window_view(values, horizon)[ends + 1]
    return inputs, targets, ends


def last_window(values: np.ndarray, lookback: int) -> np.ndarray:
    padded = np.concatenate([np.full(lookback, values[0]), values])
    return padded[-lookback:]


def build_network(torch, hidden: int, layers: int, dropout: float, horizon: int):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(
                1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0
            )
            self.drop = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(hidden, horizon)

        def forward(self, x):
            out, _ = self.rnn(x.unsqueeze(-1))
            return self.head(self.drop(out[:, -1]))

    return Network()


def train_network(torch, inputs, targets, order, horizon, params, seed):
    """mse and adam; the newest share of windows (by end index) drives early stopping"""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    sorted_idx = np.argsort(order, kind="stable")
    cut = max(1, int(round(len(sorted_idx) * (1 - VAL_SHARE))))
    if cut >= len(sorted_idx):
        cut = len(sorted_idx)
    train_idx, val_idx = sorted_idx[:cut], sorted_idx[cut:]
    x = torch.as_tensor(inputs, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    net = build_network(torch, params["hidden"], params["layers"], params["dropout"], horizon)
    optimizer = torch.optim.Adam(net.parameters(), lr=params["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    history = {"metric": "squared_error", "training": [], "validation": []}
    best, best_state, stale = float("inf"), None, 0
    train_t = torch.as_tensor(train_idx)
    for _ in range(MAX_EPOCHS):
        net.train()
        shuffled = train_t[torch.randperm(len(train_t), generator=generator)]
        total = 0.0
        for start in range(0, len(shuffled), BATCH):
            batch = shuffled[start : start + BATCH]
            optimizer.zero_grad()
            loss = loss_fn(net(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history["training"].append(total / len(shuffled))
        if len(val_idx):
            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(x[val_idx]), y[val_idx]))
        else:
            val_loss = history["training"][-1]
        history["validation"].append(val_loss)
        if val_loss < best - 1e-9:
            best = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, history


def forecast_rows(task: RawSeriesTask, params: dict, seed: int, nthread: int):
    """returns predictions aligned to task.rows and the first model's epoch history"""
    torch = load_torch()

    frame, rows = task.frame, task.rows
    series = np.asarray(frame.series[rows])
    origins = np.asarray(frame.origin[rows]).astype("datetime64[ns]")
    leads = compute_leads(task, rows)
    prediction = np.full(len(series), np.nan)
    if not len(series):
        return prediction, None
    horizon = int(task.bucket[1])
    if leads.min() < 1 or leads.max() > horizon:
        raise ValueError("lstm forecast leads must be within the requested horizon")
    lookback = params["lookback"]
    steps_all = np.asarray(task.panel[STEP]).astype("datetime64[ns]")
    # per series history sorted by step, nonfinite values filled with the finite mean
    history = {}
    for key in np.unique(task.panel[SERIES]):
        mine = np.flatnonzero(task.panel[SERIES] == key)
        mine = mine[np.argsort(steps_all[mine], kind="stable")]
        values = np.asarray(task.panel[task.target][mine], dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            history[key] = (steps_all[mine], np.where(finite, values, values[finite].mean()))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, nthread))
    first_history, net = None, None
    try:
        for number, origin in enumerate(np.unique(origins)):
            at = origins == origin
            # every series is cut at this origin: nothing later is ever read
            cut = {}
            for key, (steps, values) in history.items():
                end = int(np.searchsorted(steps, origin, side="right")) - 1
                if end >= 0:
                    cut[key] = values[: end + 1]
            for key in np.unique(series[at]):
                if key not in history:
                    raise ValueError(f"no finite history for series {key!r}")
                steps = history[key][0]
                end = int(np.searchsorted(steps, origin, side="right")) - 1
                if end < 0 or steps[end] != origin:
                    raise ValueError(f"unobserved origin for series {key!r}: {origin}")
            if number % REFIT_EVERY == 0:
                xs, ys, ends = [], [], []
                for values in cut.values():
                    scaled = values / scale_of(values)
                    inputs, targets, end = make_windows(scaled, lookback, horizon, len(values) - 1)
                    xs.append(inputs)
                    ys.append(targets)
                    ends.append(end)
                inputs, targets, order = (
                    np.concatenate(xs),
                    np.concatenate(ys),
                    np.concatenate(ends),
                )
                if not len(inputs):
                    raise ValueError(f"insufficient history to build windows at {origin}")
                net, curves = train_network(
                    torch, inputs, targets, order, horizon, params, seed + number
                )
                if first_history is None:
                    first_history = curves
            for key in np.unique(series[at]):
                values = cut[key]
                scale = scale_of(values)
                window = torch.as_tensor(
                    last_window(values / scale, lookback)[None], dtype=torch.float32
                )
                with torch.no_grad():
                    forecast = net(window)[0].numpy().astype(float) * scale
                hit = np.flatnonzero(at & (series == key))
                prediction[hit] = forecast[leads[hit] - 1]
    finally:
        torch.set_num_threads(old_threads)
    return prediction, first_history


class LstmFamily:
    name = "lstm"
    label = "lstm"
    needs = "raw_series"
    pip_extra = "neural"

    def available(self) -> tuple[bool, str]:
        return probe_import("torch", self.pip_extra)

    def space(self) -> list[Param]:
        return list(SPACE)

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.space()}

    def objectives(self, task: str) -> tuple[str, ...]:
        if task != "regression":
            raise ValueError(f"family {self.name!r} requires regression")
        return ("squared",)

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        rows, _ = shape
        p = resolve_params(params)
        return float(rows * p["layers"] * p["hidden"] ** 2 / 64)

    def fit(
        self,
        x: RawSeriesTask,
        *,
        objective: str,
        y=None,
        feature_types=None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        **options,
    ) -> LstmFitted:
        from smolsmort.forecast.pipeline import PipelineError

        available, reason = self.available()
        if not available:
            raise PipelineError(f"family {self.name!r} is unavailable; {reason}")
        try:
            resolved = resolve_params(params)
            # ineligible series never reach the model and stay n/a
            selected = x.rows & np.isin(x.frame.series, list(x.eligibility.eligible))
            task = replace(x, rows=selected)
            prediction = np.full(int(x.rows.sum()), np.nan)
            covered = selected[x.rows]
            curves = None
            if selected.any():
                values, curves = forecast_rows(task, resolved, seed, nthread)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite predictions for eligible series")
                prediction[covered] = values
        except (ValueError, IndexError, KeyError, RuntimeError, FloatingPointError) as exc:
            raise PipelineError(
                f"family {self.name!r} failed in bucket {x.bucket}: "
                f"{type(exc).__name__}: {exc}; check history at the earliest origin"
            ) from exc
        return LstmFitted(prediction, x.season_length, eval_history=curves)
