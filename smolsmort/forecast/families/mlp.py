"""feed-forward relu net in torch on cpu, fed the same lag features the trees get"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smolsmort.forecast.families.base import Param, load_torch, probe_import
from smolsmort.forecast.model import ModelError, RoundErrors

OBJECTIVES = ("squared", "absolute")
INTEGER_PARAMS = ("n_layers", "width", "batch_size")
# epochs are the rounds; a cap of 300 keeps a cpu fit to seconds on tabular lag features
EPOCH_CAP = 300


@dataclass
class Preprocessor:
    """training-only medians, scales, missing flags and one-hot levels with an unknown slot"""

    columns: int
    numeric: np.ndarray
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    missing: np.ndarray
    categories: tuple[tuple[int, np.ndarray], ...]

    @classmethod
    def learn(cls, values: np.ndarray, types: list[str]) -> Preprocessor:
        numeric = np.array([i for i, kind in enumerate(types) if kind == "q"], dtype=int)
        raw = values[:, numeric]
        # entirely missing columns use zero; they still carry a missing indicator
        medians = np.array(
            [np.median(col[~np.isnan(col)]) if (~np.isnan(col)).any() else 0.0 for col in raw.T]
        )
        imputed = np.where(np.isnan(raw), medians, raw)
        means, scales = imputed.mean(axis=0), imputed.std(axis=0)
        scales[scales == 0] = 1.0
        categories = tuple(
            (i, np.unique(values[~np.isnan(values[:, i]), i]))
            for i, kind in enumerate(types)
            if kind == "c"
        )
        missing = np.flatnonzero(np.isnan(values).any(axis=0))
        return cls(values.shape[1], numeric, medians, means, scales, missing, categories)

    def transform(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.columns:
            raise ModelError("mlp needs a matrix with the training column count")
        if np.isinf(values).any():
            raise ModelError("mlp features must be finite or nan")
        numeric = values[:, self.numeric]
        imputed = np.where(np.isnan(numeric), self.medians, numeric)
        blocks = [(imputed - self.means) / self.scales]
        blocks.append(np.isnan(values[:, self.missing]).astype(float))
        for column, levels in self.categories:
            encoded = values[:, column, None] == levels
            unknown = ~encoded.any(axis=1, keepdims=True)
            blocks.extend((encoded, unknown))
        return np.column_stack(blocks).astype(np.float32)


@dataclass
class MlpFitted:
    prep: Preprocessor
    model: object
    target_mean: float
    target_scale: float
    epochs: int
    eval_history: dict | None = None

    def predict(self, x) -> np.ndarray:
        torch = load_torch()

        design = torch.from_numpy(self.prep.transform(x))
        self.model.eval()
        with torch.no_grad():
            scaled = self.model(design).squeeze(1).numpy().astype(np.float64)
        values = scaled * self.target_scale + self.target_mean
        if not np.isfinite(values).all():
            raise ModelError("mlp returned non-finite predictions")
        return values


def _build_net(torch, inputs: int, config: dict):
    layers, size = [], inputs
    for _ in range(config["n_layers"]):
        layers += [
            torch.nn.Linear(size, config["width"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(config["dropout"])),
        ]
        size = config["width"]
    layers.append(torch.nn.Linear(size, 1))
    return torch.nn.Sequential(*layers)


def _train(torch, config, objective, seed, x, y, epochs, patience=0, check=None, errors=None):
    """train for up to `epochs`; with a check set, stop early and return the best weights"""
    torch.manual_seed(seed)
    model = _build_net(torch, x.shape[1], config)
    loss_fn = torch.nn.MSELoss() if objective == "squared" else torch.nn.L1Loss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    shuffler = torch.Generator().manual_seed(seed)
    x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
    batch = config["batch_size"]
    training, validation = [], []
    best, best_epoch, best_state = np.inf, 0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_t), generator=shuffler)
        total = 0.0
        for start in range(0, len(order), batch):
            pick = order[start : start + batch]
            optimiser.zero_grad()
            loss = loss_fn(model(x_t[pick]).squeeze(1), y_t[pick])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(pick)
        training.append(total / len(x_t))
        if check is None:
            continue
        model.eval()
        with torch.no_grad():
            tail = model(check[0]).squeeze(1)
            score = float(loss_fn(tail, check[1]))
        validation.append(score)
        if errors is not None:
            errors.add(tail.numpy().astype(np.float64) * check[2] + check[3])
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif patience and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, training, validation, best_epoch + 1


class MlpFamily:
    objective_aliases = {"squared": "squared", "absolute": "absolute"}
    name = "mlp"
    label = "mlp"
    needs = "lag_features"
    pip_extra = "neural"

    def available(self) -> tuple[bool, str]:
        return probe_import("torch", self.pip_extra)

    def space(self) -> list[Param]:
        return [
            Param("n_layers", "int", 1, 4, default=2),
            Param("width", "log", 16, 512, default=64),
            Param("dropout", "float", 0.0, 0.5, default=0.1),
            Param("learning_rate", "log", 1e-4, 1e-2, default=1e-3),
            Param("weight_decay", "log", 1e-8, 1e-2, default=1e-5),
            Param("batch_size", "choice", choices=(32, 64, 128, 256), default=64),
        ]

    def defaults(self) -> dict:
        return {param.name: param.default for param in self.space()}

    def objectives(self, task: str) -> tuple[str, ...]:
        return OBJECTIVES if task == "regression" else ()

    def cost(self, shape: tuple[int, int], params: dict) -> float:
        """relative work hint from rows, summed layer widths squared and layers"""
        rows, columns = shape
        layers = int(round(float(params.get("n_layers", 2))))
        width = float(params.get("width", 64))
        widths = columns + width * layers + 1
        return float(rows * widths**2 * layers)

    def fit(
        self,
        x,
        *,
        objective: str = "squared",
        y=None,
        feature_types: list[str] | None = None,
        params: dict | None = None,
        seed: int = 0,
        nthread: int = 1,
        inner: float = 0.15,
        max_rounds: int = 2000,
        patience: int = 50,
        round_errors: bool = True,
        metric_transform: str = "none",
        metric_base=None,
    ) -> MlpFitted:
        if objective not in OBJECTIVES:
            raise ModelError(f"no mlp objective called {objective!r}")
        torch = load_torch()

        values = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if values.ndim != 2 or target.shape != (len(values),):
            raise ModelError("mlp needs a feature matrix and one target per row")
        types = ["q"] * values.shape[1] if feature_types is None else feature_types
        if len(types) != values.shape[1] or any(kind not in {"c", "q"} for kind in types):
            raise ModelError("mlp needs one c or q type marker per feature")
        rows = np.flatnonzero(np.isfinite(target))
        if len(rows) < 20:
            raise ModelError(f"only {len(rows)} rows carry a usable target - too few to fit")
        config = {**self.defaults(), **(params or {})}
        # a log-scale search draws floats; layer counts, widths and batches are integers
        for name in INTEGER_PARAMS:
            config[name] = int(round(float(config[name])))
        epochs_cap = min(int(max_rounds), EPOCH_CAP)
        metric = "squared_error" if objective == "squared" else "absolute_error"

        previous_threads = torch.get_num_threads()
        previous_det = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(max(1, int(nthread)))
        torch.use_deterministic_algorithms(True)
        try:
            cut = int(len(rows) * (1 - inner))
            head, tail = rows[:cut], rows[cut:]
            best_epochs, history = epochs_cap, None
            if patience and len(head) >= 10 and len(tail) >= 10:
                # the probe sees only head statistics, so the tail stays honest
                prep = Preprocessor.learn(values[head], types)
                mean, scale = float(target[head].mean()), float(target[head].std()) or 1.0
                check = (
                    torch.from_numpy(prep.transform(values[tail])),
                    torch.from_numpy(((target[tail] - mean) / scale).astype(np.float32)),
                    scale,
                    mean,
                )
                errors = (
                    RoundErrors(
                        target[tail],
                        metric_transform,
                        None if metric_base is None else metric_base[tail],
                    )
                    if round_errors
                    else None
                )
                _, training, validation, best_epochs = _train(
                    torch,
                    config,
                    objective,
                    seed,
                    prep.transform(values[head]),
                    ((target[head] - mean) / scale).astype(np.float32),
                    epochs_cap,
                    patience,
                    check,
                    errors,
                )
                if np.isfinite([training, validation]).all():
                    history = {"metric": metric, "training": training, "validation": validation}
                    curves = errors.curves(len(validation)) if errors else None
                    if curves:
                        history["curves"] = curves
            # refit on every usable row for the best epoch count, like lightgbm
            prep = Preprocessor.learn(values[rows], types)
            mean, scale = float(target[rows].mean()), float(target[rows].std()) or 1.0
            model, training, _, _ = _train(
                torch,
                config,
                objective,
                seed,
                prep.transform(values[rows]),
                ((target[rows] - mean) / scale).astype(np.float32),
                best_epochs,
            )
            if history is None:
                history = {"metric": metric, "training": training}
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_det)
        return MlpFitted(prep, model, mean, scale, best_epochs, history)
