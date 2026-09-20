"""the box cnn behind the loop's ModelBackend seam.

the loop hands over examples as plain dicts and frames as path strings, and wants candidates back;
this is the only place that translates between that and boxes.train, so the loop never learns which
backend it is driving
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from smolsmort.backends import example_from, sidecar
from smolsmort.boxes import train as box_train
from smolsmort.boxes.model import WORK_LONG_SIDE

NAME = "box"


@dataclass
class BoxWeights:
    model: object
    classes: dict[str, int]
    long_side: int = WORK_LONG_SIDE


class BoxBackend:
    name = NAME

    def __init__(
        self,
        *,
        epochs: int = 30,
        device: str | None = None,
        long_side: int = WORK_LONG_SIDE,
        min_score: float = 0.3,
        max_per_frame: int = 50,
        seed: int = 0,
        learning_rate: float = 2e-3,
        widths: tuple[int, int, int, int, int] | None = None,
        optimizer: str = "adamw",
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        self.epochs = epochs
        self.device = device
        self.long_side = long_side
        self.min_score = min_score
        self.max_per_frame = max_per_frame
        self.seed = seed
        self.learning_rate = learning_rate
        self.widths = widths
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay

    def train(self, examples, *, classes, on_progress=None) -> BoxWeights:
        model, _ = box_train.train(
            [example_from(e) for e in examples],
            epochs=self.epochs,
            device=self.device,
            seed=self.seed,
            learning_rate=self.learning_rate,
            classes=dict(classes) or None,
            long_side=self.long_side,
            widths=self.widths,
            optimizer=self.optimizer,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            on_progress=(lambda p: on_progress(p.epoch, p.epochs)) if on_progress else None,
        )
        return BoxWeights(model=model, classes=dict(classes), long_side=self.long_side)

    def evaluate(self, weights, examples, *, classes) -> float:
        """mean training loss of the weights over examples, no augmentation"""
        return box_train.evaluate(
            weights.model,
            [example_from(e) for e in examples],
            classes=dict(classes) or None,
            long_side=weights.long_side,
        )

    def predict(self, weights, frames, *, classes) -> list[dict]:
        return box_train.sweep(
            weights.model,
            dict(classes),
            [Path(f) for f in frames],
            min_score=self.min_score,
            max_per_frame=self.max_per_frame,
            long_side=weights.long_side,
        )

    def save(self, weights, path: Path) -> Path:
        box_train.save(weights.model, path)
        meta = {"backend": NAME, "classes": weights.classes, "long_side": weights.long_side}
        sidecar(path).write_text(json.dumps(meta, indent=2))
        return path

    def load(self, path: Path) -> BoxWeights:
        meta = json.loads(sidecar(path).read_text()) if sidecar(path).is_file() else {}
        return BoxWeights(
            model=box_train.load(path, device=self.device),
            classes=meta.get("classes", {}),
            long_side=meta.get("long_side", WORK_LONG_SIDE),
        )
