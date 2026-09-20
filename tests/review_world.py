"""a throwaway review world for tests: real frames on disk under a tmp_path, `paths` repointed at
it, and FAKE seams (a torch-free model backend, a scripted playback reader) - never real data."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from smolsmort import backends
from smolsmort.review import paths
from smolsmort.review.classscheme import definitions

FRAME_SIZE = (200, 120)  # width, height
BOX = (30, 10)  # width, height of every drawn box


def write_frame(path: Path, seed: int = 0) -> None:
    """a small deterministic frame; content is irrelevant, only its size and that it decodes"""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def make_world(tmp_path: Path, monkeypatch, recordings=("rec_a",), frames: int = 3):
    """sessions/<recording>/frames/f00.png..., and every `paths` directory under tmp_path"""
    training = tmp_path / "training"
    world = SimpleNamespace(
        root=tmp_path,
        sessions=tmp_path / "sessions",
        labels=training / "boxes",
        tiles=training / "tiles",
        sets=training / "sets",
        checkpoints=training / "weights" / "checkpoints",
        classes=training / "classes",
        ui=tmp_path / "ui",
    )
    for directory in (
        world.sessions,
        world.labels,
        world.tiles,
        world.sets,
        world.checkpoints,
        world.classes,
        world.ui,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "SESSIONS_DIR", world.sessions)
    monkeypatch.setattr(paths, "LABELS_DIR", world.labels)
    monkeypatch.setattr(paths, "TILES_DIR", world.tiles)
    monkeypatch.setattr(paths, "TILES_SYNTH_DIR", training / "tiles-synth")
    monkeypatch.setattr(paths, "DATASETS_DIR", world.sets)
    monkeypatch.setattr(paths, "CHECKPOINTS_DIR", world.checkpoints)
    monkeypatch.setattr(definitions, "DEFS_DIR", world.classes)
    for name in recordings:
        for i in range(frames):
            write_frame(world.sessions / name / "frames" / f"f{i:02d}.png", seed=i)
    return world


def drawn_boxes(frames: int = 3) -> list[dict]:
    """two boxes per frame, well inside it"""
    boxes = []
    for i in range(frames):
        for left, top in ((40, 40), (110, 60)):
            boxes.append(
                {
                    "path": f"f{i:02d}.png",
                    "left": left,
                    "top": top,
                    "width": BOX[0],
                    "height": BOX[1],
                }
            )
    return boxes


def save_definition(name: str = "kinds", members=("alpha", "beta")) -> None:
    definition = definitions.ClassDef(
        name=name, dimensions=[definitions.Dimension(name="kind", members=list(members))]
    )
    definitions.save(definition)


# ---------------------------------------------------------------- a fake model backend


@dataclass
class WorldWeights:
    classes: dict[str, int]
    trained_on: int
    box: tuple[int, int]


class WorldBackend:
    """the model-backend seam, faked. 'trains' by remembering the class map, 'predicts' one
    candidate per frame at the box size it was trained on. accepts the options the train tab
    sends, so a run can prove they reached it."""

    seen: dict = {}

    def __init__(self, *, epochs: int = 2, learning_rate: float = 3e-4, seed: int = 0):
        self.epochs = epochs
        type(self).seen = {"epochs": epochs, "learning_rate": learning_rate, "seed": seed}

    def train(
        self,
        examples: list[dict],
        *,
        classes: Mapping[str, int],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> WorldWeights:
        usable = [e for e in examples if e["centres"]]
        if not usable:
            raise ValueError("no example has a confirmed object")
        for epoch in range(1, self.epochs + 1):
            if on_progress:
                on_progress(epoch, self.epochs)
        return WorldWeights(dict(classes), len(usable), (usable[0]["width"], usable[0]["height"]))

    def predict(self, weights, frames, *, classes) -> list[dict]:
        names = sorted(classes, key=classes.get) or ["object"]
        width, height = weights.box
        return [
            {
                "path": Path(frame).name,
                "left": 50 + i,
                "top": 30,
                "width": width,
                "height": height,
                "matched_template": names[i % len(names)],
                "score": 0.9 - 0.05 * i,
            }
            for i, frame in enumerate(frames)
        ]

    def save(self, weights, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(weights)))
        return path

    def load(self, path: Path) -> WorldWeights:
        data = json.loads(path.read_text())
        return WorldWeights(data["classes"], data["trained_on"], tuple(data["box"]))


def register_world_backend() -> None:
    backends.register("world", "review_world", "WorldBackend")


def unregister_world_backend() -> None:
    backends._REGISTRY.pop("world", None)


# ---------------------------------------------------------------- a fake playback reader


class FakeReader:
    """the playback seam, faked: one recording, whose frames just echo the moment asked for"""

    def sessions_under(self, root: Path) -> list[dict]:
        return [{"name": "rec_a"}]

    def load(self, root: Path, name: str):
        return SimpleNamespace(
            name=name,
            duration=4.0,
            states=[1, 2],
            events=[],
            labels=[(0, "x"), (1, None)],
            start=10.0,
        )

    def frame_at(self, loaded, when: float) -> dict:
        return {"when": when}
