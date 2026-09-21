"""an xgboost classifier behind the loop's ModelBackend seam, for rows instead of frames.

A "frame" is a path to a one-row json feature file (`{"feature": value, ...}`). train reads each
example's row and its judged label; predict reads the same shape with no label and returns a class
+ score per row - the tabular echo of a box's `matched_template` + `score`.

FEATURE COLUMNS ARE FIXED AT TRAIN TIME, from the union of keys seen across the training rows,
sorted for a stable column order. predict fills a training row missing at inference with 0.0 rather
than refusing, since a real forecast row can legitimately lack a feature a training row had.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# torch (loaded elsewhere in the same test process) bundles its own openmp runtime; xgboost's
# homebrew libomp aborts the process on macos if it finds one already initialised. this must be
# set before xgboost's import triggers its own openmp init - see docs/REVIEW_TOOL_DESIGN.md's
# provenance rule: this default exists because the suite crashed without it, not by guess
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import xgboost as xgb

NAME = "xgboost"


class TabularBackendError(Exception):
    pass


@dataclass
class TabularWeights:
    booster: xgb.Booster
    classes: dict[str, int]
    feature_names: list[str]


def _row(path: Path) -> dict[str, float]:
    """one row's features, read from its json sidecar"""
    data = json.loads(Path(path).read_text())
    features = data.get("features", data)  # a bare feature dict is accepted too
    return {k: float(v) for k, v in features.items()}


def _example_row(item: dict) -> tuple[Path, str | None]:
    """a training example as (path, label), from the loop's dict shape"""
    return Path(item["path"]), item.get("label")


def _matrix(rows: list[dict[str, float]], feature_names: list[str]) -> np.ndarray:
    return np.array(
        [[row.get(name, 0.0) for name in feature_names] for row in rows], dtype=np.float32
    )


class TabularBackend:
    name = NAME

    def __init__(self, *, rounds: int = 50, max_depth: int = 4, seed: int = 0):
        self.rounds = rounds
        self.max_depth = max_depth
        self.seed = seed

    def train(self, examples: list, *, classes, on_progress=None) -> TabularWeights:
        pairs = [_example_row(e) for e in examples]
        labelled = [(p, label) for p, label in pairs if label is not None]
        if not labelled:
            raise TabularBackendError("no example carries a label - nothing to fit")
        rows = [_row(p) for p, _ in labelled]
        feature_names = sorted({name for row in rows for name in row})
        if not feature_names:
            raise TabularBackendError("no example row carries a feature")
        class_map = dict(classes) or {
            name: i for i, name in enumerate(sorted({lb for _, lb in labelled}))
        }
        x = _matrix(rows, feature_names)
        y = np.array([class_map[label] for _, label in labelled], dtype=np.int32)
        dtrain = xgb.DMatrix(x, label=y)
        params = {
            "max_depth": self.max_depth,
            "objective": "multi:softprob" if len(class_map) > 2 else "binary:logistic",
            "seed": self.seed,
            # single-threaded: xgboost's openmp pool crashes the process when it spins up
            # alongside torch's own openmp runtime in the same interpreter (macos libomp/libiomp
            # conflict) - the loop test suite imports both in one process
            "nthread": 1,
        }
        if len(class_map) > 2:
            params["num_class"] = len(class_map)
        booster = xgb.train(params, dtrain, num_boost_round=self.rounds)
        if on_progress:
            on_progress(self.rounds, self.rounds)
        return TabularWeights(booster=booster, classes=class_map, feature_names=feature_names)

    def predict(self, weights: TabularWeights, frames: list, *, classes) -> list[dict]:
        rows = [_row(Path(f)) for f in frames]
        x = _matrix(rows, weights.feature_names)
        dmatrix = xgb.DMatrix(x, feature_names=weights.feature_names, nthread=1)
        scores = weights.booster.predict(dmatrix)
        names = sorted(weights.classes, key=weights.classes.get)
        out: list[dict] = []
        for frame, score in zip(frames, scores, strict=True):
            if len(names) > 2:
                index = int(np.argmax(score))
                label, confidence = names[index], float(score[index])
            else:
                confidence = float(score)
                label = names[1] if confidence >= 0.5 else names[0]
            out.append({"path": str(frame), "matched_template": label, "score": confidence})
        return out

    def save(self, weights: TabularWeights, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # xgboost picks its on-disk format from the extension; ubj is its own recommended default
        weights.booster.save_model(str(path.with_name(path.name + ".ubj")))
        meta = {"backend": NAME, "classes": weights.classes, "feature_names": weights.feature_names}
        path.with_name(path.name + ".json").write_text(json.dumps(meta, indent=2))
        return path

    def load(self, path: Path) -> TabularWeights:
        path = Path(path)
        meta = json.loads(path.with_name(path.name + ".json").read_text())
        booster = xgb.Booster()
        booster.load_model(str(path.with_name(path.name + ".ubj")))
        return TabularWeights(
            booster=booster, classes=meta["classes"], feature_names=meta["feature_names"]
        )
