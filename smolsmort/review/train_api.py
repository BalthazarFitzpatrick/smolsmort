"""the train and sweep tabs' server side: one object routes.py talks to for everything model-shaped.

a thin layer over `TrainState` (train.py), which drives a `ModelBackend` and holds the state of
one run. this adds what the web tool needs around it: turning a promoted training set on disk into
the examples a run binds, the train/val split, clamped start options, named checkpoints under the
checkpoints folder, and the sweep that turns a trained model back into a candidates file - the
return edge of the loop. it never imports a backend's own module for anything the seam covers.

`status()` is TrainState's own snapshot, passed through untouched: whatever fields TrainState
reports (progress, losses, a `final` flag) reach the page without this layer knowing them.
"""

from __future__ import annotations

import inspect
import json
import statistics
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from smolsmort import backends
from smolsmort.detect.dataset import DatasetError, Example, build_training_set
from smolsmort.review import hyperparams, paths, setconfig, splits
from smolsmort.review.naming import set_filename
from smolsmort.review.recordings import flat_recording, frame_files, frames_dir_of
from smolsmort.review.train import TrainState, TrainStateError, saved_checkpoints

CHECKPOINT_SUFFIX = ".pt"


def counts_above(candidates: list[dict]) -> list[int]:
    """how many proposals sit at or above each 0.01 step, so a slider can say so without asking.

    101 integers rather than the scores themselves: exact at every position the slider can take,
    the same tiny payload for 12 proposals or 120,000, and moving the threshold costs no round trip.
    """
    counts = [0] * 101
    for candidate in candidates:
        step = int(max(0.0, min(1.0, candidate.get("score", 0.0))) * 100)
        counts[step] += 1
    running = 0
    for i in range(100, -1, -1):
        running += counts[i]
        counts[i] = running
    return counts


def example_dict(example: Example) -> dict:
    """an Example as the loop's dict shape, which every backend and fake accepts. `width` and
    `height` are the first object's, for a fixed-size backend that reads a single size."""
    width, height = example.sizes[0] if example.sizes else (0, 0)
    return {
        "path": str(example.path),
        "centres": list(example.centres),
        "labels": list(example.labels),
        "sizes": list(example.sizes),
        "ignore": list(example.ignore),
        "negatives": list(example.negatives),
        "exhaustive": example.exhaustive,
        "width": width,
        "height": height,
    }


class TrainApi:
    def __init__(self, state, trainer: TrainState | None = None, backend: str = "heatmap"):
        self.state = state
        self.trainer = trainer if trainer is not None else TrainState(backend)
        self.lock = threading.Lock()
        self._examples: list[Example] = []
        self.sweep_job: dict = self._idle_sweep()

    # ---------------------------------------------------------------- the bound set

    @staticmethod
    def _set_path(name: str) -> Path:
        """a set's file: a plain name, or a path relative to the sets folder. a path that would
        leave the folder raises ValueError, so a picker cannot bind something outside it"""
        return setconfig.set_path(name)

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _load_set(self, name: str) -> tuple[list[Example], dict[str, int]]:
        """the TRAIN split of a promoted set, rebuilt from disk each time - the set can change
        under us, and a cached list would train on yesterday's labels. a set written before
        splits existed reads as all train."""
        path = self._set_path(name)
        # the fixed-size backend has no box size that is right across two capture resolutions;
        # a size-aware one rescales its input, so the check does not apply to it
        refuse = self.trainer.backend_name != "box"
        examples, classes = build_training_set(
            path, paths.SESSIONS_DIR, refuse_mixed_resolutions=refuse
        )
        return splits.keep(examples, self._rows(path), paths.SESSIONS_DIR, "train"), classes

    def _use_backend(self, name: str) -> None:
        """make the trainer drive this backend, rebuilding it only when the name changes"""
        if name == self.trainer.backend_name:
            return
        backend = backends.get_backend(name)
        self.trainer.backend_name = name
        self.trainer.backend_options = {}
        self.trainer._backend = backend

    def _adopt_set_backend(self, name: str) -> None:
        """switch to the backend a set names. a set without a `_backend.json` keeps whatever the
        trainer already drives, so sets written before the file existed train as they did"""
        config = setconfig.read_set_config(name)
        if config:
            self._use_backend(config["backend"])

    def set_backend(self, name: str, backend: str, size_mode: str | None = None) -> dict:
        """name the backend (and box size mode) a training set trains with, and remember it.

        an unknown backend is refused with the known names. the set need not exist yet: the mode
        matters while boxes are still being drawn for it.
        """
        if not name:
            return {"error": "no set name given"}
        try:
            saved = setconfig.write_set_config(name, backend, size_mode)
        except setconfig.SetConfigError as exc:
            return {"error": str(exc), "backends": backends.names()}
        except (OSError, ValueError) as exc:
            return {"error": str(exc)}
        self.state.active_set = name
        if self.trainer.training_set == name:
            self.bind(name)
        return {"name": name, **saved}

    def bind(self, name: str) -> dict:
        """which promoted set the next run trains on. binding drops any loaded weights: a
        checkpoint carries its own channel map, and keeping one against a new set would offer
        that model's classes for this set's data."""
        if not name:
            self._examples = []
            self.trainer.bind([], {}, training_set=None)
            return self.class_names()
        try:
            self._adopt_set_backend(name)
            examples, classes = self._load_set(name)
        except (DatasetError, OSError, ValueError, backends.BackendError, ImportError) as exc:
            return {"error": str(exc), **self.class_names()}
        self.state.active_set = name
        self._examples = examples
        self.trainer.bind([example_dict(e) for e in examples], classes, training_set=name)
        return self.class_names()

    def class_names(self) -> dict:
        """the channel order of what is bound or loaded. `set` and `weights` are both always
        present: the page paints both heads from this one response."""
        names = self.trainer.class_names()
        return {
            "set": names["training_set"],
            "weights": names["weights"],
            "classes": names["classes"],
        }

    def info(self) -> dict:
        """what the OPEN training set holds, and nothing else.

        the one thing that decides a run at small data sizes is how thin the rarest class is: a
        wide head over a handful of examples each is the real limit, not the frame count.
        """
        name = self.trainer.training_set
        base = {"device": self._device(), "model_exists": self.trainer.weights is not None}
        if not name:
            return {"set": None, **base}
        base.update(setconfig.set_config(name))
        try:
            path = self._set_path(name)
            counts = dict.fromkeys(splits.SPLITS, 0)
            for split in splits.of_rows(self._rows(path)).values():
                counts[split] += 1
        except OSError as exc:
            return {"error": str(exc), "set": name, **base}
        per_class: dict[str, int] = {}
        for example in self._examples:
            for label in example.labels:
                if label:
                    per_class[label] = per_class.get(label, 0) + 1
        thinnest = min(per_class.items(), key=lambda kv: kv[1]) if per_class else None
        return {
            "set": name,
            "frames": len(self._examples),
            "objects": sum(len(e.centres) for e in self._examples),
            "negatives": sum(len(e.negatives) for e in self._examples),
            "split": counts,
            "classes": len(per_class),
            "thinnest": {"label": thinnest[0], "count": thinnest[1]} if thinnest else None,
            **base,
        }

    @staticmethod
    def _device() -> str:
        try:
            import torch
        except ImportError:
            return "torch not installed"
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def frames(self) -> list[dict]:
        """every frame carrying at least one confirmed object - the list down the side of the page"""
        return [
            {
                "index": i,
                "name": example.path.name,
                "objects": example.object_count,
                "ignored": len(example.ignore),
            }
            for i, example in enumerate(self._examples)
            if example.object_count
        ]

    def trained_box_size(self) -> tuple[int, int]:
        """the median box size of the bound set, falling back to the find tab's tile size - the
        size is a constant of the data, so it is carried in from what the model was trained on"""
        name = self.trainer.training_set
        if name and self._set_path(name).is_file():
            sizes = [
                (r["width"], r["height"])
                for r in self._rows(self._set_path(name))
                if not r.get("negative")
            ]
            if sizes:
                return (
                    int(statistics.median(w for w, _ in sizes)),
                    int(statistics.median(h for _, h in sizes)),
                )
        return self.state.uniform_width, self.state.uniform_height

    def window_floor(self) -> dict:
        """the smallest training window this set's boxes fit inside, whole. reported, not just
        enforced: a number that looks reasonable and quietly clips a box is the failure mode."""
        from smolsmort.detect.model import DEFAULT_DOWNSCALE as DOWNSCALE
        from smolsmort.detect.train import CROP, minimum_window, snapped_window

        width, height = self.trained_box_size()
        floor = snapped_window(minimum_window(width, height))
        return {
            "floor": floor,
            "box": [width, height],
            "default": max(CROP, floor),
            "downscale": DOWNSCALE,
        }

    # ---------------------------------------------------------------- training

    def _accepted(self, options: dict) -> dict:
        """only the options the bound backend's constructor takes, so a fake or a backend without
        a batch size is not handed keywords it never declared"""
        params = inspect.signature(type(self.trainer._backend).__init__).parameters
        return {k: v for k, v in options.items() if v is not None and k in params}

    def start(self, payload: dict) -> dict:
        """clamp the options, apply them to the backend, and train on a background thread.

        CLAMPED LIKE THE OLD POPUP: a learning rate is the one hyperparameter that can waste a
        long run outright, so a value outside the range that ever makes sense is corrected rather
        than trusted. a `name` saves the run's weights under it once it finishes.
        """
        if not self.trainer.examples:
            return {"error": "nothing bound to train on - bind a training set first"}
        # the set names its backend: build that one, whatever the trainer was started with
        try:
            self._adopt_set_backend(self.trainer.training_set or "")
        except (backends.BackendError, ImportError) as exc:
            return {"error": str(exc)}
        epochs = max(1, min(2000, int(payload.get("epochs", 200))))
        batch = max(1, min(64, int(payload.get("batch", 8))))
        crop = payload.get("crop")
        rate = payload.get("learning_rate")
        rate = None if rate in (None, "") else max(1e-6, min(1e-1, float(rate)))
        seed = payload.get("seed")
        seed = None if seed in (None, "") else int(seed)
        name = payload.get("name")
        name = None if name in (None, "") else str(name)

        try:
            model_options = self._model_options(payload)
        except (hyperparams.HyperparamError, ValueError) as exc:
            return {"error": str(exc)}
        options = self._accepted(
            {
                "epochs": epochs,
                "batch": batch,
                "crop": None if crop is None else int(crop),
                "learning_rate": rate,
                "seed": seed,
                **model_options,
            }
        )
        # TrainState builds its backend once at construction, so options chosen at start time
        # have to rebuild it
        self.trainer.backend_options.update(options)
        self.trainer._backend = backends.get_backend(
            self.trainer.backend_name, **self.trainer.backend_options
        )
        # the window goes to the run, which refuses one the box would not fit inside. a backend
        # whose train() has no window knob is not handed one
        takes_window = "window" in inspect.signature(self.trainer._backend.train).parameters
        started = self.trainer.start(
            window=int(crop) if takes_window and crop is not None else None
        )
        if name and "error" not in started:
            threading.Thread(target=self._save_when_done, args=(name,), daemon=True).start()
        return started

    def _model_options(self, payload: dict) -> dict:
        """the config menu's choices as constructor options: optimiser settings as they are, and
        a size name turned into what this backend's model is built from (channels or widths)"""
        found: dict = {}
        if payload.get("optimizer"):
            found["optimizer"] = str(payload["optimizer"])
        for key in ("momentum", "weight_decay"):
            if payload.get(key) not in (None, ""):
                found[key] = float(payload[key])
        size = payload.get("size")
        if size:
            if self.trainer.backend_name == "heatmap":
                custom = payload.get("custom_channels")
                found["channels"] = hyperparams.heatmap_channels_for(
                    str(size), custom_channels=None if custom in (None, "") else int(custom)
                )
            elif self.trainer.backend_name == "box":
                scale = payload.get("custom_scale")
                found["widths"] = hyperparams.box_widths_for(
                    str(size), custom_scale=None if scale in (None, "") else float(scale)
                )
        return found

    def _save_when_done(self, name: str) -> None:
        """wait for the run to end, and save it under `name` only if it finished cleanly"""
        while self.trainer.status().get("running"):
            time.sleep(0.05)
        if self.trainer.status().get("finished"):
            self.save_checkpoint(name, "")

    def abort(self) -> dict:
        return self.trainer.abort()

    def status(self) -> dict:
        return self.trainer.status()

    # ---------------------------------------------------------------- checkpoints

    def checkpoint_folders(self, under: str = "") -> dict:
        return self.trainer.checkpoint_folders(paths.CHECKPOINTS_DIR, under)

    def saved_checkpoints(self) -> dict:
        return saved_checkpoints(paths.CHECKPOINTS_DIR)

    def save_checkpoint(self, name: str | None = None, folder: str = "") -> dict:
        """snapshot the current weights under a name, in a folder under the checkpoints root"""
        root = paths.CHECKPOINTS_DIR
        if self.trainer.weights is None:
            return {"error": "nothing trained or loaded yet - train a model first"}
        stem = set_filename(name or self.checkpoint_folders(folder)["name"])
        target = (root / folder / stem).with_suffix(CHECKPOINT_SUFFIX).resolve()
        if root.resolve() not in target.parents:
            return {"error": "that folder is outside the checkpoints directory"}
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            saved = self.trainer.save(target)
        except TrainStateError as exc:
            return {"error": str(exc)}
        return {"name": target.relative_to(root.resolve()).as_posix(), **saved}

    def load_checkpoint(self, name: str) -> dict:
        path = self.trainer.resolve_checkpoint(paths.CHECKPOINTS_DIR, name)
        if path is None:
            return {"error": f"no checkpoint called {name!r}"}
        self._examples = []
        try:
            self.trainer.load_weights(path)
        except (OSError, ValueError, KeyError) as exc:
            return {"error": f"could not load {name}: {exc}"}
        return self.class_names()

    # ---------------------------------------------------------------- inspecting a model

    def peak_distribution(self, frames: int = 8) -> dict:
        """where the model's response falls, split by whether it sits on a confirmed object.

        THIS IS WHAT A THRESHOLD IS CHOSEN AGAINST: response on an object should sit above the
        cut and response on background below it, and the gap between them is the thing to look
        at. sampled, not peak-decoded - reading the heatmap directly costs array indexing.
        NOT A HOLDOUT: these are frames the model was fitted on, the optimistic case.
        """
        try:
            from smolsmort.detect.model import DEFAULT_DOWNSCALE as DOWNSCALE
            from smolsmort.detect.model import STRIDE
            from smolsmort.detect.train import SCORE_BUCKETS
        except ImportError as exc:
            return {"error": f"the heatmap backend is unavailable: {exc}"}
        if self.trainer.weights is None:
            return {"error": "no trained model yet"}
        examples = [e for e in self._examples if e.centres][:frames]
        if not examples:
            return {"error": "no frames with confirmed objects"}

        scale = STRIDE * DOWNSCALE
        width, height = self.trained_box_size()
        on_object: list[float] = []
        elsewhere: list[float] = []
        try:
            for example in examples:
                hot = self.trainer.channel_heatmap(example.path, -1)
                rows, cols = hot.shape
                mask = np.zeros(hot.shape, dtype=bool)
                for cx, cy in example.centres:
                    r, c = int(cy / scale), int(cx / scale)
                    dr, dc = max(1, int(height / scale / 2)), max(1, int(width / scale / 2))
                    lo_r, hi_r = max(0, r - dr), min(rows, r + dr + 1)
                    lo_c, hi_c = max(0, c - dc), min(cols, c + dc + 1)
                    if hi_r > lo_r and hi_c > lo_c:
                        on_object.append(float(hot[lo_r:hi_r, lo_c:hi_c].max()))
                        mask[lo_r:hi_r, lo_c:hi_c] = True
                elsewhere.extend(hot[~mask].ravel().tolist())
        except (TrainStateError, ImportError) as exc:
            return {"error": str(exc)}

        def histogram(values: list[float]) -> list[int]:
            counts, _ = np.histogram(values, bins=SCORE_BUCKETS, range=(0.0, 1.0))
            return [int(n) for n in counts]

        return {
            "bins": SCORE_BUCKETS,
            "on_object": histogram(on_object),
            "elsewhere": histogram(elsewhere),
            "counts": {"on_object": len(on_object), "elsewhere": len(elsewhere)},
            "frames": len(examples),
            "highest": round(float(max(on_object, default=0.0)), 3),
        }

    def overlay_bytes(self, index: int, channel: int = -1) -> bytes | None:
        """the model's response on one bound frame as a greyscale png, or None when there is none"""
        if self.trainer.weights is None or not 0 <= index < len(self._examples):
            return None
        try:
            hot = self.trainer.channel_heatmap(self._examples[index].path, channel)
        except (TrainStateError, ImportError, OSError):
            return None
        buf = BytesIO()
        pixels = (np.clip(hot, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(buf, format="PNG")
        return buf.getvalue()

    # ---------------------------------------------------------------- sweeping

    @staticmethod
    def _idle_sweep() -> dict:
        return {
            "running": False,
            "done": 0,
            "total": 0,
            "found": None,
            "tiles": None,
            "out": None,
            "error": None,
            "finished": False,
            "min_score": 0.5,
            "highest": 0.0,
        }

    # held on the job for send_sweep, never sent to the page: `proposals` is the whole candidate
    # list and the page polls this twice a second
    SWEEP_STATUS_HIDDEN = ("proposals", "frames_dir")

    def sweep_status(self) -> dict:
        with self.lock:
            return {k: v for k, v in self.sweep_job.items() if k not in self.SWEEP_STATUS_HIDDEN}

    def sweep_recordings(self) -> dict:
        """which recordings a sweep may run over: every session that has frames. deliberately not
        only the ones drawn on - covering the rest is what a sweep is for."""
        root = paths.SESSIONS_DIR
        found = []
        for frames in sorted(root.glob("**/frames")) if root.is_dir() else []:
            count = len(frame_files(frames))
            if count:
                found.append({"name": str(frames.parent.relative_to(root)), "frames": count})
        return {"recordings": found}

    def start_sweep(self, recording: str, percent: int = 100, min_score: float = 0.5) -> dict:
        """run the trained model over a recording and write what it finds as a candidates file.

        THIS STAGE CLOSES THE LOOP: the model becomes proposals again, landing in the same select
        tab drawn tiles went through. it writes a NEW file, never over one - a candidates file has
        decisions keyed to it, and rewriting it re-points every recorded keep at a different box.
        """
        if self.trainer.weights is None:
            return {"error": "no trained or loaded weights to sweep with - train or load first"}
        frames_dir = frames_dir_of(paths.SESSIONS_DIR / recording)
        found = frame_files(frames_dir)
        if not found:
            return {"error": f"no frames under {frames_dir}"}
        with self.lock:
            if self.trainer.status().get("running") or self.sweep_job["running"]:
                return {"error": "a run is already going"}
            self.sweep_job = {**self._idle_sweep(), "running": True, "min_score": min_score}
        share = max(1, round(len(found) * min(100, max(1, percent)) / 100))
        if share < len(found):
            step = len(found) / share
            found = [found[int(i * step)] for i in range(share)]
        threading.Thread(
            target=self._run_sweep, args=(recording, frames_dir, found, min_score), daemon=True
        ).start()
        return {"ok": True}

    def _run_sweep(
        self, recording: str, frames_dir: Path, found: list[Path], min_score: float
    ) -> None:
        try:
            with self.lock:
                self.sweep_job["total"] = len(found)
            # SWEEP WIDE, FILTER AFTER: the sweep is the expensive half and the threshold is chosen
            # by looking at what came back, so keep everything down to the floor
            floor = min(min_score, paths.SWEEP_KEEP_FLOOR)
            candidates: list[dict] = []
            chunk = 8
            for start in range(0, len(found), chunk):
                candidates.extend(
                    c
                    for c in self.trainer.sweep(found[start : start + chunk])
                    if c.get("score", 0.0) >= floor
                )
                with self.lock:
                    self.sweep_job.update(
                        done=min(len(found), start + chunk),
                        found=len(candidates),
                        highest=round(max((c.get("score", 0.0) for c in candidates), default=0), 3),
                    )
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out = Path(self.state.bases["labels"]) / (
                f"{flat_recording(recording)}.cnn-{stamp}.candidates.jsonl"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(c) + "\n" for c in candidates))
            # the size comes from the training set's own boxes, or failing that from the proposals
            width, height = self.trained_box_size()
            if candidates and not self.trainer.training_set:
                width = int(statistics.median(c["width"] for c in candidates))
                height = int(statistics.median(c["height"] for c in candidates))
            # NOTHING IS CUT HERE: send_sweep decides how much of a sweep is worth looking at,
            # and it cannot decide what already landed in the pool
            with self.lock:
                self.sweep_job.update(
                    running=False,
                    finished=True,
                    found=len(candidates),
                    out=out.name,
                    proposals=candidates,
                    width=width,
                    height=height,
                    frames_dir=str(frames_dir),
                    tag=f"{flat_recording(recording)}_cnn-{stamp}",
                    above=counts_above(candidates),
                )
        except Exception as exc:
            # broad on purpose: this runs on a worker thread, and anything not caught here leaves
            # the job pinned at running=True with the page polling a dead run forever
            with self.lock:
                self.sweep_job.update(running=False, error=f"{type(exc).__name__}: {exc}")

    def send_sweep(self, min_score: float) -> dict:
        """cut the last sweep's proposals above this score into the pool. the sweep is untouched
        by the choice, so a threshold can be tried, looked at and tried again for the cost of
        the cut. THIS IS THE ONLY PLACE THE POOL IS TOUCHED by a sweep."""
        job = self.sweep_job
        proposals = job.get("proposals") or []
        if not proposals:
            return {"error": "no sweep to send - run one first"}
        keep = {i for i, c in enumerate(proposals) if c.get("score", 0) >= min_score}
        if not keep:
            return {
                "error": f"nothing at or above {min_score:.2f} - the best was {job.get('highest')}"
            }
        # by index into the WHOLE list: the file on disk holds every proposal, and the index in a
        # tile's name is what promotion resolves back to a box
        cut = self.state._cut_drawn(
            proposals,
            job["width"],
            job["height"],
            frames_dir=Path(job["frames_dir"]),
            tag=job["tag"],
            select=keep,
        )
        with self.lock:
            self.sweep_job["tiles"] = cut
        return {"sent": len(keep), "tiles": cut, "of": len(proposals), "tag": job["tag"]}
