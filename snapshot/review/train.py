"""training the nameplate cnn, sweeping a recording with it, and rendering what it saw.

Holds no paths of its own: everything it reads comes from the ReviewState it is given, or
from review.paths. THE MODEL LOCK IS THE LOAD-BEARING PART - torch's mps backend is not
thread safe and this drives it from a worker and from http request threads at once, so every
model call takes it, and the read side declines rather than blocking the page for minutes.
"""

from __future__ import annotations

import json
import shutil
import statistics
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np

from snapshot.review import paths, serverlog, splits
from snapshot.review.naming import _flat


def _counts_above(candidates: list[dict]) -> list[int]:
    """how many proposals sit at or above each 0.01 step, so the slider can say so without asking.

    101 integers rather than the scores themselves: it is exact at every position the slider can
    take, it is the same tiny payload whether a sweep found 12 proposals or 120,000, and it means
    moving the threshold costs no round trip. The histogram cannot answer this - its buckets are
    2.5% wide and the slider steps by 1%.
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


# train()'s own default, named here so the ui can show what "unset" means rather than an empty box
DEFAULT_RATE = 3e-4

# peaks kept per val frame. a synthetic frame carries about five plates, so this never trims a real one
HOLDOUT_PEAK_LIMIT = 50


class TrainingAborted(Exception):
    """raised from the progress callback - smolsmort only reports once per epoch, so that is when
    an abort lands"""

    def __init__(self, epoch: int):
        super().__init__(f"aborted at epoch {epoch}")
        self.epoch = epoch


class TrainState:
    """the cnn training run, as the web ui sees it.

    training happens on a BACKGROUND THREAD so the page stays responsive - a run is seconds at the
    current data size but will not stay that way, and a blocked server means no progress bar and no
    way to look at anything while it works.

    everything here is about ONE label: "a nameplate is centred here". see vision/plate_dataset for
    why the labels are treated as incomplete rather than as ground truth.
    """

    # ABSOLUTE, VIA paths. This was Path("parent/vision/plate_model.pt") - relative, so it only
    # resolved when the tool happened to be launched from the repo root, and silently wrote a
    # second model somewhere else when it was not. Tests still override it per instance.
    MODEL_PATH = paths.MODEL_PATH

    def __init__(self, state):
        self.state = state
        self.lock = threading.Lock()
        # TORCH'S MPS BACKEND IS NOT THREAD SAFE, and this process drives it from two threads: a
        # sweep or a training run on a worker, and heatmap overlays on whatever request thread the
        # http server hands them. Doing both at once segfaults inside a Metal tile - three hard
        # crashes on 2026-09-01, every one during a sweep, all with the same stack:
        #   at::native::mps::MetalShaderLibrary::exec_unary_kernel / relu_mps_ / SIGSEGV
        # A segfault cannot be caught in Python, so the only fix is to never overlap. EVERY call
        # into the model takes this.
        self.model_lock = threading.Lock()
        self.job = {
            "running": False,
            "epoch": 0,
            "epochs": 0,
            "loss": None,
            "history": [],
            "error": None,
            "finished": False,
            "separation": None,
            "holdout": None,
            "aborted": None,
        }
        self.examples: list = []
        self.holdout: list = []
        self.model = None
        # which checkpoint is in play, or None when the model came from a training run. the two
        # pickers name mutually exclusive things - a set says what to train on, weights say what is
        # already trained - so one of them being set has to clear the other
        self.loaded_weights = None
        # the training thread, so a stuck "running" flag can be checked against whether anything is
        # actually running rather than trusted on its own
        self._worker = None
        # set by the abort button, read at the next epoch boundary, cleared by every start
        self._abort = threading.Event()
        # PICKING A SET USED TO BE COSMETIC. the name lived in the page and nothing sent it here,
        # so the trainer went on building examples from whatever candidates queue happened to be
        # bound - unreviewed boxes, which is exactly what the picker exists to prevent
        self.training_set: str | None = None
        self.classes: dict[str, int] = {}
        self.sweep_job = {
            "running": False,
            "done": 0,
            "total": 0,
            "found": None,
            "tiles": None,
            "out": None,
            "error": None,
            "finished": False,
            # carried so an empty sweep can say how far under the floor it was, rather than
            # reporting "nothing above NaN" - these were missing and the page read undefined
            "min_score": 0.5,
            "highest": 0.0,
            "histogram": None,
            # why nothing is moving, when something is in the way. cleared on every start, or a
            # note from the last run reads as the state of this one
            "note": None,
        }

    # ---- data -------------------------------------------------------------------------
    def dataset(self) -> list:
        """examples for the CURRENTLY BOUND dataset, rebuilt each time - decisions change under us
        as the user reviews, and a cached example list would train on yesterday's labels"""
        from smolsmort.detect.dataset import build_from, build_training_set

        if self.training_set:
            path = paths.DATASETS_DIR / f"{self.training_set}.jsonl"
            examples, self.classes = build_training_set(path, paths.SESSIONS_DIR)
            rows = self._rows(path)
            # TRAIN ON THE TRAIN SPLIT ONLY. a set written before splits existed reads as all
            # train, so the older sets load exactly as they did
            self.examples = splits.keep(examples, rows, paths.SESSIONS_DIR, "train")
            # val is scored after every run; test stays untouched until a model is chosen
            self.holdout = splits.keep(examples, rows, paths.SESSIONS_DIR, "val")
            return self.examples

        self.classes = {}
        self.holdout = []
        self.examples = build_from(
            self.state.candidates,
            self.state.decisions,
            self.state.frames_dir,
            uniform_width=self.state.uniform_width,
            uniform_height=self.state.uniform_height,
        )
        return self.examples

    @staticmethod
    def _rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _split_frames(self) -> dict[str, int]:
        """frames per split in the open set - the holdouts are not in `self.examples`, so the
        tab would otherwise have no way to show they exist"""
        path = paths.DATASETS_DIR / f"{self.training_set}.jsonl"
        counts = dict.fromkeys(splits.SPLITS, 0)
        for split in splits.of_rows(self._rows(path)).values():
            counts[split] += 1
        return counts

    def info(self) -> dict:
        """what the OPEN TRAINING SET holds, and nothing else.

        it used to describe the bound candidates QUEUE - tiles staged, regions mined but never
        confirmed - which is the world before training took a promoted set. With a set open it
        reported "bound dataset (none)" and "0 of 0 frames" while 460 rows sat loaded above it.
        Wrong, and also the wrong question: the pool's contents belong to discard / promote.

        THE ONE THING THAT DECIDES A RUN at this data size is how thin the rarest class is - a
        15-channel head over a handful of examples each is the real limit, not frame count.
        """
        if not self.training_set:
            return {
                "set": None,
                "device": self._device(),
                "model_exists": self.MODEL_PATH.is_file(),
            }
        try:
            examples = self.dataset()
        except Exception as exc:  # a missing frames dir or set file is a normal state here
            return {"error": str(exc), "set": self.training_set, "device": self._device()}

        per_class: dict[str, int] = {}
        for example in examples:
            for label in example.labels:
                if label:
                    per_class[label] = per_class.get(label, 0) + 1
        thinnest = min(per_class.items(), key=lambda kv: kv[1]) if per_class else None
        return {
            "set": self.training_set,
            "frames": len(examples),
            "plates": sum(len(e.centres) for e in examples),
            "negatives": sum(len(e.negatives) for e in examples),
            "split": self._split_frames(),
            "classes": len(per_class),
            "thinnest": {"label": thinnest[0], "count": thinnest[1]} if thinnest else None,
            "device": self._device(),
            "model_exists": self.MODEL_PATH.is_file(),
        }

    def _device(self) -> str:
        try:
            import torch

            return "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            return "torch not installed"

    def frames(self) -> list[dict]:
        """every frame that carries at least one confirmed plate, newest scoring first once a
        model exists - this is the list the ui renders down the right-hand side"""
        examples = self.examples or self.dataset()
        rows = []
        for i, example in enumerate(examples):
            if not example.object_count:
                continue
            rows.append(
                {
                    "index": i,
                    "name": example.path.name,
                    "plates": example.object_count,
                    "ignored": len(example.ignore),
                }
            )
        return rows

    # ---- training ---------------------------------------------------------------------
    def class_names(self) -> dict:
        """the channel order of the bound training set, for the picker over the heatmap.

        DERIVED, never a fixed list of ten: plate_dataset builds the channel map from the labels
        PRESENT, sorted, so a set with three labels has three channels, and a hardcoded list would
        point the picker at channels the model does not have.
        """
        if not self.classes and self.training_set:
            self.dataset()
        pairs = sorted(self.classes.items(), key=lambda kv: kv[1])
        return {
            "set": self.training_set,
            "weights": self.loaded_weights,
            "classes": [name for name, _ in pairs],
        }

    def bind(self, name: str) -> dict:
        """which promoted set the next run trains on. picking one used to only change a label.

        BINDING A SET DROPS THE LOADED MODEL. a checkpoint carries its own channel map, so keeping
        one alive against a newly bound set leaves the heatmap picker offering that model's classes
        for this set's data - both individually valid, and the overlay would read every channel
        under the wrong name with nothing to object.
        """
        self.training_set = name or None
        self.examples, self.classes = [], {}
        self.model = None
        self.loaded_weights = None
        return self.class_names()

    def _trained_box_size(self) -> tuple[int, int]:
        """the median box size of the bound training set, falling back to the review default.

        a heatmap peak says WHERE a plate is, never how big - the size is a constant of the screen
        (see plate_dataset), so it has to be carried in from the data the model was trained on.
        """
        if self.training_set:
            path = paths.DATASETS_DIR / f"{self.training_set}.jsonl"
            if path.is_file():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                sizes = [(r["width"], r["height"]) for r in rows if not r.get("negative")]
                if sizes:
                    return (
                        int(statistics.median(w for w, _ in sizes)),
                        int(statistics.median(h for _, h in sizes)),
                    )
        return self.state.uniform_width, self.state.uniform_height

    def _checkpoint_path(self, relative: str) -> Path | None:
        """a path inside the checkpoints folder, or None when it would leave it"""
        root = paths.CHECKPOINTS_DIR.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return None
        return target

    def checkpoint_folders(self, under: str = "") -> dict:
        """the folders a checkpoint can go into, one level at a time, with a suggested name.

        confined to the checkpoints folder: the weights picker lists only what is under it, so a
        checkpoint saved anywhere else could never be loaded back from the page
        """
        root = paths.CHECKPOINTS_DIR
        root.mkdir(parents=True, exist_ok=True)
        here = self._checkpoint_path(under or "")
        if here is None or not here.is_dir():
            here = root.resolve()
        relative = here.relative_to(root.resolve()).as_posix()
        relative = "" if relative == "." else relative
        parent = None
        if relative:
            parent = Path(relative).parent.as_posix()
            parent = "" if parent == "." else parent
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return {
            "here": relative,
            "parent": parent,
            "folders": sorted(p.name for p in here.iterdir() if p.is_dir()),
            "name": f"{self.training_set or 'unbound'}-{stamp}",
        }

    def _checkpoint_target(self, name: str | None, folder: str = "") -> tuple[Path, str] | dict:
        """(weights path, its name under the checkpoints folder) for a typed name, or an error.

        shared by the manual save and by a training run that names its own weights, so both refuse
        the same things: a folder outside the checkpoints folder, and a name already taken
        """
        root = paths.CHECKPOINTS_DIR
        checkpoints = self._checkpoint_path(folder or "")
        if checkpoints is None:
            return {"error": "that folder is outside the checkpoints folder"}
        checkpoints.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # rebuilt from an allowlist, so a typed name cannot carry a path or a dot that would
        # confuse the class map's own file name beside it
        raw = (name or "").strip().removesuffix(".pt")
        typed = "".join(c for c in raw if c.isalnum() or c in "-_")
        stem = typed or f"{self.training_set or 'unbound'}-{stamp}"
        weights = checkpoints / f"{stem}.pt"
        relative = weights.relative_to(root.resolve()).as_posix()
        if weights.exists():
            return {"error": f"{relative} already exists - pick another name"}
        return weights, relative

    def _write_class_map(self, weights: Path) -> None:
        """the class map beside a checkpoint - see save_checkpoint for why it is not optional"""
        weights.with_suffix(".classes.json").write_text(
            json.dumps(
                {
                    "training_set": self.training_set,
                    "classes": sorted(self.classes, key=self.classes.get),
                    "separation": self.job.get("separation"),
                    "holdout": self.job.get("holdout"),
                    "saved": datetime.now().strftime("%Y%m%d-%H%M%S"),
                },
                indent=2,
            )
        )

    def save_checkpoint(self, name: str | None = None, folder: str = "") -> dict:
        """snapshot the trained model under a name, beside its class map.

        A CHECKPOINT IS ONLY MEANINGFUL WITH THE MAP IT WAS TRAINED UNDER - the channel order comes
        from the labels present in a training set, so loading weights against a different map
        silently renames every class. plate_train.load takes that map as an argument for exactly
        this reason, so the map is written next to the weights rather than left to be guessed.

        plate_model.pt is overwritten by every run that does not name its weights; these are not.
        the name and the folder are picked in the page, both defaulting to what was used before
        """
        if not self.MODEL_PATH.is_file():
            return {"error": "nothing trained yet - train a model first"}
        target = self._checkpoint_target(name, folder)
        if isinstance(target, dict):
            return target
        weights, relative = target
        shutil.copy2(self.MODEL_PATH, weights)
        self._write_class_map(weights)
        return {"name": relative, "classes": len(self.classes), "bytes": weights.stat().st_size}

    def peak_distribution(self, frames: int = 8) -> dict:
        """where the model's response falls, split by whether it sits on a confirmed plate.

        THIS IS WHAT A THRESHOLD IS CHOSEN AGAINST. a bare histogram of heights says nothing about
        whether a cut is good; two populations do - response on a plate should sit above the cut
        and response on grass below it, and the gap between them is the thing worth looking at.

        SAMPLED, NOT PEAK-DECODED. the first version ran decode_peaks over every channel of every
        frame at a low floor and took minutes - it was finding tens of thousands of peaks in an
        undertrained model's noise. reading the heatmap directly answers the same question for the
        cost of array indexing.

        NOT A HOLDOUT, and must not be shown as one: there is no train/test split, so these are
        frames the model was fitted on and the separation here is the optimistic case.
        """

        from smolsmort.detect.model import STRIDE, downscale_of
        from smolsmort.detect.train import SCORE_BUCKETS, heatmaps_for

        model = self.model
        if model is None and self.MODEL_PATH.is_file():
            from smolsmort.detect.train import load

            model = self.model = load(self.MODEL_PATH)
        if model is None:
            return {"error": "no trained model yet"}
        examples = self.examples or self.dataset()
        examples = [e for e in examples if e.centres][:frames]
        if not examples:
            return {"error": "no frames with confirmed plates"}

        scale = STRIDE * downscale_of(model)
        width, height = self._trained_box_size()
        on_plate, elsewhere = [], []
        # NON-BLOCKING, like the overlay. a sweep or a training run holds the model for minutes,
        # and a page request that waits behind one occupies a browser connection the whole time -
        # six of those and every poll on the page stalls, which reads as "the screen froze"
        if not self.model_lock.acquire(blocking=False):
            return {"error": "the model is busy - try again when the run finishes"}
        try:
            for example in examples:
                hot = heatmaps_for(model, example.path).max(axis=0)
                rows, cols = hot.shape
                mask = np.zeros(hot.shape, dtype=bool)
                for cx, cy in example.centres:
                    r, c = int(cy / scale), int(cx / scale)
                    dr, dc = max(1, int(height / scale / 2)), max(1, int(width / scale / 2))
                    lo_r, hi_r = max(0, r - dr), min(rows, r + dr + 1)
                    lo_c, hi_c = max(0, c - dc), min(cols, c + dc + 1)
                    if hi_r > lo_r and hi_c > lo_c:
                        on_plate.append(float(hot[lo_r:hi_r, lo_c:hi_c].max()))
                        mask[lo_r:hi_r, lo_c:hi_c] = True
                elsewhere.extend(hot[~mask].ravel().tolist())
        finally:
            self.model_lock.release()

        def histogram(values):
            counts, _ = np.histogram(values, bins=SCORE_BUCKETS, range=(0.0, 1.0))
            return [int(n) for n in counts]

        return {
            "bins": SCORE_BUCKETS,
            "on_plate": histogram(on_plate),
            "elsewhere": histogram(elsewhere),
            "counts": {"on_plate": len(on_plate), "elsewhere": len(elsewhere)},
            "frames": len(examples),
            "highest": round(float(max(on_plate, default=0.0)), 3),
        }

    def saved_checkpoints(self) -> dict:
        """the named checkpoints save_checkpoint wrote, newest first, subfolders included"""
        # beside the model it snapshots, wherever that has been pointed
        checkpoints = paths.CHECKPOINTS_DIR
        found = []
        saved = checkpoints.rglob("*.pt") if checkpoints.is_dir() else []
        for path in sorted(saved, key=lambda p: p.stat().st_mtime, reverse=True):
            meta = path.with_suffix("").with_suffix(".classes.json")
            classes = []
            if meta.is_file():
                try:
                    classes = json.loads(meta.read_text()).get("classes", [])
                except (OSError, json.JSONDecodeError):
                    classes = []
            found.append(
                {
                    "name": path.relative_to(checkpoints).as_posix(),
                    "classes": classes,
                    "kb": round(path.stat().st_size / 1024),
                }
            )
        return {"weights": found}

    def load_checkpoint(self, name: str) -> dict:
        """load a checkpoint AND the class map it was trained under.

        THE MAP IS NOT OPTIONAL. channel order comes from whichever labels a training set held, so
        weights loaded against a different map silently rename every class - a heatmap for
        "hostile (target)" would be read as something else with no error anywhere. save_checkpoint
        writes the map beside the file for exactly this, and this refuses without it.
        """
        from smolsmort.detect.train import load

        path = self._checkpoint_path(name)
        if path is None or not path.is_file():
            return {"error": f"no weights named {name!r}"}
        meta = path.with_suffix("").with_suffix(".classes.json")
        if not meta.is_file():
            return {"error": f"{name} has no class map beside it - it cannot be read safely"}
        try:
            saved = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {"error": f"{meta.name} is unreadable: {exc}"}
        names = saved.get("classes") or []
        if not names:
            return {"error": f"{meta.name} names no classes"}

        self.model = load(path)
        self.classes = {name_: index for index, name_ in enumerate(names)}
        self.loaded_weights = name
        # the checkpoint's own set, or none - never the set that happened to be bound before, which
        # would name a set these weights were not trained on
        self.training_set = saved.get("training_set")
        shutil.copy2(path, self.MODEL_PATH)
        return {"name": name, "classes": names, "set": self.training_set}

    # held on the job for send_sweep, never sent to the page: `proposals` is the whole candidate
    # list and the page polls this twice a second
    SWEEP_STATUS_HIDDEN = ("proposals", "frames_dir")

    def sweep_status(self) -> dict:
        """what the page needs to render a sweep, without the payload it does not."""
        with self.lock:
            return {k: v for k, v in self.sweep_job.items() if k not in self.SWEEP_STATUS_HIDDEN}

    def send_sweep(self, min_score: float) -> dict:
        """cut the last sweep's proposals above this score into the pool.

        the sweep itself is untouched by the choice, so a threshold can be tried, looked at, and
        tried again for the cost of the cut rather than another full run.

        THIS IS THE ONLY PLACE THE POOL IS TOUCHED. the sweep used to cut every proposal it kept
        the moment it finished - everything down to SWEEP_KEEP_FLOOR, 0.05 - so a run landed 4105
        tiles in discard / promote before the threshold had been chosen at all, and the slider
        then filtered a cut that had already happened. Balthazar Fitzpatrick: "It also doesn't take the cut of the
        sweep when adding back to discard / promote, but takes everything."
        """
        job = self.sweep_job
        proposals = job.get("proposals") or []
        if not proposals:
            return {"error": "no sweep to send - run one first"}
        keep = {i for i, c in enumerate(proposals) if c.get("score", 0) >= min_score}
        if not keep:
            return {
                "error": f"nothing at or above {min_score:.2f} - the best was {job.get('highest')}"
            }
        # BY INDEX INTO THE WHOLE LIST, not the filtered one - see _cut_drawn's `select`. the
        # candidates file on disk holds every proposal, and the index in a tile's name is what
        # promote resolves back to a box
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

    def sweep_recordings(self) -> dict:
        """which recordings a sweep may run over: every session that has frames.

        DELIBERATELY NOT ONLY THE ONES DRAWN ON. running over recordings nobody drew on is the
        entire point - it is what turns one afternoon of drawing into coverage of everything else.
        """
        root = paths.SESSIONS_DIR
        found = []
        for frames in sorted(root.glob("**/frames")) if root.is_dir() else []:
            if not frames.is_dir():
                continue
            count = len(list(frames.glob("*.jpg"))) + len(list(frames.glob("*.png")))
            if count:
                found.append({"name": str(frames.parent.relative_to(root)), "frames": count})
        return {"recordings": found}

    def start_sweep(self, recording: str, percent: int = 100, min_score: float = 0.5) -> dict:
        """run the trained model over a recording and write what it finds as a candidates file.

        THIS IS THE STAGE THAT CLOSES THE LOOP. everything before it turns Balthazar Fitzpatrick's rectangles into
        a model; this turns the model back into proposals, which land in the same discard / promote
        tab the drawn tiles went through. Balthazar Fitzpatrick: "the cnn adds new candidate datasets, that I can
        then in turn open in Keep Assign / Discard."

        WRITES A NEW FILE, NEVER OVER ONE. a candidates file has decisions keyed to it, and
        rewriting it re-points every recorded keep at a different box.
        """
        with self.lock:
            if self.job["running"] or self.sweep_job["running"]:
                return {"error": "a run is already going"}
            self.sweep_job = {
                "running": True,
                "done": 0,
                "total": 0,
                "found": None,
                "tiles": None,
                "out": None,
                "error": None,
                "finished": False,
                "min_score": min_score,
                "highest": 0.0,
                "histogram": None,
                "note": None,
            }

        def run() -> None:
            serverlog.LOG.info("sweep start %s peak_rss=%.0fMB", recording, serverlog.peak_rss_mb())
            try:
                from smolsmort.detect.train import load, sweep

                model = self.model
                if model is None:
                    if not self.MODEL_PATH.is_file():
                        raise RuntimeError("no trained model yet - train one on this tab first")
                    model = self.model = load(self.MODEL_PATH)
                if not self.classes:
                    self.dataset()
                if not self.classes:
                    raise RuntimeError("open a training set first - the sweep needs its class map")

                frames_dir = paths.SESSIONS_DIR / recording / "frames"
                found = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
                if not found:
                    raise FileNotFoundError(f"no frames under {frames_dir}")
                share = max(1, round(len(found) * min(100, max(1, percent)) / 100))
                if share < len(found):
                    step = len(found) / share
                    found = [found[int(i * step)] for i in range(share)]
                with self.lock:
                    self.sweep_job["total"] = len(found)

                # sweep reports (position, total, found so far) - all three, so the bar can show
                # proposals accumulating rather than only frames consumed
                def progress(
                    done: int,
                    total: int,
                    so_far: int,
                    highest: float = 0.0,
                    histogram=None,
                ) -> None:
                    with self.lock:
                        self.sweep_job["done"] = done
                        self.sweep_job["total"] = total
                        self.sweep_job["found"] = so_far
                        # A SWEEP THAT FINDS NOTHING MUST SAY WHY. min_score used to be a constant
                        # 0.5 with no control, and a model whose peaks top out at 0.33 wrote three
                        # empty candidates files and reported success. carrying the highest peak
                        # seen turns that into "0 proposals, best peak 0.33 against a floor of 0.5"
                        self.sweep_job["highest"] = round(highest, 3)
                        if histogram is not None:
                            self.sweep_job["histogram"] = [int(n) for n in histogram]

                # THE SIZE COMES FROM THE TRAINING SET, not from whatever the review state happens
                # to hold. state.uniform_* is a default sized for the OLD template library (64x10),
                # so a sweep took it and emitted proposals a quarter the size of a real plate -
                # every tile cut from one would have been wrong. the set's own rows carry the
                # fitted size the boxes were drawn at, which is the only right answer here
                width, height = self._trained_box_size()
                # SWEEP WIDE, FILTER AFTER. the sweep is the expensive half - minutes of forward
                # passes - and the threshold is a decision made by LOOKING at what came back. tying
                # them together meant every adjustment re-ran the whole thing, and the distribution
                # the slider draws could only ever describe a run already spent
                floor = min(min_score, paths.SWEEP_KEEP_FLOOR)
                # SAY SO WHILE WAITING. only one thing may touch the model at a time, so a sweep
                # started during a training run blocks here - and a blocked sweep reporting "0 / N"
                # for minutes is indistinguishable from a hung one. name the reason instead.
                if not self.model_lock.acquire(blocking=False):
                    with self.lock:
                        self.sweep_job["note"] = "waiting for the training run to finish"
                    self.model_lock.acquire()
                    with self.lock:
                        self.sweep_job["note"] = None
                try:
                    candidates = sweep(
                        model,
                        self.classes,
                        found,
                        width=width,
                        height=height,
                        min_score=floor,
                        on_progress=progress,
                    )
                finally:
                    self.model_lock.release()
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                out = Path(self.state.bases["labels"]) / (
                    f"{_flat(recording)}.cnn-{stamp}.candidates.jsonl"
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("".join(json.dumps(c) + "\n" for c in candidates))

                # NOTHING IS CUT HERE. this used to cut every candidate into the pool the instant
                # the sweep finished, which is what put 4105 tiles in discard / promote before
                # the threshold slider had been touched - the send below is the step that decides
                # how much of a sweep is worth looking at, and it cannot decide what already landed
                # EVERYTHING send_sweep NEEDS, kept on the job. none of proposals/width/height/
                # frames_dir/tag was ever stored, so "send above the threshold" read an empty list
                # and answered "no sweep to send - run one first" after a sweep that had just found
                # 1272 proposals. The candidates only ever lived in this closure.
                with self.lock:
                    self.sweep_job.update(
                        running=False,
                        finished=True,
                        found=len(candidates),
                        tiles=None,
                        out=out.name,
                        proposals=candidates,
                        width=width,
                        height=height,
                        frames_dir=str(frames_dir),
                        tag=f"{_flat(recording)}_cnn-{stamp}",
                        above=_counts_above(candidates),
                    )
                serverlog.LOG.info(
                    "sweep finished %d proposals peak_rss=%.0fMB",
                    len(candidates),
                    serverlog.peak_rss_mb(),
                )
            except Exception as exc:  # noqa: BLE001
                # deliberately broad: this runs on a worker thread, so anything not caught here
                # leaves sweep_job pinned at running=True and the page polling a dead run forever
                serverlog.LOG.exception("sweep failed peak_rss=%.0fMB", serverlog.peak_rss_mb())
                with self.lock:
                    self.sweep_job.update(running=False, error=f"{type(exc).__name__}: {exc}")

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    def window_floor(self) -> dict:
        """the smallest training window this set's boxes fit inside, whole.

        REPORTED, NOT JUST ENFORCED: "283" means nothing without the box that produced it, and the
        whole failure mode here is a number that looks reasonable and quietly clips a plate.
        """
        from smolsmort.detect.model import DEFAULT_DOWNSCALE
        from smolsmort.detect.train import CROP, minimum_window, snapped_window

        width, height = self._trained_box_size()
        # SNAPPED HERE TOO, so the number shown is the number trained at. train() snaps anyway, but
        # reporting an unsnapped 283 while training at 284 is the same silent mismatch one layer up
        floor = snapped_window(minimum_window(width, height, downscale=DEFAULT_DOWNSCALE))
        # `downscale` so the page can say what a window is in capture pixels without knowing it
        return {
            "floor": floor,
            "box": [width, height],
            "default": max(CROP, floor),
            "downscale": DEFAULT_DOWNSCALE,
        }

    def start(
        self,
        epochs: int,
        batch: int,
        crop: int | None = None,
        learning_rate: float | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> dict:
        # A WINDOW BELOW THE FLOOR TEACHES HALF-PLATES. _crop_window offsets by up to JITTER_FRACTION
        # of the window, so a box wider than the remaining margin is clipped at the extremes - which
        # is the one thing the jitter was tuned to avoid. Refuse rather than train something subtly
        # wrong, and name both numbers so the refusal is actionable.
        limits = self.window_floor()
        if crop is not None and int(crop) < limits["floor"]:
            box = limits["box"]
            return {
                "error": (
                    f"a {box[0]}x{box[1]} box needs a window of at least {limits['floor']}; "
                    f"{int(crop)} would clip it at the jitter extremes and teach a half-plate "
                    "as a whole one"
                )
            }
        # A NAMED RUN WRITES ITS OWN WEIGHTS AND LEAVES plate_model.pt ALONE. refused up front when the
        # name is taken, so a long run is never wasted on a name it cannot be saved under
        target = None
        if name is not None and name.strip():
            target = self._checkpoint_target(name)
            if isinstance(target, dict):
                return target
        with self.lock:
            if self.job["running"] and self._worker is not None and self._worker.is_alive():
                return {"error": "a training run is already going"}
            self._abort.clear()
            self.job = {
                "running": True,
                "epoch": 0,
                "epochs": epochs,
                "loss": None,
                "history": [],
                "error": None,
                "finished": False,
                "separation": None,
                "holdout": None,
                "aborted": None,
                # WHAT THIS RUN USED, reported back. a separation figure without the settings that
                # produced it cannot be repeated or compared against the next attempt
                "learning_rate": DEFAULT_RATE if learning_rate is None else float(learning_rate),
                "seed": 0 if seed is None else int(seed),
                "window": limits["default"] if crop is None else int(crop),
                "batch": batch,
                "weights": target[1] if target else None,
            }
        window = limits["default"] if crop is None else int(crop)
        # THE TWO KNOBS train() ALWAYS HAD AND NOTHING COULD REACH. learning_rate and seed were
        # arguments with defaults, so every run this tool ever started used 3e-4 and seed 0 - which
        # means a run that diverged could not be retried differently without editing the library.
        rate = DEFAULT_RATE if learning_rate is None else float(learning_rate)
        run_seed = 0 if seed is None else int(seed)
        self._worker = threading.Thread(
            target=self._run,
            args=(epochs, batch, window, rate, run_seed, target[0] if target else None),
            daemon=True,
        )
        self._worker.start()
        return {"ok": True}

    def abort(self) -> dict:
        """stop the running training at its next epoch boundary. nothing is saved, so the model
        already installed stays the one in use"""
        with self.lock:
            running = self.job["running"] and self._worker is not None and self._worker.is_alive()
        if not running:
            return {"error": "no training run is going"}
        self._abort.set()
        return {"ok": True}

    def _run(
        self,
        epochs: int,
        batch: int,
        window: int,
        rate: float,
        seed: int,
        weights: Path | None = None,
    ) -> None:
        # peak memory at every milestone: an oom kill leaves no traceback, only the last line here
        log = serverlog.LOG
        log.info(
            "train start epochs=%d batch=%d window=%d peak_rss=%.0fMB",
            epochs,
            batch,
            window,
            serverlog.peak_rss_mb(),
        )
        try:
            from smolsmort.detect.train import save, train

            examples = self.dataset()
            log.info(
                "train dataset %d examples peak_rss=%.0fMB", len(examples), serverlog.peak_rss_mb()
            )

            def progress(p):
                log.info(
                    "train epoch %d loss %.4f peak_rss=%.0fMB",
                    p.epoch,
                    p.loss,
                    serverlog.peak_rss_mb(),
                )
                with self.lock:
                    self.job["epoch"] = p.epoch
                    self.job["loss"] = p.loss
                    self.job["history"].append(p.loss)
                if self._abort.is_set():
                    raise TrainingAborted(p.epoch)

            with self.model_lock:
                model, _ = train(
                    examples,
                    epochs=epochs,
                    batch=batch,
                    crop=window,
                    learning_rate=rate,
                    seed=seed,
                    on_progress=progress,
                    classes=self.classes or None,
                )
            save(model, weights or self.MODEL_PATH)
            self.model = model
            separation = self._separation(model, examples)
            holdout = self._holdout_score(model)
            with self.lock:
                self.job["running"] = False
                self.job["finished"] = True
                self.job["separation"] = separation
                self.job["holdout"] = holdout
            if weights is not None:
                # after the scores, so the class map carries them like a manual save's does
                self._write_class_map(weights)
                self.loaded_weights = weights.relative_to(
                    paths.CHECKPOINTS_DIR.resolve()
                ).as_posix()
            log.info("train finished peak_rss=%.0fMB", serverlog.peak_rss_mb())
        except TrainingAborted as stop:
            with self.lock:
                self.job["running"] = False
                self.job["aborted"] = stop.epoch
        except Exception as exc:
            log.exception("train failed peak_rss=%.0fMB", serverlog.peak_rss_mb())
            with self.lock:
                self.job["running"] = False
                self.job["error"] = f"{type(exc).__name__}: {exc}"

    def _separation(self, model, examples) -> dict | None:
        """the only honest accuracy figure at this data size: how much brighter the heatmap is at a
        confirmed plate than it is over the frame generally. a precision/recall number would imply
        the labels are complete, and they are not."""

        from smolsmort.detect.model import STRIDE, downscale_of
        from smolsmort.detect.train import heatmap_for

        scale = STRIDE * downscale_of(model)
        at_plates, backgrounds = [], []
        for example in [e for e in examples if e.object_count][:8]:
            with self.model_lock:
                heat = heatmap_for(model, example.path)
            backgrounds.append(float(np.percentile(heat, 99)))
            for cx, cy in example.centres:
                row, col = int(cy / scale), int(cx / scale)
                if 0 <= row < heat.shape[0] and 0 <= col < heat.shape[1]:
                    at_plates.append(
                        float(heat[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2].max())
                    )
        if not at_plates:
            return None
        plate = float(np.median(at_plates))
        background = float(np.median(backgrounds))
        return {
            "plate": plate,
            "background": background,
            "ratio": plate / background if background else None,
            "samples": len(at_plates),
        }

    def _holdout_score(self, model) -> dict | None:
        """recall and false-positive rate on the VAL split, through smolsmort's own scorer.

        frames the model never trained on, so this outranks separation. still the same recording
        as train though - a held-out zone is the stricter test, and this is not that
        """
        if not self.holdout:
            return None
        from smolsmort.detect import scoring
        from smolsmort.detect.box import Box
        from smolsmort.detect.model import PEAK_MIN_SCORE, decode_peaks, downscale_of
        from smolsmort.detect.train import heatmaps_for

        width, height = self._trained_box_size()
        downscale = downscale_of(model)

        def box_at(x: float, y: float) -> Box:
            return Box(int(x - width // 2), int(y - height // 2), width, height, origin="model")

        predictions, truths = [], []
        for example in self.holdout:
            with self.model_lock:
                hot = heatmaps_for(model, example.path).max(axis=0)
            # capped, since an undertrained model is warm nearly everywhere [see decode_peaks]
            peaks = decode_peaks(hot, limit=HOLDOUT_PEAK_LIMIT, downscale=downscale)
            predictions.append([box_at(p.x, p.y) for p in peaks])
            truths.append([box_at(x, y) for x, y in example.centres])
        result = scoring.score(predictions, truths)
        return {
            "frames": len(truths),
            "with_plate": result.frames_with_object,
            "without_plate": result.frames_without_object,
            "recall": result.recall,
            "false_positive_rate": result.false_positive_rate,
            "spurious": result.spurious,
            "min_score": PEAK_MIN_SCORE,
            "passes": result.passes,
            "beats_teacher": result.beats_the_teacher(),
        }

    def status(self) -> dict:
        with self.lock:
            return dict(self.job)

    # ---- rendering --------------------------------------------------------------------
    def overlay_bytes(self, index: int, channel: int = -1) -> bytes | None:
        """the frame with the model's heatmap burned over it, plus the confirmed plates boxed.

        channel picks WHICH class to burn in; -1 (the default) shows the loudest across all of them.

        thresholded rather than normalised to the maximum: normalising made the whole frame glow
        because the background sits around a fifth of the peak, which looks like the model fires
        everywhere when it does not.
        """
        from PIL import Image, ImageDraw

        examples = self.examples or self.dataset()
        if index < 0 or index >= len(examples):
            return None
        example = examples[index]
        frame = Image.open(example.path).convert("RGB")

        model = self.model
        if model is None and self.MODEL_PATH.is_file():
            from smolsmort.detect.train import load

            model = self.model = load(self.MODEL_PATH)

        canvas = np.asarray(frame).astype(np.float32)
        if model is not None:
            from smolsmort.detect.train import heatmap_for

            # NON-BLOCKING. a sweep holds the model for minutes, and waiting behind it would
            # freeze the page - so an overlay asked for mid-sweep declines instead, and the tab
            # keeps showing the picture it already has
            if not self.model_lock.acquire(blocking=False):
                return None
            try:
                heat = heatmap_for(model, example.path)
                if self.classes:
                    from smolsmort.detect.train import heatmaps_for

                    stack = heatmaps_for(model, example.path)
            finally:
                self.model_lock.release()
            if self.classes:
                # ONE CHANNEL, OR THE LOUDEST OF THEM. the channels are independent sigmoids, not a
                # softmax, so max across them is the honest "something fires here" view - summing
                # would make ten quiet channels outshout one confident one
                heat = stack[channel] if 0 <= channel < len(stack) else stack.max(axis=0)
            floor = float(np.percentile(heat, 99))
            span = max(heat.max() - floor, 1e-6)
            shown = np.clip((heat - floor) / span, 0, 1)
            up = (
                np.asarray(
                    Image.fromarray((shown * 255).astype(np.uint8)).resize(
                        frame.size, Image.BILINEAR
                    )
                ).astype(np.float32)
                / 255.0
            )
            canvas[..., 0] = np.clip(canvas[..., 0] + up * 220, 0, 255)
            canvas[..., 1] *= 1 - up * 0.55
            canvas[..., 2] *= 1 - up * 0.55

        image = Image.fromarray(canvas.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        # THE SET'S OWN FITTED SIZE, not a constant. this was hardcoded at 132x24 - a leftover
        # from the old template library - so every confirmed plate was boxed at about half the
        # width and two thirds the height of the thing it was marking. these boxes are ground
        # truth, drawn to judge the heatmap against, so drawing them wrong misjudges the model
        width, height = self._trained_box_size()
        half_w, half_h = width / 2, height / 2
        for cx, cy in example.centres:
            draw.rectangle(
                [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
                outline=(90, 255, 120),
                width=3,
            )
        image = image.resize((frame.width // 2, frame.height // 2), Image.BILINEAR)
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
