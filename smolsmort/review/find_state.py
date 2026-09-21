"""the find half of the review state: the bound recording, boxes drawn on it, and the tiles cut.

a mixin over `ReviewState` (state.py), which owns the attributes it reads (`bases`, `pools`,
`lock`, `renderer`, `guesser`). no http and no rendering decisions live here - routes.py calls in.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime
from pathlib import Path

from smolsmort.review import paths
from smolsmort.review.naming import _flat, _tile_tag
from smolsmort.review.recordings import (
    flat_recording,
    frame_files,
    frames_dir_of,
    recording_name,
    session_frames_dirs,
)
from smolsmort.review.render import read_frame
from smolsmort.review.tiles import TileError, cut_tile, save_tile

CROP_MODES = ("percent", "absolute")
ASPECTS = ("fixed", "free")
# the largest crop edge or box a client may ask for; a typo should not cut a gigapixel tile
MAX_CROP_PX = 4096


def write_json_atomic(path: Path, value, *, indent: int | None = None, **dump) -> None:
    """write json so a reader never sees half a file: temp beside it, then replace"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=indent, **dump))
    os.replace(temp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FindMixin:
    # ---------------------------------------------------------------- binding

    def rebind(
        self, frames_dir: Path, candidates: list[dict], decisions_path: Path, session_tag: str
    ) -> None:
        """swap the active dataset in place, no restart. the pool is shared by every dataset, so
        it stays exactly as it was."""
        with self.lock:
            self.frames_dir = frames_dir
            self.candidates = candidates
            self.decisions_path = decisions_path
            self.session_tag = session_tag
            self.decisions = self._load_decisions()

    def unbind(self) -> dict:
        """let go of the bound dataset. NOTHING ON DISK MOVES.

        closing is not a save and not a delete - only "stop showing me this one". drawn boxes are
        saved explicitly, so closing a recording must never make them vanish.
        """
        self.rebind(paths.SESSIONS_DIR, [], self.pools[0] / "_unbound.decisions.json", "")
        return {"ok": True, "archived": []}

    def _load_decisions(self) -> dict[str, dict]:
        if self.decisions_path.exists():
            return json.loads(self.decisions_path.read_text())
        return {}

    def _save_decisions(self) -> None:
        write_json_atomic(self.decisions_path, self.decisions, indent=0)

    # ---------------------------------------------------------------- drawing

    def _bound_recording(self) -> str:
        return recording_name(self.frames_dir)

    def draw_frames(self, sample: int = 40, percent: int | None = None) -> dict:
        """the frames to draw boxes on, evenly spread through the bound recording.

        evenly spread, not random: neighbouring frames are near-duplicates, so a random draw
        spends the budget on the same picture twice. boxes already drawn come back too - they are
        saved on every save, and a page that held them only in memory showed a blank canvas over
        frames that had already been worked.
        """
        if not self.frames_dir.is_dir():
            return {"ready": False, "reason": "bind a recording first", "frames": []}
        found = frame_files(self.frames_dir)
        if not found:
            return {"ready": False, "reason": f"no frames under {self.frames_dir}", "frames": []}
        # a percentage means the same coverage on a 40-frame run and a 400-frame one
        if percent:
            sample = max(1, round(len(found) * min(100, max(1, percent)) / 100))
        if sample and 0 < sample < len(found):
            step = len(found) / sample
            found = [found[int(i * step)] for i in range(sample)]
        names = [p.name for p in found]

        # a recording and its drawn set are the same work: binding the raw recording loads its
        # boxes back, or a fresh pass would look like a shrink against the set on disk
        candidates = self.candidates
        if not candidates:
            drawn = sorted(
                Path(self.bases["labels"]).glob(
                    f"{_flat(self._bound_recording())}.drawn-*.candidates.jsonl"
                )
            )
            if drawn:
                candidates = read_jsonl(drawn[0])

        existing: dict[str, list[dict]] = {}
        negatives: dict[str, list[dict]] = {}
        for candidate in candidates:
            box = {k: candidate[k] for k in ("left", "top", "width", "height")}
            # the negative flag is the only thing that says which a stored box is; without the
            # split a saved negative comes back as a positive and the next save doubles the set
            target = negatives if candidate.get("negative") else existing
            target.setdefault(candidate["path"], []).append(box)
        names.extend(name for name in existing if name not in names)
        return {
            "ready": True,
            "reason": "",
            "frames": sorted(names),
            "boxes": existing,
            "negatives": negatives,
            "dataset": self.session_tag or "",
            "recording": self._bound_recording(),
            "drawn": sum(len(v) for v in existing.values()),
        }

    def frame_bytes(self, name: str) -> bytes | None:
        """one whole frame from the bound recording, for drawing on"""
        try:
            return self.renderer.frame(name, frames_dir=self.frames_dir)
        except (FileNotFoundError, OSError):
            return None

    def save_drawn(
        self,
        boxes: list[dict],
        mode: str = "drawn",
        negatives: list[dict] | None = None,
        set_name: str | None = None,
    ) -> dict:
        """turn drawn boxes into a candidates file, bind it, and cut every box into the pool.

        THE SIZE IS FITTED, NOT TAKEN: a hand-drawn box is a few pixels out in each direction, and
        the median width and height across the set is a better estimate of the real object than
        any single one, so each box keeps its position and takes the set's size.

        THE EXCEPTION IS A NATIVE SET. when the training set (`set_name`, else the active one)
        has size_mode "native" every box keeps the width and height it was drawn at, marked
        `size_mode: native` so a later re-cut reads the stored rectangle and never refits it.
        """
        if not boxes:
            return {"error": "no boxes drawn"}
        widths = sorted(int(b["width"]) for b in boxes)
        heights = sorted(int(b["height"]) for b in boxes)
        width = widths[len(widths) // 2]
        height = heights[len(heights) // 2]
        # imported here: setconfig itself imports this module's atomic writer
        from smolsmort.review import setconfig

        size_mode = setconfig.set_config(set_name or self.active_set)["size_mode"]
        native = size_mode == "native"

        def candidate_of(box: dict, **extra) -> dict:
            return {
                "path": Path(box["path"]).name,
                "left": int(box["left"]),
                "top": int(box["top"]),
                "width": int(box["width"]) if native else width,
                "height": int(box["height"]) if native else height,
                "matched_template": "drawn",
                "score": 1.0,  # a human drew it; nothing scored it
                **({"size_mode": "native"} if native else {}),
                **extra,
            }

        # negatives come from the page, placed by the tool and correctable by hand before saving,
        # so nobody is surprised by one landing on something that looks like the real thing
        candidates = [candidate_of(b) for b in boxes]
        candidates += [candidate_of(b, negative=True) for b in (negatives or [])]
        # refuse rather than write an orphan no later step could resolve back to a recording
        if not self.session_tag:
            return {"error": "no dataset bound - pick one before saving"}
        base = _flat(self.session_tag.split(".")[0])
        labels_root = Path(self.bases["labels"])

        # re-saving a recording updates its set instead of adding another; a stamped sibling per
        # save would fill the list with duplicates of one recording
        existing = sorted(labels_root.glob(f"{base}.{mode}-*.candidates.jsonl"))
        if existing:
            out = existing[0]
            # a save must never silently shrink a set: a page bound to the raw recording holds
            # nothing, and rewriting over 94 boxes with 1 loses them with no way back
            held = sum(1 for line in out.read_text().splitlines() if line.strip())
            if len(candidates) < held:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                out = labels_root / f"{base}.{mode}-{stamp}.candidates.jsonl"
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            out = labels_root / f"{base}.{mode}-{stamp}.candidates.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(c) + "\n" for c in candidates))

        # a drawn box is already confirmed, and the dataset builder has to be told: it only counts
        # a candidate as an object when its decision says keep, and treats the rest as ignore
        decisions = {
            str(index): {"keep": True, "origin": "drawn"} for index in range(len(candidates))
        }
        decisions_path = out.with_suffix(".decisions.json")
        write_json_atomic(decisions_path, decisions, indent=1)

        saved = self._cut_drawn(candidates, width, height)
        self.rebind(self.frames_dir, candidates, decisions_path, out.stem)
        return {
            "out": str(out),
            # positives only: the page prints this as "N positives"
            "count": len(boxes),
            "negatives": len(negatives or []),
            "width": width,
            "height": height,
            "size_mode": size_mode,
            "tiles": saved,
            "spread": {"width": [widths[0], widths[-1]], "height": [heights[0], heights[-1]]},
        }

    # ---------------------------------------------------------------- cutting tiles

    def crop_pads(self, width: int, height: int) -> tuple[int, int]:
        """how much ground a tile keeps around a box of this size, per side, in pixels.

        THE ONE PLACE TILE GEOMETRY IS DECIDED - `_cut_drawn` and the settings preview both read
        it. percent mode is a fraction of the box per side (pad_x of the width, pad_y of the
        height); absolute mode is a crop size centred on the box. fixed aspect keeps the crop the
        box's own shape (one pad fraction for both axes, or crop_h derived from crop_w); free uses
        both numbers as given. never smaller than the box.
        """
        if self.crop_mode == "absolute":
            crop_w = max(width, int(self.crop_w))
            if self.aspect == "fixed":
                crop_h = round(crop_w * height / width) if width else height
            else:
                crop_h = max(height, int(self.crop_h))
            return round((crop_w - width) / 2), round((crop_h - height) / 2)
        pad_y_share = self.pad_x if self.aspect == "fixed" else self.pad_y
        return round(width * self.pad_x), round(height * pad_y_share)

    def _cut_drawn(
        self,
        candidates: list[dict],
        width: int,
        height: int,
        frames_dir: Path | None = None,
        tag: str | None = None,
        start_index: int = 0,
        select: set[int] | None = None,
    ) -> int:
        """one tile per box, straight into the primary pool.

        start_index offsets the names, so re-cutting ONE corrected box overwrites its own tile.

        select cuts only some candidates while naming them by position in the WHOLE list: a tile
        resolves back to its box by the index in its name, so handing this a filtered list would
        name 40 kept proposals k00000..k00039 against a file whose first 40 rows are other boxes.

        frames_dir and tag are for a sweep, which cuts from a recording other than the bound one
        and wants its own tag so the batch can be closed separately.

        each FRAME is opened once, not once per box. a box whose padded crop runs off the frame
        falls back to the bare box, and one that cannot be cut at all is skipped: losing a crop
        is better than losing the save.
        """
        self.unsorted_dir.mkdir(parents=True, exist_ok=True)
        source_dir = frames_dir or self.frames_dir
        recording = tag or _flat(recording_name(source_dir))
        by_frame: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            if select is not None and index not in select:
                continue
            by_frame.setdefault(candidate["path"], []).append(index)

        saved = 0
        for name, indices in by_frame.items():
            path = source_dir / name
            if not path.is_file():
                continue
            try:
                frame = read_frame(path)
            except (OSError, ValueError):
                continue
            for index in indices:
                candidate = candidates[index]
                # a native box is cut from its own stored rectangle, not the set's fitted size
                if candidate.get("size_mode") == "native":
                    box_w, box_h = int(candidate["width"]), int(candidate["height"])
                else:
                    box_w, box_h = width, height
                pad_x, pad_y = self.crop_pads(box_w, box_h)
                # named after the RECORDING, not the dataset: the dataset tag carries the file's
                # timestamp, so every save would mint fresh names and replace nothing
                tile_name = f"{recording}_k{start_index + index:05d}"
                top, left = int(candidate["top"]), int(candidate["left"])
                try:
                    tile = cut_tile(
                        frame,
                        tile_name,
                        top=top - pad_y,
                        left=left - pad_x,
                        height=box_h + pad_y * 2,
                        width=box_w + pad_x * 2,
                    )
                except TileError:
                    try:
                        tile = cut_tile(
                            frame, tile_name, top=top, left=left, height=box_h, width=box_w
                        )
                    except TileError:
                        continue
                save_tile(tile, self.unsorted_dir)
                saved += 1
        return saved

    # ---------------------------------------------------------------- crop settings

    def crop_settings(self) -> dict:
        """the crop rule, and what it means for a box of the size actually being drawn.

        the preview matters: a fraction is not something anyone can picture.
        """
        width, height = self.uniform_width, self.uniform_height
        pad_x, pad_y = self.crop_pads(width, height)
        return {
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "crop_mode": self.crop_mode,
            "crop_w": self.crop_w,
            "crop_h": self.crop_h,
            "aspect": self.aspect,
            "box": [width, height],
            "crop": [width + 2 * pad_x, height + 2 * pad_y],
        }

    def set_crop_settings(
        self,
        pad_x: float | None = None,
        pad_y: float | None = None,
        *,
        crop_mode: str | None = None,
        crop_w: int | None = None,
        crop_h: int | None = None,
        aspect: str | None = None,
    ) -> dict:
        """CHANGING THESE MOVES NO PIXELS: tiles already cut keep the crop they were cut with
        until a re-cut. pad_x/pad_y alone still work, so an older client or settings file that
        knows only those keeps its meaning (percent, free)."""
        if crop_mode is not None and crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}")
        if aspect is not None and aspect not in ASPECTS:
            raise ValueError(f"aspect must be one of {ASPECTS}, got {aspect!r}")
        if pad_x is not None:
            self.pad_x = max(0.0, min(3.0, float(pad_x)))
        if pad_y is not None:
            self.pad_y = max(0.0, min(3.0, float(pad_y)))
        if crop_mode is not None:
            self.crop_mode = crop_mode
        if aspect is not None:
            self.aspect = aspect
        if crop_w is not None:
            self.crop_w = max(0, min(MAX_CROP_PX, int(crop_w)))
        if crop_h is not None:
            self.crop_h = max(0, min(MAX_CROP_PX, int(crop_h)))
        self._save_crop_settings()
        return self.crop_settings()

    def _save_crop_settings(self) -> None:
        if self.crop_file is None:
            return
        settings = self.crop_settings()
        keep = ("pad_x", "pad_y", "crop_mode", "crop_w", "crop_h", "aspect")
        # a setting that cannot be remembered still works for this run
        with contextlib.suppress(OSError):
            write_json_atomic(self.crop_file, {k: settings[k] for k in keep}, indent=2)

    def load_crop_settings(self) -> None:
        """apply the remembered crop rule, if any. a file with only pad_x/pad_y (written before
        the other fields existed) loads as percent/free."""
        if self.crop_file is None:
            return
        try:
            saved = json.loads(self.crop_file.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(saved, dict):
            with contextlib.suppress(ValueError, TypeError):
                self.set_crop_settings(
                    saved.get("pad_x"),
                    saved.get("pad_y"),
                    crop_mode=saved.get("crop_mode"),
                    crop_w=saved.get("crop_w"),
                    crop_h=saved.get("crop_h"),
                    aspect=saved.get("aspect"),
                )

    # ---------------------------------------------------------------- what find can open

    def bindable_recordings(self) -> dict:
        """what find can open: EVERY recording under sessions/, plus the model's proposal sets.

        every recording is always listed - skipping one that already had boxes made it vanish from
        the picker the moment a single box was drawn. drawn sets are not rows of their own: binding
        the raw recording loads its boxes back, so the drawn work shows as a count on the
        recording's row. model sweeps stay listed as a second column, since they are meant for
        select but can be inspected here.
        """
        classed: dict[str, int] = {}
        negative: dict[str, int] = {}
        for tile_name, record in self.pool_records().items():
            bucket = (
                negative if record.get("excluded") else (classed if record.get("label") else None)
            )
            if bucket is not None:
                tag = _tile_tag(tile_name)
                bucket[tag] = bucket.get(tag, 0) + 1

        drawn: dict[str, dict] = {}
        sweep_rows = []
        if paths.LABELS_DIR.is_dir():
            for path in sorted(paths.LABELS_DIR.glob("*.candidates.jsonl")):
                tag = path.stem.removesuffix(".candidates")
                lines = [line for line in path.read_text().splitlines() if line.strip()]
                decided = kept = 0
                decisions_path = path.with_suffix(".decisions.json")
                if decisions_path.exists():
                    records = json.loads(decisions_path.read_text()).values()
                    decided = sum(1 for r in records if r.get("keep") or r.get("discard"))
                    kept = sum(1 for r in records if r.get("keep"))
                if ".cnn-" in tag:
                    # an empty sweep is not worth offering: opening it changes nothing on the page
                    if lines:
                        pool_tag = tag.replace(".cnn-", "_cnn-")
                        sweep_rows.append(
                            {
                                "kind": "dataset",
                                "name": tag,
                                "frame_count": len(lines),
                                "decided": decided,
                                "kept": kept,
                                "classed": classed.get(pool_tag, 0),
                                "negative": negative.get(pool_tag, 0),
                            }
                        )
                    continue
                session = tag.split(".drawn-")[0].replace("__", "/")
                previous = drawn.get(session, {"boxes": 0, "decided": 0, "kept": 0})
                drawn[session] = {
                    "boxes": previous["boxes"] + len(lines),
                    "decided": previous["decided"] + decided,
                    "kept": previous["kept"] + kept,
                }

        rows = []
        if paths.SESSIONS_DIR.is_dir():
            for frames_dir in session_frames_dirs(paths.SESSIONS_DIR):
                name = str(frames_dir.parent.relative_to(paths.SESSIONS_DIR))
                work = drawn.get(name)
                flat = flat_recording(name)
                rows.append(
                    {
                        "kind": "session",
                        "name": name,
                        "frame_count": len(frame_files(frames_dir)),
                        "boxes": work["boxes"] if work else 0,
                        "decided": work["decided"] if work else None,
                        "kept": work["kept"] if work else None,
                        "classed": classed.get(flat, 0),
                        "negative": negative.get(flat, 0),
                    }
                )
        rows.extend(sweep_rows)
        # which row is live, so the other tabs' datasets can be told apart
        return {"rows": rows, "current": self.session_tag}

    def bind_recording(self, kind: str, name: str) -> dict:
        """open a dataset (a candidates file) or a raw recording in find.

        raises FileNotFoundError when it is not there and ValueError for an unknown kind.
        """
        from smolsmort.review.recordings import session_frames_for

        if kind == "dataset":
            candidates_path = paths.LABELS_DIR / f"{_flat(name)}.candidates.jsonl"
            if not candidates_path.exists():
                raise FileNotFoundError(candidates_path)
            candidates = read_jsonl(candidates_path)
            frames_dir = session_frames_for(name)
            if frames_dir is None:
                raise FileNotFoundError(frames_dir_of(paths.SESSIONS_DIR / name))
            self.rebind(
                frames_dir, candidates, candidates_path.with_suffix(".decisions.json"), name
            )
        elif kind == "session":
            frames_dir = frames_dir_of(paths.SESSIONS_DIR / name)
            if not frames_dir.is_dir():
                raise FileNotFoundError(frames_dir)
            decisions_path = paths.LABELS_DIR / f"{_flat(name)}.candidates.decisions.json"
            self.rebind(frames_dir, [], decisions_path, name)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        return {"ok": True, "session_tag": self.session_tag, "frame_count": len(self.candidates)}

    # ---------------------------------------------------------------- judging candidates

    def is_pending(self, index: int) -> bool:
        decision = self.decisions.get(str(index), {})
        return not decision.get("keep") and not decision.get("discard")

    def _visible_indices(self, hide_done: bool = False) -> list[int]:
        """every candidate except ones marked reviewed. reviewed is a SEPARATE flag from
        keep/discard: a decided row still shows (dimmed) when paging back, until someone says
        "done looking at this one". hide_done also drops anything already decided."""
        return [
            i
            for i in range(len(self.candidates))
            if not self.decisions.get(str(i), {}).get("reviewed")
            and (not hide_done or self.is_pending(i))
        ]

    def _decision_record(self, index: int) -> dict:
        return self.decisions.get(str(index), {"keep": False, "discard": False, "saved_as": None})

    def mark_reviewed(self, index: int) -> None:
        with self.lock:
            record = self._decision_record(index)
            record["reviewed"] = True
            self.decisions[str(index)] = record
            self._save_decisions()

    def set_height_delta(self, index: int, delta: int) -> int:
        with self.lock:
            record = self._decision_record(index)
            record["height_delta"] = record.get("height_delta", 0) + delta
            self.decisions[str(index)] = record
            self._save_decisions()
            return record["height_delta"]

    def effective_height(self, index: int) -> int:
        """the global tile height plus this ONE candidate's own correction"""
        delta = self.decisions.get(str(index), {}).get("height_delta", 0)
        return max(4, self.uniform_height + delta)

    def page(self, page: int, size: int, hide_done: bool = False) -> dict:
        """stable pagination over visible candidates, in their original order - paging back lands
        on the same page it did before, decided rows included, so a decision is never mistaken
        for something that got lost."""
        visible = self._visible_indices(hide_done)
        chunk = visible[page * size : page * size + size]
        rows = []
        for index in chunk:
            candidate = self.candidates[index]
            decision = self.decisions.get(str(index), {})
            guess = None
            if self.guesser is not None:
                guess = self.guesser.guess(candidate)
            rows.append(
                {
                    "index": index,
                    "path": candidate["path"],
                    "left": candidate["left"],
                    "top": candidate["top"],
                    "width": candidate["width"],
                    "height": candidate["height"],
                    # a prefill only: the class scheme's optional guesser, never authoritative
                    "guess": guess,
                    "height_delta": decision.get("height_delta", 0),
                    "saved_rect": decision.get("rect"),
                    "keep": decision.get("keep", False),
                    "discard": decision.get("discard", False),
                }
            )
        total = len(visible)
        return {
            "rows": rows,
            "page": page,
            "total": total,
            "total_pages": max(1, -(-total // size)),
            "page_done": bool(chunk) and all(not self.is_pending(i) for i in chunk),
        }

    def crop_bytes(self, index: int) -> bytes:
        """the padded crop of one bound candidate. IndexError for an out-of-range index."""
        return self.renderer.crop(self.candidates[index], frames_dir=self.frames_dir)
