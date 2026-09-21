"""the select half of the review state: the tile pool, its judgements, and the unsaved buffer.

the pool is every tile cut from every dataset, clustered by the class a human gave it. a
judgement lives beside the tiles it is about, in `_labels.json` in each pool directory, keyed by
tile NAME - so re-cutting a tile keeps its label, and an index that exists in every dataset can
never collide.

CLASS ASSIGNMENTS ARE BUFFERED, NOT WRITTEN. `buffer_label` records one in memory keyed by dataset;
nothing reaches `_labels.json` until `save_labels`. promotion reads disk only, so an unsaved
assignment can never leak into a training set.
"""

from __future__ import annotations

import contextlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from smolsmort.review import paths
from smolsmort.review.classscheme import NOT_A_CLASS
from smolsmort.review.find_state import read_jsonl, write_json_atomic
from smolsmort.review.naming import _tile_index, _tile_tag
from smolsmort.review.recordings import dataset_tag, session_frames_for
from smolsmort.review.render import corrected_box
from smolsmort.review.tiles import load_tile, tile_png

UNJUDGED = "unjudged"


def _dataset_of(tile_name: str) -> str:
    """the dataset a tile belongs to; a bare "k00005" written before prefixes existed has none"""
    return _tile_tag(tile_name) or "untagged"


class PoolMixin:
    # ---------------------------------------------------------------- where the tiles are

    @property
    def unsorted_dir(self) -> Path:
        """the PRIMARY pool - where a newly cut tile is written and pool-wide state lives.

        reading spans every pool; writing only ever goes here, because which of several
        directories a new tile belongs to has no answer the tool could work out.
        """
        return self.pools[0]

    def tile_paths(self) -> list[Path]:
        """every tile in every pool, primary first. sidecars are not tiles: each is
        underscore-prefixed (_labels.json, _closed.json) while a tile is "<tag>_k<n>"."""
        found: dict[str, Path] = {}
        for pool in self.pools:
            if not pool.is_dir():
                continue
            for path in sorted(pool.glob("*.npz")):
                if not path.name.startswith("_"):
                    found.setdefault(path.stem, path)  # a name in two pools resolves to the first
        return list(found.values())

    def tile_path(self, name: str) -> Path | None:
        for pool in self.pools:
            candidate = pool / f"{name}.npz"
            if candidate.is_file():
                return candidate
        return None

    def pool_of(self, name: str) -> Path:
        """the pool a tile belongs to: where its file is, else where its record was read from,
        else the primary - so a detached pool's judgements do not migrate to the primary"""
        found = self.tile_path(name)
        if found is not None:
            return found.parent
        return self._record_origin.get(name, self.pools[0])

    # ---------------------------------------------------------------- closed sources

    @property
    def closed_file(self) -> Path:
        return self.pools[0] / "_closed.json"

    def closed_tags(self) -> set[str]:
        try:
            value = json.loads(self.closed_file.read_text())
        except (OSError, json.JSONDecodeError):
            return set()
        return set(value) if isinstance(value, list) else set()

    def _write_closed_tags(self, tags: set[str]) -> None:
        with contextlib.suppress(OSError):
            write_json_atomic(self.closed_file, sorted(tags), indent=0)

    def unsorted_names(self) -> list[str]:
        """every visible tile's NAME, without reading a pixel"""
        closed = self.closed_tags()
        names = sorted(path.stem for path in self.tile_paths())
        return [n for n in names if _dataset_of(n) not in closed] if closed else names

    def pool_sources(self) -> dict:
        """which recordings the pool's tiles came from, and how many each contributed.

        a closed source is still listed, marked closed, so the page can offer it back instantly
        rather than sending it round the cutting path as though it were new.
        """
        counts: dict[str, int] = {}
        for path in self.tile_paths():
            tag = _dataset_of(path.stem)
            counts[tag] = counts.get(tag, 0) + 1
        closed = self.closed_tags()
        return {
            "sources": [
                {"tag": t, "tiles": n, "closed": t in closed} for t, n in sorted(counts.items())
            ]
        }

    def pool_source_state(self, tag: str) -> dict:
        """what closing this source would cost, so the page can ask before doing anything.

        `judged` is the on-disk work: a tile carrying a class or marked "not a class". `unsaved`
        is what would be lost from the buffer.
        """
        records = self.pool_records()
        tiles = judged = 0
        for path in self.tile_paths():
            if _dataset_of(path.stem) != tag:
                continue
            tiles += 1
            record = records.get(path.stem) or {}
            if record.get("label") or record.get("excluded"):
                judged += 1
        return {"tag": tag, "tiles": tiles, "judged": judged, "unsaved": self.pending_count(tag)}

    def close_pool_source(self, tag: str, discard: bool = False) -> dict:
        """stop showing a source. NOTHING MOVES.

        closing is a view decision: the tag is remembered as hidden, the tiles stay on disk and
        keep their judgements, and opening it again is instant. an unsaved buffer for the source
        is REPORTED ({"unsaved": N}) and the source is left open; only `discard` drops it, and it
        also drops the source's saved judgements - the one destructive path, never the default.
        """
        if not tag:
            return {"error": "no source given"}
        unsaved = self.pending_count(tag)
        if unsaved and not discard:
            return {"tag": tag, "unsaved": unsaved}
        state = self.pool_source_state(tag)
        with self.lock:
            if discard:
                self._label_buffer.pop(tag, None)
                records = self.pool_records()
                for path in self.tile_paths():
                    if _dataset_of(path.stem) == tag:
                        records.pop(path.stem, None)
                self._write_pool_records(records)
            self._write_closed_tags(self.closed_tags() | {tag})
        return {
            "tag": tag,
            "closed": state["tiles"],
            "discarded": state["judged"] if discard else 0,
        }

    def open_pool_source(self, tag: str) -> dict:
        """show a source that was closed. instant, because closing never moved anything."""
        if not tag:
            return {"error": "no source given"}
        with self.lock:
            self._write_closed_tags(self.closed_tags() - {tag})
        return self.pool_source_state(tag)

    # ---------------------------------------------------------------- judgements on disk

    def pool_records(self) -> dict:
        """tile name -> {"label": str|None, "excluded": bool}, POOL-WIDE.

        a decision record keyed by candidate index is per dataset, but the grid shows every tile
        in the pool - judging a tile from one recording while another was bound wrote onto the
        other recording's index 5.
        """
        out = {}
        for name, value in self._read_pool_file().items():
            # older files stored a bare label string; keep reading them
            out[name] = {"label": value, "excluded": False} if isinstance(value, str) else value
        return out

    def _read_pool_file(self) -> dict:
        """every pool's judgements, merged, the primary winning a clash.

        one file per pool, not one for all: a generated pool's labels reproduce from its
        generator and must not land in the file holding what a human decided.
        """
        merged: dict = {}
        self._record_origin = {}
        for pool in self.pools:
            path = pool / "_labels.json"
            if not path.is_file():
                continue
            try:
                found = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(found, dict):
                continue
            for name, value in found.items():
                self._record_origin.setdefault(name, pool)
                merged.setdefault(name, value)
        return merged

    def _write_pool_records(self, records: dict) -> None:
        """back to the pool each record came from, atomically (temp file, then replace)"""
        split: dict[Path, dict] = {pool: {} for pool in self.pools}
        for name, record in records.items():
            split.setdefault(self.pool_of(name), {})[name] = record
        for pool, subset in split.items():
            path = pool / "_labels.json"
            # a pool that has never held a record and still holds none is left without a file
            if not subset and not path.is_file():
                continue
            with contextlib.suppress(OSError):
                write_json_atomic(path, subset, indent=1, sort_keys=True)

    def set_pool_excluded(self, name: str, excluded: bool) -> None:
        records = self.pool_records()
        record = records.setdefault(name, {"label": None, "excluded": False})
        record["excluded"] = excluded
        # a record claiming nothing is not a record
        if not record.get("label") and not excluded:
            records.pop(name, None)
        self._write_pool_records(records)

    def pool_labels(self) -> dict:
        """tile name -> its saved class"""
        return {n: r["label"] for n, r in self.pool_records().items() if r.get("label")}

    def set_excluded(self, name: str, excluded: bool) -> bool | None:
        """DECLARATIVE, not a toggle: state what this tile should be and it becomes that.

        reversible, not a delete - the tile stays on disk, dimmed in its group. a toggle came out
        inverted on a multi-selection where some tiles were already judged. the audit line is
        written only on a real change: a no-op press is not history. None if no such tile.

        marking a tile "not a class" drops any unsaved class assignment for it: the later press
        wins, and a flush must not resurrect a class beside the exclusion.
        """
        with self.lock:
            if self.tile_path(name) is None:
                return None
            before = bool(self.pool_records().get(name, {}).get("excluded", False))
            if before == excluded:
                return excluded
            if excluded:
                self._label_buffer.get(_dataset_of(name), {}).pop(name, None)
            self.set_pool_excluded(name, excluded)
            entry = {
                "name": name,
                "index": _tile_index(name),
                "at": datetime.now(UTC).isoformat(),
                "action": "excluded" if excluded else "re-included",
            }
            with (self.unsorted_dir / "_exclusions.jsonl").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            return excluded

    # ---------------------------------------------------------------- the unsaved buffer

    def buffer_label(self, name: str, label: str, definition: str) -> int | None:
        """record a class assignment in memory, keyed by dataset. returns the pending total, or
        None when the tile does not exist or no class was given."""
        if self.tile_path(name) is None or not label:
            return None
        with self.lock:
            self._label_buffer.setdefault(_dataset_of(name), {})[name] = (label, definition)
            return self.pending_count()

    def pending_count(self, tag: str | None = None) -> int:
        if tag is not None:
            return len(self._label_buffer.get(tag, {}))
        return sum(len(entries) for entries in self._label_buffer.values())

    def labels_buffer(self) -> dict:
        return {
            "pending": self.pending_count(),
            "datasets": [
                {"tag": tag, "pending": len(entries)}
                for tag, entries in sorted(self._label_buffer.items())
                if entries
            ],
        }

    def save_labels(self) -> dict:
        """flush the buffer to disk in ONE atomic write per pool, and empty it.

        assigning a class is the opposite press to "not a class", so it un-excludes. a buffered
        tile that has since gone from the pool is skipped and not counted.
        """
        with self.lock:
            records = self.pool_records()
            saved = 0
            for entries in self._label_buffer.values():
                for name, (label, definition) in entries.items():
                    if self.tile_path(name) is None:
                        continue
                    record = records.setdefault(name, {"label": None, "excluded": False})
                    record.update(label=label, classdef=definition, excluded=False)
                    saved += 1
            if saved:
                self._write_pool_records(records)
            self._label_buffer.clear()
        return {"saved": saved}

    def pool_definitions(self) -> dict:
        """tile name -> the class definition it was judged under, for tiles that record one.
        a tile without one is not broken, only older than definitions."""
        return {n: r["classdef"] for n, r in self.pool_records().items() if r.get("classdef")}

    # ---------------------------------------------------------------- the grid

    def clusters(self) -> dict:
        """every tile grouped by the class it was given, in a fixed order.

        assigned classes first (by name), then everything nobody has decided, then "not a class"
        last because it is the one group you are finished with. within a group, by name, so a
        tile only ever moves when its judgement does. unsaved assignments are shown (`pending`)
        so the grid matches what was just done; they are not on disk until saved.
        """
        names = self.unsorted_names()
        records = self.pool_records()
        pending = {n: v for entries in self._label_buffer.values() for n, v in entries.items()}

        def item_of(name: str) -> dict:
            record = records.get(name, {})
            buffered = pending.get(name)
            return {
                "name": name,
                "excluded": False if buffered else record.get("excluded", False),
                "assigned": buffered[0] if buffered else record.get("label"),
                "pending": buffered is not None,
                "source": _dataset_of(name),
            }

        by_label: dict[str, list[dict]] = {}
        unjudged: list[dict] = []
        excluded: list[dict] = []
        for name in names:
            item = item_of(name)
            if item["excluded"]:
                excluded.append(item)
            elif item["assigned"]:
                by_label.setdefault(item["assigned"], []).append(item)
            else:
                unjudged.append(item)

        clusters = [{"label": label, "items": by_label[label]} for label in sorted(by_label)]
        for label, group in ((UNJUDGED, unjudged), (NOT_A_CLASS, excluded)):
            if group:
                clusters.append({"label": label, "items": group})
        return {
            "clusters": clusters,
            "counts": {
                "boxes_loaded": len(names),
                "classes_assigned": sum(len(group) for group in by_label.values()),
            },
        }

    # ---------------------------------------------------------------- resolving a tile to its box

    def _dataset_for(self, tag: str):
        """(candidates, decisions, frames_dir) for the dataset a tile BELONGS to, which is not
        necessarily the bound one. cached per tag: the grid asks for ~100 thumbs at a time."""
        from smolsmort.review.recordings import candidates_file_for

        if not tag or tag == self.session_tag:
            return self.candidates, self.decisions, self.frames_dir
        if tag in self._dataset_cache:
            return self._dataset_cache[tag]
        candidates_path = candidates_file_for(tag)
        frames_dir = session_frames_for(tag)
        if candidates_path is None or frames_dir is None:
            entry = (self.candidates, self.decisions, self.frames_dir)
        else:
            decisions_path = candidates_path.with_suffix(".decisions.json")
            decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
            entry = (read_jsonl(candidates_path), decisions, frames_dir)
        self._dataset_cache[tag] = entry
        return entry

    def unsorted_thumb_bytes(self, name: str) -> bytes | None:
        """a pool tile shown with real frame around it, from the ORIGINAL frame of the dataset
        it came from. falls back to the saved tile itself only when no box is on record."""
        path = self.tile_path(name)
        if path is None:
            return None
        index = _tile_index(name)
        candidates, decisions, frames_dir = self._dataset_for(_tile_tag(name))
        if index < len(candidates):
            shown = corrected_box(candidates[index], decisions.get(str(index)))
            return self.renderer.thumb(shown, frames_dir=frames_dir)
        return tile_png(load_tile(path))

    def _pool_box(self, name: str) -> dict | None:
        """everything a pool tile needs to be re-aligned: its box, its file and its frames.

        a tile is "<tag>_k<index>" and the index counts the NON-NEGATIVE boxes, not the file's
        line numbers. both are carried: the index addresses the tile, the line is written back.
        """
        if "_k" not in name:
            return None
        tag, _, digits = name.rpartition("_k")
        if not digits.isdigit():
            return None
        index = int(digits)
        labels_root = Path(self.bases["labels"])
        path = next(
            (
                p
                for p in sorted(labels_root.glob("*.candidates.jsonl"))
                if dataset_tag(p.name)[0] == tag
            ),
            None,
        )
        if path is None:
            return None
        boxes = read_jsonl(path)
        live = [(i, b) for i, b in enumerate(boxes) if not b.get("negative")]
        if index >= len(live):
            return None
        line_index, box = live[index]
        frames_dir = session_frames_for(path.name.split(".")[0])
        if frames_dir is None or not frames_dir.is_dir():
            return None
        kept = [b for _i, b in live]
        return {
            "path": path,
            "boxes": boxes,
            "line_index": line_index,
            "box": box,
            "frames_dir": frames_dir,
            # the cut size is the median over the whole set, exactly as _cut_drawn derives it: one
            # box moving must not change the size every other tile was cut at
            "width": int(statistics.median(b["width"] for b in kept)),
            "height": int(statistics.median(b["height"] for b in kept)),
        }

    def pool_box_info(self, name: str) -> dict:
        """where this tile's box currently sits inside the padded crop the browser draws.
        left/top are MARGIN_X/MARGIN_Y by construction, so a correct box needs no drag."""
        found = self._pool_box(name)
        if found is None:
            return {"error": f"no source on disk for {name!r}"}
        box = found["box"]
        return {
            "name": name,
            "rect": {
                "left": paths.MARGIN_X,
                "top": paths.MARGIN_Y,
                "width": int(box["width"]),
                "height": int(box["height"]),
            },
            "bounds": {
                "width": int(box["width"]) + 2 * paths.MARGIN_X,
                "height": int(box["height"]) + 2 * paths.MARGIN_Y,
            },
        }

    def pool_tile_bytes(self, name: str) -> bytes | None:
        """the padded crop around a pool tile's box, from ITS OWN recording, not the bound one"""
        found = self._pool_box(name)
        if found is None:
            return None
        return self.renderer.crop(found["box"], frames_dir=found["frames_dir"])

    def realign_pool_tile(self, name: str, left: float, top: float) -> dict:
        """move one box by the drag, then re-cut just its tile over the old file.

        THE NAME IS UNCHANGED, which is the point: labels are keyed by name, so correcting a crop
        costs a re-cut and keeps the class it already had.
        """
        found = self._pool_box(name)
        if found is None:
            return {"error": f"no source on disk for {name!r}"}
        box, path = found["box"], found["path"]
        # the drag is crop-local and the box sat at (MARGIN_X, MARGIN_Y), so the shift is the
        # difference, applied to the box's real position in the frame
        box["left"] = int(round(box["left"] + (left - paths.MARGIN_X)))
        box["top"] = int(round(box["top"] + (top - paths.MARGIN_Y)))
        found["boxes"][found["line_index"]] = box
        tag, _, digits = name.rpartition("_k")
        with self.lock:
            path.write_text("".join(json.dumps(b) + "\n" for b in found["boxes"]))
            cut = self._cut_drawn(
                [box],
                found["width"],
                found["height"],
                frames_dir=found["frames_dir"],
                tag=tag,
                start_index=int(digits),
            )
        if not cut:
            return {"error": f"{name} could not be cut there - it runs off the frame"}
        return {"name": name, "left": box["left"], "top": box["top"]}

    def recut_pool(self) -> dict:
        """re-cut every drawn set's tiles from its frames, at the current crop rule.

        labels survive: a tile's filename comes from its recording and index, so re-cutting
        overwrites in place and `_labels.json` still points at the same box.
        """
        done, failed = [], []
        for path in sorted(Path(self.bases["labels"]).glob("*.drawn-*.candidates.jsonl")):
            tag = path.name.split(".drawn-")[0]
            boxes = [b for b in read_jsonl(path) if not b.get("negative")]
            if not boxes:
                continue
            frames_dir = session_frames_for(tag)
            if frames_dir is None or not frames_dir.is_dir():
                failed.append({"tag": tag, "reason": "no frames directory"})
                continue
            width = int(statistics.median(b["width"] for b in boxes))
            height = int(statistics.median(b["height"] for b in boxes))
            cut = self._cut_drawn(boxes, width, height, frames_dir=frames_dir, tag=tag)
            done.append(
                {"tag": tag, "boxes": len(boxes), "tiles": cut, "size": f"{width}x{height}"}
            )
        return {"recut": done, "failed": failed}

    # ---------------------------------------------------------------- opening datasets

    def openable_datasets(self) -> dict:
        """candidates files on disk that are NOT currently in the pool.

        the pool is the open set: a dataset exists as a candidates file the moment find saves it
        or a sweep writes it, and becomes visible in select once its tiles are cut in.
        """
        open_tags = {_dataset_of(path.stem) for path in self.tile_paths()}
        found = []
        for path in sorted(Path(self.bases["labels"]).glob("*.candidates.jsonl")):
            tag, source = dataset_tag(path.name)
            if tag in open_tags:
                continue
            boxes = sum(1 for line in path.read_text().splitlines() if line.strip())
            # a sweep that found nothing still writes its file; offering it made "open" look broken
            if not boxes:
                continue
            found.append({"tag": tag, "file": path.name, "boxes": boxes, "source": source})
        return {"datasets": found}

    def open_dataset(self, name: str) -> dict:
        """cut one candidates file's tiles into the pool, which is what makes it visible"""
        path = Path(self.bases["labels"]) / Path(name).name
        if not path.is_file():
            return {"error": f"no dataset named {name!r}"}
        boxes = [b for b in read_jsonl(path) if not b.get("negative")]
        if not boxes:
            return {"error": f"{name} holds no boxes"}
        tag, _source = dataset_tag(path.name)
        frames_dir = session_frames_for(path.name.split(".")[0])
        if frames_dir is None or not frames_dir.is_dir():
            return {"error": f"no frames directory for {tag}"}
        width = int(statistics.median(b["width"] for b in boxes))
        height = int(statistics.median(b["height"] for b in boxes))
        cut = self._cut_drawn(boxes, width, height, frames_dir=frames_dir, tag=tag)
        return {"tag": tag, "tiles": cut, "boxes": len(boxes)}

    def training_sets(self) -> dict:
        """what a model may train on: the durable sets promotion writes, never a raw candidates
        queue - a queue is unreviewed and unlabelled by definition."""
        root = paths.DATASETS_DIR
        rows = []
        for path in sorted(root.glob("*.jsonl")) if root.is_dir() else []:
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            classes: dict[str, int] = {}
            for line in lines:
                try:
                    label = json.loads(line).get("label")
                except json.JSONDecodeError:
                    continue
                if label:
                    classes[label] = classes.get(label, 0) + 1
            rows.append(
                {"name": path.stem, "path": str(path), "rows": len(lines), "classes": classes}
            )
        return {"sets": rows}


__all__ = ["UNJUDGED", "PoolMixin"]
