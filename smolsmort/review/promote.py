"""promotion: every judged tile in the pool becomes a row of ONE durable, named training set.

WHY A SEPARATE FILE rather than training straight off a candidates file. a candidates file is a
review QUEUE - one recording, one run, and a new one every time anything is drawn or swept. the
training set is the accumulating answer: boxes drawn on one recording and model proposals confirmed
on another land here under one schema, and training reads one file.

a tile is "<dataset tag>_k<candidate index>", which is how a row resolves back to the box, frame
and recording it came from. PROMOTION READS THE PROMOTED STATE ON DISK ONLY: class assignments
still sitting in the unsaved buffer (see pool_state) are not part of any set.
"""

from __future__ import annotations

import json
import random
import zlib
from datetime import datetime
from pathlib import Path

from PIL import Image

from smolsmort.review import paths, setconfig
from smolsmort.review.find_state import read_jsonl, write_json_atomic
from smolsmort.review.naming import DEFAULT_SET_NAME, _tile_index, _tile_tag, set_filename
from smolsmort.review.recordings import session_frames_for
from smolsmort.review.render import corrected_box

# what makes two rows the same box: the recording, the frame and the rect. NOT the source tag - the
# same box re-proposed by a later sweep is still the same box, and a merge must not keep both
ROW_KEY = ("recording", "frame", "left", "top", "width", "height")


def row_key(row: dict) -> tuple:
    return tuple(row.get(field) for field in ROW_KEY)


def synthetic_negatives(
    boxes: list[dict], frame_size: tuple[int, int], seed: int = 0
) -> list[dict]:
    """one background box per drawn box, placed where no drawn box is.

    rests on one assumption: every object visible in the frame was marked. given that, anywhere
    else in the frame IS background, which makes a balanced negative set free. the rectangles must
    not overlap, though the training window around them may. seeded, so promoting twice gives the
    same set rather than quietly growing it.
    """
    width, height = frame_size
    rng = random.Random(seed)
    placed = [(b["left"], b["top"], b["width"], b["height"]) for b in boxes]
    out = []
    for box in boxes:
        box_w, box_h = box["width"], box["height"]
        for _ in range(40):  # a bounded search; a frame this full simply yields fewer
            left = rng.randrange(0, max(1, width - box_w))
            top = rng.randrange(0, max(1, height - box_h))
            if any(
                left < px + pw and left + box_w > px and top < py + ph and top + box_h > py
                for px, py, pw, ph in placed
            ):
                continue
            placed.append((left, top, box_w, box_h))
            out.append(
                {"path": box["path"], "left": left, "top": top, "width": box_w, "height": box_h}
            )
            break
    return out


def read_training_set(path: Path) -> list[dict]:
    """the rows already on disk, skipping anything unreadable rather than refusing the file"""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def merge_rows(existing: list[dict], fresh: list[dict]) -> list[dict]:
    """add this pool's rows to a set without disturbing what other pools put there.

    the pool speaks only for its own sources: keeping every existing row would leave a re-judged
    source's old rows beside their replacements, and dropping them all would discard work done
    from another pool. a row survives unless this promote covers its source or re-states its
    exact box. synthetic negatives are never carried over - they are generated per promote from
    the current positives.
    """
    replaced = {row["source"] for row in fresh} | {"synthetic"}
    keys = {row_key(row) for row in fresh}
    kept = [
        row for row in existing if row.get("source") not in replaced and row_key(row) not in keys
    ]
    return kept + fresh


class PromoteMixin:
    @staticmethod
    def _candidates_for_tag(labels_root: Path, tag: str) -> Path | None:
        """the candidates file a tile with this tag actually came from.

        a sweep's file sorts before a drawn one (".cnn-" < ".drawn-"), so taking the first glob
        match resolved drawn tiles against an empty sweep of the same recording: every index fell
        out of range and promotion wrote zero rows while reporting success. a sweep tile names its
        file exactly through the stamp in its tag; a bare tag is drawn work and never matches a
        sweep; where several drawn passes exist, the caller picks the one holding the index.
        """
        if "_cnn-" in tag:
            recording, stamp = tag.rsplit("_cnn-", 1)
            exact = labels_root / f"{recording}.cnn-{stamp}.candidates.jsonl"
            return exact if exact.is_file() else None
        drawn = paths.drawn_candidates(labels_root, tag)
        if not drawn:
            # a mined set predates the ".drawn-" naming and has no infix at all
            plain = labels_root / f"{tag}.candidates.jsonl"
            return plain if plain.is_file() else None
        return drawn[0] if len(drawn) == 1 else None

    def _resolve_candidates(
        self, labels_root: Path, tag: str, indexes: list[int]
    ) -> tuple[list[dict], Path | None]:
        """the boxes for one tag, and the file they came from - the decisions sidecar must be the
        sidecar of the file actually chosen, or one box's correction lands on another"""
        exact = self._candidates_for_tag(labels_root, tag)
        options = [exact] if exact else sorted(labels_root.glob(f"{tag}.drawn-*.candidates.jsonl"))
        best: list[dict] = []
        best_path: Path | None = None
        for path in options:
            rows = read_jsonl(path)
            if indexes and max(indexes) < len(rows):
                return rows, path
            if len(rows) > len(best):
                best, best_path = rows, path
        return best, best_path

    @staticmethod
    def _corrections_beside(candidates_path: Path | None) -> dict:
        """the decisions sidecar's per-index records, or nothing (a sweep writes none)"""
        if candidates_path is None:
            return {}
        try:
            value = json.loads(candidates_path.with_suffix(".decisions.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def promote_to_training(self, name: str = DEFAULT_SET_NAME, mode: str = "new") -> dict:
        """every saved, labelled tile written into ONE durable training set.

        THE SET IS NAMED, and a name that already exists is a decision, not an overwrite: mode
        "new" refuses and reports {"exists": true} so the page can ask; "overwrite" replaces the
        file whole; "merge" keeps every row the current pool does not speak for (merge_rows). a
        sidecar <name>.meta.json records which source contributed what.
        """
        records = self.pool_records()

        # two definitions in one set is not a set: the channel map comes from the labels present,
        # so labels judged under two carve-ups would become unrelated channels for one thing
        seen_defs = {
            r["classdef"] for r in records.values() if r.get("classdef") and r.get("label")
        }
        if len(seen_defs) > 1:
            return {
                "error": (
                    "this pool holds tiles judged under different class definitions - "
                    + ", ".join(sorted(seen_defs))
                    + ". Close the datasets belonging to one of them and promote separately."
                )
            }

        by_tag: dict[str, list[tuple[int, str | None, bool, bool]]] = {}
        for tile, record in records.items():
            excluded = bool(record.get("excluded"))
            not_object = bool(record.get("not_object"))
            # an excluded swept box is a HARD NEGATIVE: the model fired there and was told no,
            # the one kind of background worth sampling deliberately. an excluded drawn box is
            # simply dropped - nothing proposed it. a not-an-object verdict is a hard negative on
            # any frame, drawn or swept, explicit or exhaustive
            if not excluded and not not_object and not record.get("label"):
                continue
            tag = _tile_tag(tile)
            if tag:
                by_tag.setdefault(tag, []).append(
                    (_tile_index(tile), record.get("label"), excluded, not_object)
                )

        rows, missing = [], 0
        labels_root = Path(self.bases["labels"])
        for tag, entries in by_tag.items():
            if not self._candidates_for_tag(labels_root, tag):
                missing += len(entries)
                continue
            candidates, candidates_path = self._resolve_candidates(
                labels_root, tag, [index for index, _, _, _ in entries]
            )
            corrections = self._corrections_beside(candidates_path)
            frames_dir = session_frames_for(tag)
            tag_rows: list[dict] = []
            for index, label, excluded, not_object in entries:
                if index >= len(candidates) or frames_dir is None:
                    missing += 1
                    continue
                box = corrected_box(candidates[index], corrections.get(str(index)))
                hard = excluded or not_object
                row = {
                    "recording": str(frames_dir.parent.relative_to(paths.SESSIONS_DIR)),
                    "frame": box["path"],
                    "left": box["left"],
                    "top": box["top"],
                    "width": box["width"],
                    "height": box["height"],
                    "label": None if hard else label,
                    "negative": hard,
                    "source": tag,
                }
                if not_object:
                    row["not_object"] = True
                tag_rows.append(row)
            if frames_dir is not None and tag_rows:
                self._apply_exhaustive(tag_rows, candidates, corrections, tag, entries)
            rows.extend(tag_rows)

        synthetic = self._add_synthetic_negatives(rows)

        out = paths.DATASETS_DIR / f"{set_filename(name)}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = read_training_set(out)

        # A PROMOTE THAT WROTE NOTHING MUST NOT DESTROY THE LAST ONE. the file is rewritten
        # whole, so a run with no rows would replace a good set with an empty one - because the
        # recordings stopped being findable, or because nothing in the pool was labelled yet.
        if not rows:
            pooled = len(self.tile_paths())
            if missing:
                why = (
                    f"all {missing} labelled tiles failed to find their recording. the sessions "
                    "base is usually the cause: check it points where the recordings actually live"
                )
            elif not pooled:
                why = "the pool is empty - open a dataset in select first"
            else:
                why = (
                    f"none of the {pooled} tiles in the pool is labelled or excluded yet - "
                    "tag them in select and save, then promote"
                )
            return {
                "error": f"nothing to promote: {why}. "
                f"the existing {len(existing)}-row training set was left alone",
                "out": str(out),
                "rows": 0,
                "unresolved": missing,
                "pending": self.pending_count(),
            }

        if existing and mode == "new":
            return {
                "exists": True,
                "name": out.stem,
                "out": str(out),
                "existing_rows": len(existing),
                "would_add": len(rows),
            }

        merged = merge_rows(existing, rows) if mode == "merge" and existing else rows
        # rewritten whole, not appended: promoting twice must not double every row
        out.write_text("".join(json.dumps(r) + "\n" for r in merged))
        meta = self._write_set_meta(
            out,
            merged,
            labels_root,
            mode if existing else "new",
            {r.get("source") for r in rows},
            setconfig.read_set_config(set_filename(name)),
        )
        counts: dict[str, int] = {}
        for row in merged:
            if row["label"]:
                counts[row["label"]] = counts.get(row["label"], 0) + 1
        negatives = sum(1 for row in merged if row.get("negative"))
        return {
            "out": str(out),
            "name": out.stem,
            "mode": mode if existing else "new",
            "rows": len(merged) - negatives,
            "negatives": negatives,
            "synthetic": synthetic,
            "unresolved": missing,
            "classes": counts,
            "sources": meta["sources"],
            "pending": self.pending_count(),
        }

    @staticmethod
    def _apply_exhaustive(
        tag_rows: list[dict],
        candidates: list[dict],
        corrections: dict,
        tag: str,
        entries: list[tuple[int, str | None, bool, bool]],
    ) -> None:
        """on a frame a human declared exhaustive, every proposed box nobody kept is background.

        rows on such a frame carry `exhaustive: true`. each candidate of the tag that was not
        judged becomes a negative row, unless its centre sits inside a kept box on that frame: that
        is a misaligned copy of a real object, not a confirmed absence. explicit frames untouched.
        """
        recording = tag_rows[0]["recording"]
        flagged = setconfig.exhaustive_frames(recording)
        if not flagged:
            return
        judged = {index for index, *_ in entries}
        kept = [r for r in tag_rows if not r["negative"]]
        for row in tag_rows:
            if row["frame"] in flagged:
                row["exhaustive"] = True
        for index, candidate in enumerate(candidates):
            if index in judged or candidate["path"] not in flagged:
                continue
            box = corrected_box(candidate, corrections.get(str(index)))
            cx = box["left"] + box["width"] / 2.0
            cy = box["top"] + box["height"] / 2.0
            inside = any(
                r["frame"] == box["path"]
                and abs(cx - (r["left"] + r["width"] / 2.0)) <= r["width"] / 2.0
                and abs(cy - (r["top"] + r["height"] / 2.0)) <= r["height"] / 2.0
                for r in kept
            )
            if inside:
                continue
            tag_rows.append(
                {
                    "recording": recording,
                    "frame": box["path"],
                    "left": box["left"],
                    "top": box["top"],
                    "width": box["width"],
                    "height": box["height"],
                    "label": None,
                    "negative": True,
                    "source": tag,
                    "exhaustive": True,
                }
            )

    @staticmethod
    def _add_synthetic_negatives(rows: list[dict]) -> int:
        """append one background box per positive, per frame; returns how many were added"""
        positives: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            if not row.get("negative"):
                positives.setdefault((row["recording"], row["frame"]), []).append(row)
        added = 0
        for (recording, frame), frame_rows in positives.items():
            image_path = paths.SESSIONS_DIR / recording / "frames" / frame
            try:
                with Image.open(image_path) as handle:
                    frame_size = handle.size
            except (OSError, ValueError):
                continue
            # crc32, NOT hash(): str hashing is randomised per process, which drew different
            # negatives on every promote and made a merge (keyed on the rect) never match
            seed = zlib.crc32(frame.encode()) & 0xFFFF
            for box in synthetic_negatives(
                [dict(r, path=frame) for r in frame_rows], frame_size, seed
            ):
                rows.append(
                    {
                        "recording": recording,
                        "frame": frame,
                        "left": box["left"],
                        "top": box["top"],
                        "width": box["width"],
                        "height": box["height"],
                        "label": None,
                        "negative": True,
                        "source": "synthetic",
                    }
                )
                added += 1
        return added

    def _write_set_meta(
        self,
        out: Path,
        rows: list[dict],
        labels_root: Path,
        mode: str,
        written_sources: set[str],
        config: dict | None = None,
    ) -> dict:
        """the sidecar that says where each row in the set came from.

        the rows carry a source tag, but nothing said which candidates FILE a tag was read out
        of, and that file is the only thing that can re-resolve a row to its box. a merged set
        outlives the queue it was built from, so provenance is written down, not rediscovered.
        """
        meta_path = out.with_suffix(".meta.json")
        previous = {}
        if meta_path.is_file():
            try:
                previous = {
                    entry["source"]: entry
                    for entry in json.loads(meta_path.read_text()).get("sources", [])
                }
            except (json.JSONDecodeError, KeyError, TypeError):
                previous = {}

        stamp = datetime.now().isoformat(timespec="seconds")
        by_source: dict[str, dict] = {}
        for row in rows:
            source = row.get("source") or "untagged"
            entry = by_source.setdefault(source, {"source": source, "rows": 0, "labels": {}})
            entry["rows"] += 1
            label = row.get("label") or ("negative" if row.get("negative") else "unlabelled")
            entry["labels"][label] = entry["labels"].get(label, 0) + 1

        for source, entry in by_source.items():
            found = self._candidates_for_tag(labels_root, source)
            was = previous.get(source, {})
            # a source this promote did not write keeps the file and timestamp it came in with
            entry["candidates"] = str(found) if found else was.get("candidates")
            entry["promoted"] = stamp if source in written_sources else was.get("promoted", stamp)

        meta = {
            "name": out.stem,
            "written": stamp,
            "mode": mode,
            "rows": len(rows),
            "sources": sorted(by_source.values(), key=lambda e: e["source"]),
        }
        # the backend the set names, recorded only for a set that has one
        if config:
            meta.update(backend=config["backend"], size_mode=config["size_mode"])
        write_json_atomic(meta_path, meta, indent=2)
        return meta
