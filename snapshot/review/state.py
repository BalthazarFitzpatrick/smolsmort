"""the review tool's own state: what is bound, what is in the pool, and what has been judged.

The biggest piece of the tool and the one every tab talks to. It owns no http and no rendering -
routes.py calls into it and serialises what comes back, which is what lets all of this be tested
without a socket.

PATHS ARE READ THROUGH `paths`, never imported by value - they are repointed at runtime by
set_bases and monkeypatched by tests, and a bound copy would silently keep pointing at the real
sessions/ and labels/. See review/paths.py.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import shutil
import statistics
import threading
import zlib
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from parent.imitation.record import resolve_session_paths
from snapshot.review import paths
from snapshot.review.naming import _flat, _tile_index, _tile_tag
from snapshot.vision.plate_templates import (
    PlateTemplate,
    TemplateError,
    crop_at_rect,
    extract_template,
    guess_best_template,
    load_template,
    load_templates,
    mirrored_guide_mask,
    save_template,
)

# Balthazar Fitzpatrick 2026-09-01: "the cluster order should be target, normal, off_target". plain alphabetical
# put off_target before target, which reads as an arbitrary shuffle of the same primitive
_STATE_ORDER = {"target": 0, "": 1, "default": 1, "off_target": 2}
SCRIPT_GROUPS = {
    "wt-record": ("record", "capture"),
    "wt-import-screenshots": ("record", "capture"),
    "wt-inspect": ("record", "inspect"),
    "wt-dump-inputs": ("record", "inspect"),
    "wt-behaviour": ("record", "inspect"),
    "wt-calibrate": ("calibrate", "display"),
    "wt-calibrate-addon-readout": ("calibrate", "display"),
    "wt-calibrate-compass": ("calibrate", "display"),
    "wt-calibrate-interface": ("calibrate", "display"),
    "wt-icon-states": ("calibrate", "interface"),
    "wt-example-stats": ("calibrate", "interface"),
    "wt-fit-timing": ("calibrate", "timing"),
    "wt-time-extractors": ("calibrate", "timing"),
    # the one-recording camera fit, split then fitted, and the scales it is built from
    "wt-split-camera": ("camera", "fit"),
    "wt-fit-camera": ("camera", "fit"),
    "wt-fit-focal": ("camera", "fit"),
    "wt-pitch-scale": ("camera", "scale"),
    "wt-pitch-sensitivity": ("camera", "scale"),
    "wt-zoom-ticks": ("camera", "scale"),
    "parent": ("nameplates", "curate"),
    "wt-review-templates": ("nameplates", "curate"),
    "wt-extract-nameplate-art": ("nameplates", "synth"),
    "wt-synth-frames": ("nameplates", "synth"),
    "wt-split-set": ("nameplates", "synth"),
    "wt-export-map": ("client", "maps"),
    "wt-extract-worldmap": ("client", "maps"),
    "wt-extract-obstacles": ("client", "maps"),
    "wt-survey-textures": ("client", "maps"),
    "wt-extract-icons": ("client", "art"),
    "wt-overlay": ("execute", "overlay"),
    "wt-overlay-live": ("execute", "overlay"),
    "wt-overlay-shadow": ("execute", "overlay"),
    "wt-bearing-overlay": ("execute", "overlay"),
    "wt-annotate-plates": ("camera", "fit"),
    "wt-humaniser-demo": ("execute", "humaniser"),
    "wt-stuck-replay": ("execute", "nav"),
}

# the commands his shell history shows him typing, most often first - counted 2026-09-10. the rest
# are run by a tool or an agent, and the scripts tab marks them as such
USER_CALLED = (
    "wt-record",
    "wt-calibrate",
    "wt-calibrate-addon-readout",
    "wt-review-templates",
    "wt-overlay",
    "wt-inspect",
    "wt-time-extractors",
    "wt-humaniser-demo",
    "wt-behaviour",
    "wt-dump-inputs",
)


def _label_order(item) -> tuple[str, int]:
    """primitive first, then state in Balthazar Fitzpatrick's order rather than alphabetically.

    TWO LABEL SHAPES, because two things write them. The hardcoded identities read
    "Hostile NPC (off_target)"; a class definition composes its dimensions as "hostile / off_target".
    Only the bracketed form was understood, so every classdef-driven label fell through to the
    default and sorted alphabetically - which puts default before off_target before target, the
    exact shuffle _STATE_ORDER exists to prevent. Visible the moment the generated set was listed.
    """
    label = item[0] if isinstance(item, tuple) else item
    for name in _STATE_ORDER:
        if name and label.endswith(f"({name})"):
            return label.replace(f" ({name})", ""), _STATE_ORDER[name]
        if name and label.endswith(f" / {name}"):
            return label[: -len(f" / {name}")], _STATE_ORDER[name]
    return label, _STATE_ORDER[""]


def _library_label(name: str) -> str:
    """the class a promoted template belongs to, off its filename: "druid_off_target" -> "druid".

    off_target and target are STATES of the same unit - an off_target druid plate is still a druid
    - so all three fold together here, matching how the classes collapse for training. the files
    stay separate on disk; this only groups the view. "_dim" is the pre-rename spelling, still read
    so an older template file is not orphaned.
    """
    for suffix in ("_off_target", "_target", "_dim"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _dataset_tag(filename: str) -> tuple[str, str]:
    """a candidates file's pool tag, and whether a hand or the cnn drew it.

    THE ONE PLACE THIS MAPPING LIVES. the pool is keyed by this tag, so opening a dataset, listing
    what is still openable and finding a tile's source all have to derive it identically -
    they each carried their own copy of the string surgery, which is three ways to disagree
    """
    if ".drawn-" in filename:
        return filename.split(".drawn-")[0], "find"
    if ".cnn-" in filename:
        base, rest = filename.split(".cnn-", 1)
        return f"{base}_cnn-{rest.split('.')[0]}", "sweep"
    return filename.split(".")[0], "find"


def _slugify(label: str) -> str:
    """ "Druid (off_target)" -> "druid_off_target" - a real template library filename, in the same
    plain lowercase shape the original seed templates use (red, grey, ...)."""

    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _set_filename(name: str) -> str:
    """a typed training-set name as ONE safe filename stem.

    the name comes from a text field, so it can carry a slash, a "..", or nothing at all - and it
    becomes a path under datasets/. anything but a word character, dot or dash collapses to an
    underscore, and an empty result falls back to the name this always had.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return stem or "plates"


def _session_frames_for(tag: str):
    """the frames dir a candidates file was mined from, allowing VARIANT names.

    a candidates file is not always named exactly after its session: filtering or re-mining one
    produces siblings like "<session>.band.candidates.jsonl", and binding those used to 404
    because it looked for a sessions/<whole tag>/ that never existed. so drop one dot-separated
    suffix at a time until a real session directory turns up.
    """
    candidate = tag
    while candidate:
        # "__" is _flat()'s stand-in for the "/" in a NESTED recording's name, since labels/ is a
        # flat directory. without undoing it here a drawn dataset bound to nothing: the file said
        # nameplate_pipeline_test__1 and the recording lives at nameplate_pipeline_test/1
        for name in (candidate, candidate.replace("__", "/")):
            _, frames_dir = resolve_session_paths(paths.SESSIONS_DIR / name)
            if frames_dir.is_dir():
                return frames_dir
        # A SWEEP TAG SEPARATES WITH AN UNDERSCORE, NOT A DOT: tiles from a sweep are named
        # "<recording>_cnn-<stamp>_k<n>", so the batch tag is "<recording>_cnn-<stamp>". Stripping
        # only dot-suffixes meant this returned None for every sweep batch, and promote_to_training
        # then dropped each one as unresolved - MEASURED, 24 labelled proposals silently discarded
        # while promote reported success. That is the whole sweep-to-training return edge.
        for infix in ("_cnn-", "_drawn-"):
            if infix in candidate:
                candidate = candidate.rsplit(infix, 1)[0]
                break
        else:
            if "." not in candidate:
                return None
            candidate = candidate.rsplit(".", 1)[0]
    return None


def _corrected_box(candidate: dict, decision: dict | None) -> dict:
    """the candidate box with a re-centre or a drag applied, in FRAME coordinates.

    THIS IS THE SEAM SMART CENTRE FELL THROUGH. A correction is stored crop-local - relative to
    the padded crop the browser was shown, whose origin is the candidate inset by MARGIN_X/
    MARGIN_Y - while everything downstream of a promote works in frame coordinates. Promote read
    `candidates[index]` directly, so a re-centred tile trained the net on the box the detector had
    originally guessed and the correction reached the pool's pixels but never the training set.

    Returns the candidate unchanged when there is no correction, which is the common case.
    """
    rect = (decision or {}).get("rect")
    if not rect:
        return candidate
    return {
        **candidate,
        "left": candidate["left"] - paths.MARGIN_X + int(rect["left"]),
        "top": candidate["top"] - paths.MARGIN_Y + int(rect["top"]),
        "width": int(rect["width"]),
        "height": int(rect["height"]),
    }


def _session_frames_dirs(root: Path, depth: int = 3) -> list[Path]:
    """every frames/ directory at or under `root`, nearest first.

    a recording normally holds its own frames/, but a set gathered for one purpose is a folder OF
    recordings - sessions/nameplate_pipeline_test/{1,2,3}/frames - and requiring frames/ directly
    inside each child meant the load tab silently listed nothing for it.
    """
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        _, frames_dir = resolve_session_paths(child)
        if frames_dir.is_dir():
            found.append(frames_dir)
        elif depth > 1:
            found.extend(_session_frames_dirs(child, depth - 1))
    return found


def _candidates_file_for(tag: str) -> Path | None:
    """the candidates file a POOL TAG came from, or None when it cannot be named for certain.

    A TILE'S TAG IS NOT ITS FILE'S STEM, and assuming so made this miss for every dataset but the
    bound one. The file keeps the run that produced it - `camera_calibration__2026-09-06.drawn-
    20260906-1947.candidates.jsonl` - while the tile is tagged `camera_calibration__2026-09-06`,
    the suffix dropped. A cnn sweep keeps its suffix but flattens the dot:
    `..._cnn-20260903-234900` against `....cnn-20260903-234900.candidates.jsonl`.

    So: try the literal name, then a flattened match, then the run-suffixed forms. AMBIGUITY IS
    REFUSED rather than guessed - two runs over one recording are two different candidate lists,
    and picking the wrong one writes a rect onto an unrelated plate.
    """
    if not tag:
        return None
    exact = paths.LABELS_DIR / f"{tag}.candidates.jsonl"
    if exact.is_file():
        return exact

    matches = []
    for path in paths.LABELS_DIR.glob("*.candidates.jsonl"):
        stem = path.name[: -len(".candidates.jsonl")]
        # the pool flattens a dot to an underscore, and may drop the run suffix altogether
        if stem.replace(".", "_") == tag or stem.split(".", 1)[0] == tag:
            matches.append(path)
    if len(matches) == 1:
        return matches[0]
    return None


class ReviewState:
    def __init__(
        self,
        frames_dir: Path,
        candidates: list[dict],
        library: Path,
        decisions_path: Path,
        session_tag: str = "",
        pool: Path | None = None,
    ):
        self.frames_dir = frames_dir
        self.candidates = candidates
        self.library = library
        # every session using the SAME library shares one unsorted staging pool below - without a
        # session-specific prefix, two sessions both saving candidate index 5 would both produce
        # "k00005.npz" and silently overwrite each other's kept tile the moment a second session
        # starts reviewing (e.g. classic_seeds and a second nameplate session both indexing from 0)
        self.session_tag = session_tag
        # the three roots the pickers browse, settable at runtime from the settings popup. holding
        # them per-instance rather than as module constants is what lets this tool point at a
        # different vision task's frames/labels/templates without a code change
        self.bases = dict(paths.DEFAULT_BASES)
        # the two the caller actually handed us win over the defaults. without this a test that
        # builds a state around its own tmp_path still reported - and, once repointed, WROTE to -
        # the real library and tile pool, which is the data-loss-behind-a-green-suite failure
        # paths.py's own docstring is about
        self.bases["templates"] = str(library)
        # how much ground a cut tile keeps around its box, per axis. settable from find's crop
        # menu; a change only reaches existing tiles through a re-cut, which is why that button
        # lives in the same menu
        self.pad_x = paths.PAD_X_DEFAULT
        self.pad_y = paths.PAD_Y_DEFAULT
        # per-tag (candidates, decisions, frames) for tiles belonging to other datasets
        self._dataset_cache: dict = {}
        # kept candidates are no longer labelled by colour at Keep time - Balthazar Fitzpatrick: "prepare frames
        # and tiles that will match plates, no matter what colour they are first." they go into
        # a separate staging pool, auto-named, so this fast unlabelled pass doesn't pollute the
        # real 5-colour library that guess_best_template matches new candidates against - that
        # still needs to stay exactly the confirmed seeds, not a growing pile of unsorted tiles.
        #
        # A POOL IS A DIRECTORY, NOT A SUFFIX. This used to be `<library>_unsorted`, derived by
        # appending to the library's name - which is why looking at a generated set needed a
        # directory literally called `plate_templates_data_synth_unsorted`, and why repointing the
        # pool silently repointed the library with it. It is now picked directly.
        # SEVERAL DIRECTORIES, PRIMARY FIRST. The grid lists every tile across all of them, so a
        # generated set can be reviewed beside the real one without switching pools and without
        # copying anything. Each keeps its OWN _labels.json, which is what stops 10,000 generated
        # judgements from landing in the tracked one - see pool_records.
        self.pools: list[Path] = [pool if pool is not None else library.parent / "tiles"]
        self.bases["pool"] = [str(self.pools[0])]
        # where a record was last read from, so a write goes back to the pool it came from even if
        # its tile has since gone. without this an unmounted pool's records migrate to the primary
        self._record_origin: dict[str, Path] = {}
        # where an un-kept tile goes instead of being deleted - "hard deleted is too much,
        # they should go into an archive I can offload manually eventually" (Balthazar Fitzpatrick).
        # BENEATH the pool rather than beside it: same kind of thing, different state, and the
        # underscore keeps it out of the way of a tag glob
        self.archive_dir = self.pools[0] / "_archive"
        self.decisions_path = decisions_path
        self.decisions: dict[str, dict] = self._load_decisions()
        self.lock = threading.Lock()
        # ONE size for every candidate's rect/tile, not each guessing its own from whichever
        # template happened to match - Balthazar Fitzpatrick: "they are all different," and he's right, since a
        # per-row guess naturally varies a little between colours/mobs. width is treated as known
        # and fixed; height is the one dimension nudged live via +/- in the UI.
        self.uniform_width, self.uniform_height = self._default_uniform_size()

    def rebind(
        self, frames_dir: Path, candidates: list[dict], decisions_path: Path, session_tag: str
    ) -> None:
        """swap the active dataset in place, no restart - this is what the load tab's click
        actually does. library/unsorted_dir/archive_dir stay exactly as they were: every
        dataset's kept tiles are meant to converge into the SAME real template library,
        regardless of which session they were mined from.
        """
        with self.lock:
            self.frames_dir = frames_dir
            self.candidates = candidates
            self.decisions_path = decisions_path
            self.session_tag = session_tag
            self.decisions = self._load_decisions()

    def unbind(self) -> dict:
        """let go of the bound dataset. NOTHING ON DISK MOVES.

        This used to archive the recording's drawn boxes on the way out, on the theory that a pass
        you close is a pass you are done with. Wrong, and Balthazar Fitzpatrick hit it: closing a recording made
        its rectangles vanish from find, which reads as losing the work rather than putting it
        away. Drawn boxes are saved explicitly with the save button, and closing is not a save and
        not a delete - it is only "stop showing me this one".

        Archiving a drawn set is still worth having, but as its own deliberate action, not a side
        effect of closing.
        """
        self.rebind(paths.SESSIONS_DIR, [], self.library / "_unbound.decisions.json", "")
        return {"ok": True, "archived": []}

    def draw_frames(self, sample: int = 40, percent: int | None = None) -> dict:
        """the frames to draw boxes on, evenly spread through the bound recording.

        THE COLD START. with an empty library the matcher refuses outright, and manual-crops mode
        wants a folder where each image is ALREADY a crop of one plate - so nothing took a fresh
        recording to a first tile, and the existing library was cut by hand outside the tool.
        this is that missing step: draw a few boxes on whole frames, and they become candidates the
        tune tab turns into tiles.

        evenly spread rather than random - at 2 Hz neighbouring frames are near-duplicates, so a
        random draw spends the budget on the same picture twice.
        """
        if not self.frames_dir.is_dir():
            return {
                "ready": False,
                "reason": "bind a recording in the load tab first",
                "frames": [],
            }
        found = sorted(self.frames_dir.glob("*.jpg")) + sorted(self.frames_dir.glob("*.png"))
        if not found:
            return {"ready": False, "reason": f"no frames under {self.frames_dir}", "frames": []}
        # a PERCENTAGE of the recording rather than a fixed count, so the same choice means the
        # same coverage on a 40-frame run and a 400-frame one. evenly spread, not random: at 2 Hz
        # neighbouring frames are near-duplicates and a random draw spends the budget twice on one
        # picture
        if percent:
            sample = max(1, round(len(found) * min(100, max(1, percent)) / 100))
        if sample and 0 < sample < len(found):
            step = len(found) / sample
            found = [found[int(i * step)] for i in range(sample)]
        names = [p.name for p in found]

        # BOXES ALREADY DRAWN COME BACK. they are saved to the candidates file on every save, but
        # the page held them only in memory - so reopening find showed a blank canvas over frames
        # that had already been worked, and the only way to tell was to remember. any frame in the
        # bound set that is not in the sample is carried too, so its boxes are not lost on re-save
        # A RECORDING AND ITS DRAWN SET ARE THE SAME WORK. binding the raw recording used to load
        # no boxes, so a fresh pass looked like a SHRINK against the set already on disk and the
        # save wrote a new timestamped file instead of updating - which is how one recording ended
        # up with several part-sets and the pool with the same tiles three times over
        candidates = self.candidates
        if not candidates:
            base = _flat(
                str(self.frames_dir.parent.relative_to(paths.SESSIONS_DIR))
                if paths.SESSIONS_DIR.resolve() in self.frames_dir.resolve().parents
                else self.frames_dir.parent.name
            )
            found = sorted(Path(self.bases["labels"]).glob(f"{base}.drawn-*.candidates.jsonl"))
            if found:
                candidates = [
                    json.loads(line) for line in found[0].read_text().splitlines() if line.strip()
                ]

        existing: dict[str, list[dict]] = {}
        negatives: dict[str, list[dict]] = {}
        for candidate in candidates:
            box = {
                "left": candidate["left"],
                "top": candidate["top"],
                "width": candidate["width"],
                "height": candidate["height"],
            }
            # SPLIT ON REOPEN, or a saved negative comes back as a positive and the next save
            # doubles the set. the flag is written by save_drawn and is the only thing that says
            # which a stored box is
            target = negatives if candidate.get("negative") else existing
            target.setdefault(candidate["path"], []).append(box)
        for name in existing:
            if name not in names:
                names.append(name)
        return {
            "ready": True,
            "reason": "",
            "frames": sorted(names),
            "boxes": existing,
            "negatives": negatives,
            # which recording this is, since find no longer carries a path row that said so
            "dataset": self.session_tag or "",
            "recording": (
                str(self.frames_dir.parent.relative_to(paths.SESSIONS_DIR))
                if paths.SESSIONS_DIR.resolve() in self.frames_dir.resolve().parents
                else self.frames_dir.parent.name
            ),
            "drawn": sum(len(v) for v in existing.values()),
        }

    def frame_bytes(self, name: str) -> bytes | None:
        """one whole frame from the bound recording, for drawing on"""
        candidate = (self.frames_dir / Path(name).name).resolve()
        if self.frames_dir.resolve() not in candidate.parents or not candidate.is_file():
            return None
        return candidate.read_bytes()

    def save_drawn(
        self, boxes: list[dict], mode: str = "drawn", negatives: list[dict] | None = None
    ) -> dict:
        """turn drawn boxes into a candidates file and bind it, so tune picks up immediately.

        THE WIDTH IS FITTED, NOT TAKEN. Balthazar Fitzpatrick: "I will mark rectangles but then you fit the tile
        based on width that is desirable." a hand-drawn box is a few pixels out in each direction
        and that jitter would become the tile's size; the MEDIAN width and height across every
        box in the set is a far better estimate of the real plate than any single one of them, so
        each box keeps its position and takes the set's size.
        """
        if not boxes:
            return {"error": "no boxes drawn"}
        widths = sorted(int(b["width"]) for b in boxes)
        heights = sorted(int(b["height"]) for b in boxes)
        width = widths[len(widths) // 2]
        height = heights[len(heights) // 2]

        candidates = [
            {
                "path": Path(b["path"]).name,
                "left": int(b["left"]),
                "top": int(b["top"]),
                "width": width,
                "height": height,
                "matched_template": "drawn",
                "score": 1.0,  # a human drew it; nothing scored it
            }
            for b in boxes
        ]
        # NEGATIVES COME FROM THE PAGE NOW, one per positive, placed by the tool and correctable by
        # hand before they are saved. promote_to_training used to invent them at the last moment,
        # so nobody ever saw one - and nothing stopped one landing on the player or target frame,
        # which look enough like a nameplate that the detector learned to fire on them. They take
        # the fitted size for the same reason the positives do.
        candidates += [
            {
                "path": Path(b["path"]).name,
                "left": int(b["left"]),
                "top": int(b["top"]),
                "width": width,
                "height": height,
                "matched_template": "drawn",
                "score": 1.0,
                "negative": True,
            }
            for b in (negatives or [])
        ]
        # REFUSE RATHER THAN WRITE AN ORPHAN. this used to fall back to the literal name
        # "unbound", which produced a file promote could never resolve back to a recording - the
        # boxes are real work and looked saved, but nothing downstream could use them
        if not self.session_tag:
            return {"error": "no dataset bound - pick one before saving"}
        base = _flat(self.session_tag.split(".")[0])
        labels_root = Path(self.bases["labels"])

        # RE-SAVING A RECORDING UPDATES ITS SET, it does not add another. a stamped sibling per
        # save is right for MINED candidates - their decisions are keyed by candidate index, so
        # rewriting the file would re-point every keep/discard at a different candidate - but a
        # drawn set carries decisions this same method writes, so the pair stays consistent and
        # the load list stops filling with duplicates of one recording
        existing = sorted(labels_root.glob(f"{base}.{mode}-*.candidates.jsonl"))
        if existing:
            out = existing[0]
            # A SAVE MUST NEVER SILENTLY SHRINK A SET. updating in place stops the load list
            # filling with duplicates, but it rewrites the file with whatever the PAGE holds - and
            # a page bound to the raw recording holds nothing. that turned 94 hand-drawn boxes
            # into 1, with no warning and no way back, because labels/ is gitignored. a smaller
            # save now goes to a new stamped file instead of over the old one
            held = sum(1 for line in out.read_text().splitlines() if line.strip())
            if len(candidates) < held:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                out = labels_root / f"{base}.{mode}-{stamp}.candidates.jsonl"
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            out = labels_root / f"{base}.{mode}-{stamp}.candidates.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(c) + "\n" for c in candidates))

        # A DRAWN BOX IS ALREADY CONFIRMED, and the dataset builder has to be told so. it only
        # counts a candidate as a plate when its decision says keep; everything else - discarded OR
        # never ruled on - becomes an IGNORE region, because a mined candidate nobody judged is
        # unknowable. that is right for machine proposals and wrong for these: without a decisions
        # file every one of these boxes landed in ignore, the dataset had zero centres, and
        # training raised "no example has a confirmed plate". drawing could not reach a model at all
        decisions = {
            str(index): {"keep": True, "origin": "drawn"} for index in range(len(candidates))
        }
        decisions_path = out.with_suffix(".decisions.json")
        decisions_path.write_text(json.dumps(decisions, indent=1))

        # CUT EVERY BOX INTO THE UNSORTED POOL RIGHT AWAY. tune's keep/discard exists to judge
        # MACHINE proposals; a box Balthazar Fitzpatrick drew is already ground truth, so the review is wasted
        # motion - but the mechanical half of tune, cutting the crop, still has to happen for
        # cluster to have anything to group. doing it here is what lets draw -> cluster -> train
        # skip tune entirely while leaving every clustering option exactly as it was
        saved = self._cut_drawn(candidates, width, height)

        self.rebind(self.frames_dir, candidates, decisions_path, out.stem)
        return {
            "out": str(out),
            # POSITIVES ONLY. the page prints this as "N positives", and once negatives joined
            # the same candidates list len(candidates) quietly doubled it
            "count": len(boxes),
            "negatives": len(negatives or []),
            "width": width,
            "height": height,
            "tiles": saved,
            "spread": {"width": [widths[0], widths[-1]], "height": [heights[0], heights[-1]]},
        }

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
        """one tile per box, straight into the unsorted pool.

        start_index offsets the names, so re-cutting ONE corrected box writes over the tile it
        already had rather than minting a fresh _k00000 beside it.

        select cuts only SOME of the candidates while naming them by their position in the whole
        list. the sweep's send is the caller: it takes a threshold's worth of proposals out of a
        list that was written to disk whole, and a tile is resolved back to its box by the index
        in its name - so handing this the filtered list instead would name the 40 kept proposals
        k00000..k00039 against a file whose rows 0-39 are entirely different boxes.

        frames_dir and tag are for the CNN SWEEP, which cuts from a recording other than the bound
        one and wants its own tag so a batch of proposals can be closed separately from the drawn
        tiles it sits beside in the pool.

        opens each FRAME once rather than once per box - 298 boxes over 107 frames at 1.6 MB each
        is a re-read worth avoiding. a box whose fitted size runs off the frame edge is skipped
        rather than raising: the fitted size is a median, so a plate drawn near the edge can
        legitimately overhang, and losing that one crop is better than losing the whole save.
        """

        # PADDED, and re-derivable: the frames are on disk and so are the rectangles, so the pad
        # is never baked into anything that can't be cut again. a drawn box stops at the plate,
        # which leaves a tile with no ground around it - the matcher then has nothing to lock
        # onto but the plate's own colours, and the tile is an unreadable bar to look at
        pad_x = round(width * self.pad_x)
        pad_y = round(height * self.pad_y)
        self.unsorted_dir.mkdir(parents=True, exist_ok=True)
        source_dir = frames_dir or self.frames_dir
        recording = tag or _flat(
            str(source_dir.parent.relative_to(paths.SESSIONS_DIR))
            if paths.SESSIONS_DIR.resolve() in source_dir.resolve().parents
            else source_dir.parent.name
        )
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
                with Image.open(path) as handle:
                    frame = np.asarray(handle.convert("RGB"))
            except (OSError, ValueError):
                # a truncated or unreadable frame costs its own crops, never the whole save - the
                # candidates file is already written by this point and is the thing worth keeping
                continue
            for index in indices:
                candidate = candidates[index]
                try:
                    template = crop_at_rect(
                        frame,
                        # NAMED AFTER THE RECORDING, not the dataset. session_tag carries the
                        # candidates file's timestamp, so every save minted a fresh set of names
                        # and nothing ever replaced anything - 235 boxes had become 679 tiles
                        # across eight tags, all of them the same work saved repeatedly
                        f"{recording}_k{start_index + index:05d}",
                        top=int(candidate["top"]) - pad_y,
                        left=int(candidate["left"]) - pad_x,
                        height=height + pad_y * 2,
                        width=width + pad_x * 2,
                    )
                except TemplateError:
                    # a plate near the frame edge cannot carry its pad - keep the bare box rather
                    # than lose the crop, since an unpadded tile still trains
                    try:
                        template = crop_at_rect(
                            frame,
                            f"{recording}_k{start_index + index:05d}",
                            top=int(candidate["top"]),
                            left=int(candidate["left"]),
                            height=height,
                            width=width,
                        )
                    except TemplateError:
                        continue
                save_template(template, self.unsorted_dir)
                saved += 1
        return saved

    def dir_list(self) -> dict:
        """what the three pickers offer, each scoped to its own base so a path can never be typed
        wrong: sessions are directories holding frames, labels are candidate files, templates are
        any directory holding .npz. inferred_k comes back with them because the number of distinct
        identities the library already knows IS the sensible cluster count - asking for it is
        asking the user to restate something the data already says."""
        sessions_root, labels_root = Path(self.bases["sessions"]), Path(self.bases["labels"])
        templates_root = Path(self.bases["templates"])

        # SAME RECURSION AS THE LOAD TAB. this listed only the direct children of sessions/, so a
        # recording nested inside a set was missing from the picker even while the find tab's own
        # header said it was reading from exactly that path - the load tab could bind it and this
        # could not offer it, which reads as the picker being broken
        sessions = [d.parent.as_posix() for d in _session_frames_dirs(sessions_root)]

        labels = []
        if labels_root.is_dir():
            labels = [p.as_posix() for p in sorted(labels_root.glob("*.candidates.jsonl"))]

        templates = []
        if templates_root.is_dir():
            for path in sorted(p for p in templates_root.iterdir() if p.is_dir()):
                count = len(list(path.glob("*.npz")))
                if count:
                    templates.append({"path": path.as_posix(), "count": count})

        return {
            "bases": self.bases,
            "sessions": sessions,
            "labels": labels,
            "templates": templates,
            "inferred_k": self.inferred_k(),
        }

    @staticmethod
    def _suggestions_for(argument: str) -> list[str]:
        """values that already exist on disk for an argument that names one.

        argparse knows an argument's type and its choices, but a character or a screen profile is
        not a fixed set - it is whatever has been created. Typing one from memory is how a run
        fails on a name that differs by a hyphen, so the builder offers what is actually there.
        """
        from parent.config.character import CHARACTERS_DIR
        from parent.config.profile import PROFILES_DIR

        name = argument.lstrip("-")
        if name == "character":
            root = CHARACTERS_DIR
            return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if name == "profile":
            root = PROFILES_DIR
            return sorted(p.stem for p in root.glob("*.toml")) if root.is_dir() else []
        return []

    def script_list(self) -> dict:
        """every wt- entry point with the arguments it declares, read from pyproject and each
        tool's own argparse. read, never RUN from here - the scripts tab exists to compose a
        correct command line, and a tool that records input or drives the game must be started
        deliberately in a terminal, not by a stray click in a browser."""
        import ast

        rows = []
        pyproject = Path("pyproject.toml")
        if not pyproject.is_file():
            return {"scripts": rows}
        in_scripts = False
        for line in pyproject.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_scripts = stripped == "[project.scripts]"
                continue
            if not in_scripts or "=" not in stripped:
                continue
            name, target = (part.strip().strip('"') for part in stripped.split("=", 1))
            module = target.split(":")[0]
            path = Path(module.replace(".", "/") + ".py")
            summary, args = "", []
            if path.is_file():
                try:
                    tree = ast.parse(path.read_text())
                    doc = ast.get_docstring(tree) or ""
                    summary = doc.splitlines()[0] if doc else ""
                    # add_argument's first literal is the flag or positional name, and its help=
                    # is the one-line explanation the author already wrote. carrying it through
                    # means the ui can explain an argument without anyone documenting it twice
                    for node in ast.walk(tree):
                        if (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "add_argument"
                            and node.args
                            and isinstance(node.args[0], ast.Constant)
                        ):
                            explain = ""
                            # required=True, or a POSITIONAL (a name with no leading dash) -
                            # argparse already knows which arguments a run cannot omit, so the ui
                            # should not make someone read the help to find out
                            required = not str(node.args[0].value).startswith("-")
                            # WHETHER IT TAKES A VALUE AT ALL. store_true/store_false are flags -
                            # a builder that renders "--dump <value>" produces a command that will
                            # not run. argparse knows; this now reads it out of the same call.
                            takes_value = True
                            default = choices = kind = None
                            for keyword in node.keywords:
                                if keyword.arg == "required" and isinstance(
                                    keyword.value, ast.Constant
                                ):
                                    required = bool(keyword.value.value)
                                if keyword.arg == "action" and isinstance(
                                    keyword.value, ast.Constant
                                ):
                                    takes_value = "store_" not in str(keyword.value.value)
                                if keyword.arg == "default" and isinstance(
                                    keyword.value, ast.Constant
                                ):
                                    default = keyword.value.value
                                if keyword.arg == "type" and isinstance(keyword.value, ast.Name):
                                    kind = keyword.value.id
                                if keyword.arg == "choices" and isinstance(
                                    keyword.value, ast.List | ast.Tuple
                                ):
                                    choices = [
                                        v.value
                                        for v in keyword.value.elts
                                        if isinstance(v, ast.Constant)
                                    ]
                                if keyword.arg == "help" and isinstance(
                                    keyword.value, ast.Constant
                                ):
                                    explain = str(keyword.value.value)
                                elif keyword.arg == "help" and isinstance(
                                    keyword.value, ast.JoinedStr
                                ):
                                    # an f-string help; take the literal parts, drop the slots
                                    explain = "".join(
                                        v.value
                                        for v in keyword.value.values
                                        if isinstance(v, ast.Constant)
                                    )
                            args.append(
                                {
                                    "name": str(node.args[0].value),
                                    "help": explain,
                                    "required": required,
                                    "takes_value": takes_value,
                                    "default": default,
                                    "type": kind,
                                    "choices": choices,
                                    # not choices: argparse does not know these, the filesystem does
                                    "suggests": self._suggestions_for(str(node.args[0].value)),
                                }
                            )
                except SyntaxError:
                    pass
            category, subcategory = SCRIPT_GROUPS.get(name, ("other", "unsorted"))
            rows.append(
                {
                    "name": name,
                    "module": module,
                    "summary": summary,
                    "args": args,
                    "category": category,
                    "subcategory": subcategory,
                    "user_rank": USER_CALLED.index(name) if name in USER_CALLED else None,
                }
            )
        return {"scripts": rows}

    def inferred_k(self) -> int:
        """cluster count, derived rather than asked: the library's own template count is how many
        distinct identities this vocabulary can express, which is the meaningful number of buckets
        to sort an unlabelled pool into. falls back to the pool size when the library is empty."""
        try:
            n = len(load_templates(self.library))
        except TemplateError:
            n = 0
        if not n:
            n = len(self.tile_paths()) or 1
        return max(1, min(30, n))

    def dir_tree(self, under: str | None = None) -> dict:
        """the directories one level under a path, for the settings tree to walk.

        ONE LEVEL AT A TIME, not the whole tree: `sessions/` alone holds 21 recordings and the
        vision directory holds a pool of 10,000 tiles, so a recursive listing would be enormous and
        almost entirely uninteresting. `has_children` is what lets a row show it can be opened
        without paying to find out.

        REFUSES TO LEAVE THE ROOT. A path outside it comes back as the root's own listing rather
        than an error, because this is a browser and the honest answer to "show me /etc" here is to
        show what it is allowed to show.
        """
        root = Path(self.bases["root"]).resolve()
        try:
            here = Path(under).resolve() if under else root
        except (OSError, ValueError):
            here = root
        if not (here == root or root in here.parents) or not here.is_dir():
            here = root

        entries = []
        with contextlib.suppress(OSError):
            for child in sorted(here.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                has_children = False
                with contextlib.suppress(OSError):
                    has_children = any(
                        g.is_dir() and not g.name.startswith(".") for g in child.iterdir()
                    )
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "has_children": has_children,
                    }
                )
        return {
            "root": str(root),
            "here": str(here),
            "parent": str(here.parent) if here != root else None,
            "entries": entries,
        }

    def set_bases(self, bases: dict) -> dict:
        """repoint the pickers. only keys we know are accepted, and nothing is validated into
        existence here - a base that does not exist simply offers nothing, which is visible in the
        picker rather than failing later at bind time"""
        for key in paths.BASE_KEYS:
            value = bases.get(key)
            if key in paths.MULTI_BASES:
                # SEVERAL PATHS, and a bare string is still accepted - a review_bases.json written
                # before this, or any caller that only has one, must keep working
                wanted = [value] if isinstance(value, str) else value
                if isinstance(wanted, list):
                    cleaned = [v.strip() for v in wanted if isinstance(v, str) and v.strip()]
                    if cleaned:
                        self.bases[key] = cleaned
            elif isinstance(value, str) and value.strip():
                self.bases[key] = value.strip()

        # THE POOL AND THE LIBRARY ARE A LIVE HANDLE, not just a string: repointing either has to
        # move the directories derived from it, or the setting only takes effect after a restart -
        # exactly how the old bases behaved before they were made to persist. They are also two
        # different things, which the suffix scheme hid: `pool` is the tiles being judged, and
        # `templates` is the confirmed library plate_track matches against. Repointing the pool
        # used to silently repoint the library too.
        wanted = [Path(v) for v in self.bases["pool"]]
        if wanted and wanted != self.pools:
            self.pools = wanted
            self.archive_dir = self.pools[0] / "_archive"
            self._dataset_cache.clear()
            self._record_origin.clear()
        self.library = Path(self.bases["templates"])
        # ASSIGNED ON THE MODULE, which is the whole reason paths.py exists. every reader reaches
        # through `paths.`, so repointing here is seen everywhere at once - a module that had
        # imported these by value would go on using the old directory with nothing to say so
        paths.SESSIONS_DIR = Path(self.bases["sessions"])
        paths.LABELS_DIR = Path(self.bases["labels"])
        self._save_bases()
        return self.bases

    # WHERE A REPOINT SURVIVES A RESTART. It used to live only in the running process, so every
    # restart silently reverted to the defaults - and a promote run afterwards resolved nothing and
    # overwrote a good training set with an empty one. The setting is a decision about this
    # machine's layout, not about one session.
    BASES_FILE = paths.REVIEW_UI_DIR.parent / "review_bases.json"

    def _save_bases(self) -> None:
        # a repoint that cannot be remembered is still a working repoint, so a write failure here
        # must not break the request that made it
        with contextlib.suppress(OSError):
            self.BASES_FILE.write_text(json.dumps(self.bases, indent=2))

    def load_saved_bases(self) -> None:
        """apply the remembered repoint, if there is one. called once at startup."""
        try:
            saved = json.loads(self.BASES_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(saved, dict):
            # through set_bases, so the module paths move with it - and its own save is harmless
            self.set_bases(saved)

    def crop_settings(self) -> dict:
        """the pad per axis, and what it means for a box of the size actually being drawn.

        the preview matters: a fraction is not something anyone can picture, and the whole reason
        this is two numbers is that one number produced wildly different vertical padding depending
        on box shape.
        """
        width, height = self.uniform_width, self.uniform_height
        return {
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "box": [width, height],
            "crop": [
                width + 2 * round(width * self.pad_x),
                height + 2 * round(height * self.pad_y),
            ],
        }

    def set_crop_settings(self, pad_x: float | None, pad_y: float | None) -> dict:
        """CHANGING THESE DOES NOT MOVE ANY PIXELS. tiles already cut keep the pad they were cut
        with until a re-cut, so the setting and the re-cut button sit together in one menu."""
        if pad_x is not None:
            self.pad_x = max(0.0, min(3.0, float(pad_x)))
        if pad_y is not None:
            self.pad_y = max(0.0, min(3.0, float(pad_y)))
        return self.crop_settings()

    def bindable_recordings(self) -> dict:
        """what find can open: EVERY recording under sessions/, plus the cnn's proposal sets.

        TWO KINDS, AND EVERY RECORDING IS ALWAYS LISTED. this used to skip any session that already
        had a candidates file, on the theory that the dataset row replaced it - which meant that the
        moment you drew a single box on a recording, the recording itself vanished from the picker
        and could never be reopened. With three recordings all drawn on, the list showed no sessions
        at all and nothing but cnn sweeps.

        Drawn sets are deliberately NOT listed as rows of their own. Binding the raw recording
        already loads its drawn boxes back (see draw_frames - "a recording and its drawn set are the
        same work"), and offering both as separate entries is exactly what produced one recording
        with several part-sets and the same tiles in the pool three times over. The drawn work
        shows up as a count on the recording's own row instead.

        cnn sets stay listed, as a second column, so a sweep's proposals can be inspected in find
        even though they are meant for select.
        """
        rows = []
        # DRAWN BOXES PER RECORDING, so the recording's own row can report the work already on it.
        # WHAT THE POOL SAYS ABOUT THIS RECORDING'S TILES, counted once for every row below rather
        # than re-read per row. `decided`/`kept` beside it are the FIND tab's keep/discard on boxes,
        # a different stage - these are the discard/promote judgements, and the two together are
        # what "how far along is this recording" actually means
        classed: dict[str, int] = {}
        negative: dict[str, int] = {}
        for tile_name, record in self.pool_records().items():
            bucket = (
                negative if record.get("excluded") else (classed if record.get("label") else None)
            )
            if bucket is not None:
                bucket[_tile_tag(tile_name)] = bucket.get(_tile_tag(tile_name), 0) + 1

        # keyed by the session name the drawn file came from, un-flattened back to its real path
        drawn: dict[str, dict] = {}
        cnn_rows = []
        if paths.LABELS_DIR.is_dir():
            for path in sorted(paths.LABELS_DIR.glob("*.candidates.jsonl")):
                tag = path.stem.removesuffix(".candidates")
                lines = [line for line in path.read_text().splitlines() if line.strip()]
                decisions_path = path.with_suffix(".decisions.json")
                decided = kept = 0
                if decisions_path.exists():
                    records = json.loads(decisions_path.read_text()).values()
                    decided = sum(1 for r in records if r.get("keep") or r.get("discard"))
                    kept = sum(1 for r in records if r.get("keep"))
                if ".cnn-" in tag:
                    # a sweep that proposed nothing is not worth offering - opening it binds an
                    # empty set and the page simply does not change, which reads as a broken picker
                    if lines:
                        cnn_rows.append(
                            {
                                "kind": "dataset",
                                "name": tag,
                                "frame_count": len(lines),
                                "decided": decided,
                                "kept": kept,
                                "classed": classed.get(tag.replace(".cnn-", "_cnn-"), 0),
                                "negative": negative.get(tag.replace(".cnn-", "_cnn-"), 0),
                            }
                        )
                    continue
                # a drawn set belongs to its recording rather than standing on its own
                session = tag.split(".drawn-")[0].replace("__", "/")
                previous = drawn.get(session, {"boxes": 0, "decided": 0, "kept": 0})
                drawn[session] = {
                    "boxes": previous["boxes"] + len(lines),
                    "decided": previous["decided"] + decided,
                    "kept": previous["kept"] + kept,
                }
        if paths.SESSIONS_DIR.is_dir():
            # NESTED RECORDINGS COUNT. this used to require frames/ DIRECTLY inside each child of
            # sessions/, so a folder OF recordings - sessions/nameplate_pipeline_test/{1,2,3}/frames,
            # which is how a set gathered for one purpose is laid out - was skipped silently and the
            # load tab showed nothing. each recording is listed under its path from sessions/ so the
            # three read as three rows rather than collapsing to one ambiguous name
            for frames_dir in _session_frames_dirs(paths.SESSIONS_DIR):
                name = str(frames_dir.parent.relative_to(paths.SESSIONS_DIR))
                frame_count = len(list(frames_dir.glob("*.jpg"))) + len(
                    list(frames_dir.glob("*.png"))
                )
                work = drawn.get(name)
                rows.append(
                    {
                        "kind": "session",
                        "name": name,
                        "frame_count": frame_count,
                        "boxes": work["boxes"] if work else 0,
                        "decided": work["decided"] if work else None,
                        "kept": work["kept"] if work else None,
                        # the pool's tag for a drawn set is the recording name, flattened
                        "classed": classed.get(name.replace("/", "__"), 0),
                        "negative": negative.get(name.replace("/", "__"), 0),
                    }
                )
        rows.extend(cnn_rows)
        # which row is live right now - without this the load tab lists datasets with no way to
        # tell which one the other tabs are actually showing
        return {"rows": rows, "current": self.session_tag}

    def _default_uniform_size(self) -> tuple[int, int]:
        """width: the median across whatever's already in the library - a real measurement
        ("a nameplate we know"). height: fixed at 10 - Balthazar Fitzpatrick's own observed sweet spot after
        using the per-row +/- correction across many real candidates (was drifting 9-10 as a
        computed median, which isn't a meaningful "default" so much as noise around this value)"""
        try:
            templates = load_templates(self.library)
            widths = sorted(t.width for t in templates)
            width = widths[len(widths) // 2]
        except TemplateError:
            width = 64
        return width, 10

    def _load_decisions(self) -> dict[str, dict]:
        if self.decisions_path.exists():
            return json.loads(self.decisions_path.read_text())
        return {}

    def _save_decisions(self) -> None:
        self.decisions_path.write_text(json.dumps(self.decisions, indent=0))

    def _archive_template(self, name: str) -> None:
        """moves an un-kept tile out of the live unsorted pool instead of deleting it - a
        real delete_template() call here (the old behaviour) is what actually destroyed
        k00016/k00071 for good before the Clusters exclude button was made reversible."""
        path = self.tile_path(name)
        if path is None:
            return
        # INTO ITS OWN POOL'S ARCHIVE, not the primary's - moving a generated tile into the real
        # pool's archive would quietly mix the two sets
        archive = path.parent / "_archive"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archive / f"{name}.npz"))

    def _frame(self, path: str, frames_dir=None):
        """frames_dir defaults to the bound dataset's, but a thumbnail for a tile from ANOTHER
        dataset has to read that dataset's frames instead - see _dataset_for"""
        import numpy as np

        with Image.open((frames_dir or self.frames_dir) / path) as handle:
            return np.asarray(handle.convert("RGB"))

    def is_pending(self, index: int) -> bool:
        decision = self.decisions.get(str(index), {})
        return not decision.get("keep") and not decision.get("discard")

    def _visible_indices(self, hide_done: bool = False) -> list[int]:
        """every candidate except ones explicitly marked reviewed - reviewed is a SEPARATE flag
        from keep/discard: a decided row still shows up (dimmed) when paging back so it can be
        adjusted, right up until Balthazar Fitzpatrick actively marks it reviewed to say "done looking at this
        one," at which point it drops out of paging entirely, forward or back.

        hide_done additionally drops anything already keep/discard'd - the "don't show done"
        toggle, for when the goal is purely finding fresh work rather than reviewing.
        """
        return [
            i
            for i in range(len(self.candidates))
            if not self.decisions.get(str(i), {}).get("reviewed")
            and (not hide_done or self.is_pending(i))
        ]

    def mark_reviewed(self, index: int) -> None:
        with self.lock:
            key = str(index)
            record = self.decisions.get(key, {"keep": False, "discard": False, "saved_as": None})
            record["reviewed"] = True
            self.decisions[key] = record
            self._save_decisions()

    def page(self, page: int, size: int, hide_done: bool = False) -> dict:
        """stable pagination over VISIBLE candidates (everything not marked reviewed) in their
        original order - paging back always lands on the same page it did before, decided rows
        included and marked done, so a decision is never mistaken for something that got lost.
        finding NEW work to do is a separate concern: either the frontend's forward-only
        auto-skip past fully-decided pages, or the explicit "hide done" toggle which drops
        decided rows from paging entirely, in both directions, for the duration it's on."""
        visible = self._visible_indices(hide_done)
        start, end = page * size, page * size + size
        chunk_indices = visible[start:end]
        templates = (
            load_templates(self.library)
            if self.library.is_dir() and any(self.library.glob("*.npz"))
            else []
        )

        rows = []
        for index in chunk_indices:
            candidate = self.candidates[index]
            decision = self.decisions.get(str(index), {})
            guess = None
            if templates:
                try:
                    frame = self._frame(candidate["path"])
                    guess = guess_best_template(
                        frame, candidate["top"], candidate["left"], templates
                    )
                except (TemplateError, OSError):
                    guess = None
            guide_template = next((t for t in templates if guess and t.name == guess.name), None)
            rows.append(
                {
                    "index": index,
                    "path": candidate["path"],
                    "left": candidate["left"],
                    "top": candidate["top"],
                    "width": candidate["width"],
                    "height": candidate["height"],
                    "guess": guess.name if guess else None,
                    "guess_top": guess.top if guess else None,
                    "guess_left": guess.left if guess else None,
                    "guide_name": guide_template.name if guide_template else None,
                    # guide_width/height: the MIRRORED full-bar shape, display only (the overlay
                    # you drag against). kept_width/height: the template's own real size - what
                    # the extraction rect must actually be. these used to be conflated (the rect
                    # was sized to the mirrored width), which is why the tile preview showed
                    # the full bar instead of the 40% that actually gets saved
                    "guide_width": guide_template.width * 2 if guide_template else None,
                    "guide_height": guide_template.height if guide_template else None,
                    "kept_width": guide_template.width if guide_template else None,
                    "kept_height": guide_template.height if guide_template else None,
                    # per-candidate correction on top of the global default height - aliasing
                    # means the "right" height isn't perfectly identical frame to frame, off by
                    # a pixel here or there even when the global default is basically correct
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
            "page_done": bool(chunk_indices) and all(not self.is_pending(i) for i in chunk_indices),
        }

    def _sub_image(self, frame, candidate: dict):
        """the default box+margin crop, used only as a fallback when no alignment rect exists
        yet (e.g. before any drag) - the real path once a rect exists is _rect_template"""
        top, left = candidate["top"], candidate["left"]
        height, width = candidate["height"], candidate["width"]
        return frame[
            max(0, top - paths.MARGIN_Y) : top + height + paths.MARGIN_Y,
            max(0, left - paths.MARGIN_X) : left + width + paths.MARGIN_X,
        ]

    def _padded_frame(self, candidate: dict, frames_dir: Path | None = None):
        """the crop-with-margin, ALWAYS exactly (width+2*paths.MARGIN_X, height+2*paths.MARGIN_Y) - padded
        with black where the real frame doesn't reach, never clamped to a smaller size.

        this is what makes the displayed crop and the extracted rect agree on where (0,0) is. a
        clamped crop (the old behaviour) silently returns a SMALLER image whenever the candidate
        sits near a source image's edge - normal captures have 2560px of real margin to spare, so
        this never showed up there, but a standalone seed crop IS barely bigger than the bar
        itself, so paths.MARGIN_X/Y routinely exceeds what's actually available. the frontend still
        assumed the full margin was there, and the extraction math used a DIFFERENT, unclamped
        origin than what was actually displayed - two different coordinate systems silently
        disagreeing, which is what produced rectangles with no sane relationship to the picture.
        """
        import numpy as np

        with Image.open((frames_dir or self.frames_dir) / candidate["path"]) as handle:
            frame = np.asarray(handle.convert("RGB"))
        fh, fw = frame.shape[:2]

        pad_h = candidate["height"] + 2 * paths.MARGIN_Y
        pad_w = candidate["width"] + 2 * paths.MARGIN_X
        origin_top = candidate["top"] - paths.MARGIN_Y
        origin_left = candidate["left"] - paths.MARGIN_X

        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        src_top, src_left = max(0, origin_top), max(0, origin_left)
        src_bottom, src_right = min(fh, origin_top + pad_h), min(fw, origin_left + pad_w)
        dest_top, dest_left = src_top - origin_top, src_left - origin_left
        dest_bottom = dest_top + (src_bottom - src_top)
        dest_right = dest_left + (src_right - src_left)
        canvas[dest_top:dest_bottom, dest_left:dest_right] = frame[
            src_top:src_bottom, src_left:src_right
        ]
        return canvas

    def crop_bytes(self, index: int) -> bytes:

        canvas = self._padded_frame(self.candidates[index])
        buf = BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        return buf.getvalue()

    def guide_bytes(self, template_name: str) -> bytes:
        """the mirrored full-bar alignment guide for one library template, as a transparent PNG.

        magenta at the masked corners - the rectangle's own outline is drawn separately, as a
        dashed CSS outline on the <img> itself, since a dashed line baked into a handful of
        source pixels would just get stretched into a solid blur once the browser scales it up.
        """
        import numpy as np

        template = load_template(self.library / f"{template_name}.npz")
        full_mask = mirrored_guide_mask(template)
        rgba = np.zeros((*full_mask.shape, 4), dtype=np.uint8)
        rgba[~full_mask] = [255, 41, 214, 160]
        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def _effective_height(self, index: int) -> int:
        """the global default height, plus this ONE candidate's own correction - aliasing means
        the true height isn't perfectly identical frame to frame, off by a pixel here or there
        even when the global default is basically right"""
        delta = self.decisions.get(str(index), {}).get("height_delta", 0)
        return max(4, self.uniform_height + delta)

    def set_height_delta(self, index: int, delta: int) -> int:
        with self.lock:
            key = str(index)
            record = self.decisions.get(key, {"keep": False, "discard": False, "saved_as": None})
            record["height_delta"] = record.get("height_delta", 0) + delta
            self.decisions[key] = record
            self._save_decisions()
            return record["height_delta"]

    def smart_center(self, names: list[str]) -> dict:
        """re-cut the named tiles with the plate centred, and report what moved.

        THE HUMAN HAS ALREADY SAID THESE ARE PLATES by not marking them "not a class", so the
        locator here never asks whether there is one - only where it is. See vision/recentre: a
        thresholding locator recovered 7 of 60 nudged boxes on real frames, the ranking one 113 of
        113. Nothing is judged here, only moved.

        NON-DESTRUCTIVE IN THE SENSE THAT MATTERS: the position lands in the decision's `rect`,
        which is the same field a manual drag writes, so a later drag overrides it and re-cutting
        from the frame is always possible. The tile npz is rewritten because that is the training
        pixels - which is the point of the button.

        EVERY TILE IS RESOLVED TO ITS OWN DATASET, because the unsorted pool is library-wide and
        spans them - eight at the time of writing. A tile's name carries its tag and its index, and
        the index means nothing without the tag: reading `k00042` against whichever candidates list
        happens to be bound picks an unrelated plate in an unrelated frame, which is the trap
        `_dataset_for` exists for. So the tag decides which candidates, which decisions file and
        which frames directory this tile is measured and written against.

        A candidate that was never kept has no tile on disk to re-cut and comes back in `skipped`.
        """
        from snapshot.vision.recentre import recentre

        with self.lock:
            moved, unchanged, skipped = [], [], []
            touched: dict[str, dict] = {}
            for name in names:
                tag = _tile_tag(name)
                index = _tile_index(name)
                candidates, decisions, frames_dir = self._dataset_for(tag)
                # _dataset_for FALLS BACK TO THE BOUND DATASET when a tag's candidates file cannot
                # be found, which for a thumbnail is a wrong picture and here would be a wrong
                # WRITE - re-cutting an unrelated row and saving a rect onto it. the fallback is
                # detectable by identity, and it is reachable: a sweep tile's tag reads
                # `..._cnn-20260903-234900` while its file is `....cnn-20260903-234900.candidates`
                fell_back = (
                    tag and tag != (self.session_tag or "") and candidates is self.candidates
                )
                if fell_back or index >= len(candidates) or frames_dir is None:
                    skipped.append(name)
                    continue
                candidate = candidates[index]
                key = str(index)
                # THE TILE ON DISK IS THE PRECONDITION, not a decision record. This asked for
                # record["saved_as"] and skipped every SWEEP tile as a result - a sweep cuts its
                # tiles straight out and writes NO decisions sidecar at all, so `saved_as` is
                # missing for exactly the tiles this feature exists to fix. The tile is named
                # `<tag>_k<index>` by construction, which is how it was handed to us.
                tile = self.tile_path(name)
                if tile is None:
                    skipped.append(name)
                    continue
                record = decisions.get(key) or {"keep": True, "discard": False, "saved_as": name}
                record.setdefault("saved_as", name)

                origin_top = candidate["top"] - paths.MARGIN_Y
                origin_left = candidate["left"] - paths.MARGIN_X
                current = _corrected_box(candidate, record)
                box = (
                    current["left"],
                    current["top"],
                    current["width"],
                    current["height"],
                )

                frame = self._frame(candidate["path"], frames_dir)
                found = recentre(frame, box)
                if found is None or found.moved == 0:
                    unchanged.append(name)
                    continue

                new_rect = {
                    "left": found.left - origin_left,
                    "top": found.top - origin_top,
                    "width": found.width,
                    "height": found.height,
                }
                # CUT FROM THE FRAME, THE SAME WAY THE TILE WAS CUT, and keep its exact pixel
                # size. _rect_template cuts from the PADDED CROP instead, which cannot work here:
                # the pad is candidate width + 2*MARGIN_X, and the pool's uniform width is wider
                # than that for any candidate narrower than about 313px - measured, candidate 797
                # is 169 wide with a 229px pad against a 343px uniform width. A sweep never used
                # that path either; it cuts straight out of the frame.
                #
                # THE TILE'S OWN SIZE IS THE SIZE. Reading it back off the npz keeps every crop in
                # the pool exactly as wide and tall as it already was, which is the property the
                # net cares about - re-deriving it from a global would silently resize tiles that
                # were cut under a different setting.
                existing = load_template(tile)
                tile_h, tile_w = existing.rgb.shape[:2]
                centre_x = found.left + found.width / 2
                centre_y = found.top + found.height / 2
                cut_left = int(round(centre_x - tile_w / 2))
                cut_top = int(round(centre_y - tile_h / 2))
                # a negative start is caught by crop_at_rect's shape check, but only because
                # numpy's wrap-around happens to produce a short slice. say it outright rather
                # than rest on that - a plate near the top edge is the ordinary case here
                if cut_left < 0 or cut_top < 0:
                    unchanged.append(name)
                    continue
                try:
                    template = crop_at_rect(
                        frame, name, top=cut_top, left=cut_left, height=tile_h, width=tile_w
                    )
                except TemplateError:
                    # a plate close enough to an edge that its tile will not fit is left alone
                    # rather than cut smaller - a short tile is a worse training row than an
                    # off-centre one, and the human can still drag it
                    unchanged.append(name)
                    continue
                # BACK INTO THE POOL IT CAME FROM. writing to the primary would leave the
                # original where it was and mint a second copy under the same name
                save_template(template, tile.parent)
                record["rect"] = new_rect
                decisions[key] = record
                touched[tag] = decisions
                moved.append({"name": name, "dx": found.shift_x, "dy": found.shift_y})

            for tag, decisions in touched.items():
                if tag == (self.session_tag or "") or not tag:
                    self._save_decisions()
                else:
                    out = paths.LABELS_DIR / f"{tag}.candidates.decisions.json"
                    out.write_text(json.dumps(decisions, indent=0))
            return {"moved": moved, "unchanged": unchanged, "skipped": skipped}

    def _rect_template(
        self, candidate: dict, rect: dict, name: str, height: int, frames_dir: Path | None = None
    ) -> PlateTemplate:
        """a fixed-size crop_at_rect at the rect's position within the PADDED crop - no re-detection.

        rect coordinates are crop-local, relative to _padded_frame's own (0,0) - the same crop
        crop_bytes serves to the browser, so what's displayed and what gets extracted are
        guaranteed to agree on where every pixel is. WIDTH is always self.uniform_width, the one
        global size every candidate shares - not whatever rect['width'] the client happens to
        send, which used to come from each row's own guessed template and varied a little row to
        row ("they are all different," Balthazar Fitzpatrick's own words). HEIGHT is the caller's effective
        height (global default + this candidate's own correction, see _effective_height) - only
        rect's top/left (where the drag currently has it centred) come from the client.
        """
        canvas = self._padded_frame(candidate, frames_dir=frames_dir)
        return crop_at_rect(
            canvas,
            name,
            # rect.top/left are frequently non-integer (e.g. a height resize recentring around
            # a midpoint) - crop_at_rect slices the array directly, which raises TypeError on a
            # float index. round instead of truncating so a .5 doesn't bias the crop leftward/up
            top=int(round(rect["top"])),
            left=int(round(rect["left"])),
            height=height,
            width=self.uniform_width,
        )

    # WHICH SOURCES ARE CLOSED, per pool. Closing is a VIEW decision - the tiles stay exactly where
    # they are and keep their judgements - so it is remembered beside the tiles it hides rather than
    # in a global setting, and a repointed pool brings its own.
    @property
    def unsorted_dir(self) -> Path:
        """the PRIMARY pool - where a newly cut tile is written, and where pool-wide state lives.

        Reading spans every pool; writing a new tile only ever goes here, because "which of several
        directories should this belong to" has no answer the tool could work out.
        """
        return self.pools[0]

    def tile_paths(self) -> list[Path]:
        """every tile in every pool, primary first. Sidecars and the mean cache are not tiles."""
        found: dict[str, Path] = {}
        for pool in self.pools:
            if not pool.is_dir():
                continue
            for path in sorted(pool.glob("*.npz")):
                # a name in two pools resolves to the first, so the primary always wins.
                # SIDECARS ARE NOT TILES: every one is underscore-prefixed (_labels.json,
                # _closed.json, _exclusions.jsonl, and the _means.npz the colour sort used to
                # leave behind) while a tile is always "<tag>_k<n>"
                if not path.name.startswith("_"):
                    found.setdefault(path.stem, path)
        return list(found.values())

    def tile_path(self, name: str) -> Path | None:
        """where this tile actually is, across the pools - or None"""
        for pool in self.pools:
            candidate = pool / f"{name}.npz"
            if candidate.is_file():
                return candidate
        return None

    def pool_of(self, name: str) -> Path:
        """the pool a tile belongs to: where its file is, else where its record was read from,
        else the primary. The middle case is what keeps a record with its own pool when the tiles
        are momentarily unreachable - otherwise a detached pool's judgements migrate."""
        found = self.tile_path(name)
        if found is not None:
            return found.parent
        return self._record_origin.get(name, self.pools[0])

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
            self.closed_file.parent.mkdir(parents=True, exist_ok=True)
            self.closed_file.write_text(json.dumps(sorted(tags), indent=0))

    def unsorted_names(self) -> list[str]:
        """every visible tile's NAME, without reading a single pixel.

        clusters() only ever needed names and one colour per tile, and `unsorted_templates` handed
        it 0.99 GB of arrays to get them, for a colour sort that no longer exists.
        """
        closed = self.closed_tags()
        names = sorted(path.stem for path in self.tile_paths())
        if not closed:
            return names
        return [n for n in names if (_tile_tag(n) or "untagged") not in closed]

    def unsorted_templates(self) -> list[PlateTemplate]:
        try:
            templates = [load_template(path) for path in self.tile_paths()]
        except (TemplateError, OSError, ValueError):
            return []
        if not templates:
            return []
        closed = self.closed_tags()
        if not closed:
            return templates
        return [t for t in templates if (_tile_tag(t.name) or "untagged") not in closed]

    def _dataset_for(self, tag: str):
        """(candidates, decisions, frames_dir) for the dataset a tile BELONGS to, which is not
        necessarily the bound one - the unsorted pool is library-wide and spans datasets. resolving
        a tile's index against whatever happens to be bound renders it from an unrelated frame.

        cached per tag: the cluster grid asks for ~100 thumbs at a time, and re-reading a
        candidates file per tile would be visibly slow.
        """
        if not tag or tag == self.session_tag:
            return self.candidates, self.decisions, self.frames_dir
        if tag in self._dataset_cache:
            return self._dataset_cache[tag]

        candidates_path = _candidates_file_for(tag)
        frames_dir = _session_frames_for(tag)
        if candidates_path is None or frames_dir is None:
            entry = (self.candidates, self.decisions, self.frames_dir)
        else:
            candidates = [
                json.loads(line)
                for line in candidates_path.read_text().splitlines()
                if line.strip()
            ]
            decisions_path = candidates_path.with_suffix(".decisions.json")
            decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
            entry = (candidates, decisions, frames_dir)
        self._dataset_cache[tag] = entry
        return entry

    def unsorted_thumb_bytes(self, name: str) -> bytes | None:
        """crops from the ORIGINAL frame, sized so the bar itself always occupies exactly the
        middle 50% of the crop's height - real screenshot background above and below, not a
        blank or masked-out fill, at ANY output size the frontend later scales this to (the
        frontend fills a fixed-height viewport with this image, so whatever that height is, the
        25%/50%/25% split holds automatically). Width is left-anchored and generously
        over-cropped - the frontend's own overflow:hidden crops it down to its real display
        width, never squeezing the bar to fit.

        falls back to the isolated saved tile (transparent at the masked-out pixels) only when
        no candidate/rect is available to anchor a frame crop from - a defensive path that should
        not normally be reachable for a real kept tile.
        """

        path = self.tile_path(name)
        if path is None:
            return None

        index = _tile_index(name)
        candidates, decisions, frames_dir = self._dataset_for(_tile_tag(name))
        if index < len(candidates):
            candidate = candidates[index]
            shown = _corrected_box(candidate, decisions.get(str(index)))
            bar_top, bar_left = shown["top"], shown["left"]
            bar_h, bar_w = shown["height"], shown["width"]

            frame = self._frame(candidate["path"], frames_dir)
            fh, fw = frame.shape[:2]
            # REAL BUG FOUND AND FIXED HERE: clamping top/bottom to the frame's edges
            # INDEPENDENTLY breaks the intended 25/50/25 symmetry the moment either side runs
            # out of room - some of this session's source frames are barely bigger than the bar
            # itself (one measured 125x20px total for a 12px-tall bar), so EVERY kept candidate
            # hit this. One side could lose nearly all its margin while the other kept all of
            # its, and the frontend's height:100% scaling then stretched whatever sliver
            # survived into a large, disproportionate vertical stripe. Margin is now the same
            # amount on BOTH sides - whatever's actually available, capped by the smaller side -
            # so the bar stays centred no matter how little room the frame has, down to a
            # graceful "just the tight bar" floor instead of a warped image.
            margin = min(bar_h // 2, bar_top, fh - (bar_top + bar_h))
            margin = max(0, margin)
            top, bottom = bar_top - margin, bar_top + bar_h + margin
            # serve what is meant to be SEEN. this used to over-crop 4x wide on the assumption the
            # frontend would crop it back, but the frontend scaled by height instead - leaving about
            # 3% of the image visible at ~4.7x zoom, which is the "very zoomed in" Balthazar Fitzpatrick reported
            left = max(0, bar_left - 20)
            right = min(fw, bar_left + bar_w + 20)
            vis = frame[top:bottom, left:right]
            buf = BytesIO()
            Image.fromarray(vis).save(buf, format="PNG")
            return buf.getvalue()

        template = load_template(path)
        rgba = np.dstack([template.rgb, np.full(template.mask.shape, 255, dtype=np.uint8)])
        rgba[~template.mask, 3] = 0
        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def library_thumb_bytes(self, name: str) -> bytes | None:
        """a promoted template renders ITSELF, unlike an unsorted tile which crops its frame.

        promotion averages its members into a composite and the individuals leave the pool, so
        there is no originating frame to crop - the composite is the only pixels that exist.
        """
        from io import BytesIO

        import numpy as np

        path = self.library / f"{name}.npz"
        if not path.is_file():
            return None
        template = load_template(path)
        rgba = np.dstack([template.rgb, np.full(template.mask.shape, 255, dtype=np.uint8)])
        rgba[~template.mask, 3] = 0
        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def library_clusters(self) -> dict:
        """the promoted templates, grouped by the label they were promoted UNDER.

        no k-means: a promoted template already carries its class in its filename, so clustering it
        again would only re-derive - worse - a decision already made by hand.
        """
        try:
            templates = load_templates(self.library)
        except TemplateError:
            return {"clusters": [], "k": 0, "source": "library"}

        groups: dict[str, list] = {}
        for template in templates:
            groups.setdefault(_library_label(template.name), []).append(template.name)

        clusters = [
            {
                "label": label,
                "items": [
                    {"name": n, "excluded": False, "source": "library"} for n in sorted(names)
                ],
                "size": len(names),
            }
            for label, names in sorted(groups.items())
        ]
        return {"clusters": clusters, "k": len(clusters), "source": "library"}

    def pool_sources(self) -> dict:
        """which recordings the pool's tiles came from, and how many each contributed.

        the pool is cumulative by design - every recording's tiles cluster together, which is the
        point. but nothing could ever take one back OUT, so a set saved twice, or a recording no
        longer wanted, stayed in the grid forever with no way to close it.
        """
        counts: dict[str, int] = {}
        for path in self.tile_paths():
            tag = _tile_tag(path.stem) or "untagged"
            counts[tag] = counts.get(tag, 0) + 1
        # A CLOSED SOURCE IS STILL HERE, just not shown - its tiles and their judgements are on
        # disk. Listing it with `closed` is what lets the page offer it back instantly instead of
        # sending it round the cutting path again as though it were new.
        closed = self.closed_tags()
        return {
            "sources": [
                {"tag": t, "tiles": n, "closed": t in closed} for t, n in sorted(counts.items())
            ]
        }

    def openable_datasets(self) -> dict:
        """candidates files on disk that are NOT currently in the pool.

        THE POOL IS THE OPEN SET. a dataset exists as a candidates file the moment find saves it or
        a sweep writes it; it becomes visible in select only once its tiles are cut in.
        So "open" means cut, "close" means archive back out, and this lists what is available to
        open - anything already open is offered by pool_sources instead.
        """
        labels_root = Path(self.bases["labels"])
        open_tags = {(_tile_tag(path.stem) or "untagged") for path in self.tile_paths()}
        found = []
        for path in sorted(labels_root.glob("*.candidates.jsonl")):
            name = path.name
            tag, source = _dataset_tag(name)
            if tag in open_tags:
                continue
            boxes = sum(1 for line in path.read_text().splitlines() if line.strip())
            # A SWEEP THAT FOUND NOTHING STILL WRITES ITS FILE. offering those made "open dataset"
            # look broken: the pick succeeded, cut zero tiles, and the grid did not change
            if not boxes:
                continue
            found.append({"tag": tag, "file": name, "boxes": boxes, "source": source})
        return {"datasets": found}

    def open_dataset(self, name: str) -> dict:
        """cut one candidates file's tiles into the pool, which is what makes it visible"""
        path = Path(self.bases["labels"]) / name
        if not path.is_file():
            return {"error": f"no dataset named {name!r}"}
        boxes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        boxes = [b for b in boxes if not b.get("negative")]
        if not boxes:
            return {"error": f"{name} holds no boxes"}
        stem = path.name
        tag, _source = _dataset_tag(stem)
        frames_dir = _session_frames_for(stem.split(".")[0])
        if frames_dir is None or not frames_dir.is_dir():
            return {"error": f"no frames directory for {tag}"}
        width = int(statistics.median(b["width"] for b in boxes))
        height = int(statistics.median(b["height"] for b in boxes))
        cut = self._cut_drawn(boxes, width, height, frames_dir=frames_dir, tag=tag)
        return {"tag": tag, "tiles": cut, "boxes": len(boxes)}

    def _pool_box(self, name: str) -> dict | None:
        """everything a pool tile needs to be re-aligned: its box, its file, and its frames.

        a tile is named "<tag>_k<index>" and the index counts the NON-NEGATIVE boxes, which is
        the order _cut_drawn walks - not the file's line numbers. both are carried, because the
        index is what addresses the tile and the line is what has to be written back.
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
                if _dataset_tag(p.name)[0] == tag
            ),
            None,
        )
        if path is None:
            return None
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        boxes = [json.loads(line) for line in lines]
        live = [(i, b) for i, b in enumerate(boxes) if not b.get("negative")]
        if index >= len(live):
            return None
        line_index, box = live[index]
        frames_dir = _session_frames_for(path.name.split(".")[0])
        if frames_dir is None or not frames_dir.is_dir():
            return None
        kept = [b for _i, b in live]
        return {
            "path": path,
            "boxes": boxes,
            "line_index": line_index,
            "box": box,
            "frames_dir": frames_dir,
            # the cut size is the median over the whole set, exactly as _cut_drawn derives it -
            # one box moving must not change the size every other tile was cut at
            "width": int(statistics.median(b["width"] for b in kept)),
            "height": int(statistics.median(b["height"] for b in kept)),
        }

    def pool_box_info(self, name: str) -> dict:
        """where this pool tile's box currently sits inside the padded crop the browser draws.

        left/top are paths.MARGIN_X/paths.MARGIN_Y by construction - _padded_frame centres the box in its own
        margin - so a correctly drawn box needs no drag at all, which is what makes the guide a
        fixed target rather than a moving one
        """
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
        """the padded crop around a pool tile's box, from ITS OWN recording - not the bound one.

        the bound dataset and the dataset a pool tile came from are routinely different, which is
        why this cannot go through crop_bytes and its self.frames_dir
        """

        found = self._pool_box(name)
        if found is None:
            return None
        canvas = self._padded_frame(found["box"], frames_dir=found["frames_dir"])
        buf = BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        return buf.getvalue()

    def realign_pool_tile(self, name: str, left: float, top: float) -> dict:
        """move one drawn box by the drag, then re-cut just its tile over the old file.

        THE NAME IS UNCHANGED, which is the whole point: the pool's labels are keyed by name, so
        correcting a crop costs a re-cut and keeps whatever colour it was already labelled
        """
        found = self._pool_box(name)
        if found is None:
            return {"error": f"no source on disk for {name!r}"}
        box, path = found["box"], found["path"]
        # the drag is reported in crop-local pixels, and the box sat at (paths.MARGIN_X, paths.MARGIN_Y) - so
        # the shift is the difference, applied to the box's real position in the frame
        box["left"] = int(round(box["left"] + (left - paths.MARGIN_X)))
        box["top"] = int(round(box["top"] + (top - paths.MARGIN_Y)))
        found["boxes"][found["line_index"]] = box
        with self.lock:
            path.write_text("".join(json.dumps(b) + "\n" for b in found["boxes"]))
            cut = self._cut_drawn(
                [box],
                found["width"],
                found["height"],
                frames_dir=found["frames_dir"],
                tag=name.rpartition("_k")[0],
                start_index=int(name.rpartition("_k")[2]),
            )
        if not cut:
            return {"error": f"{name} could not be cut there - it runs off the frame"}
        return {"name": name, "left": box["left"], "top": box["top"]}

    def recut_pool(self) -> dict:
        """re-cut every drawn set's tiles from its frames, at the current padding.

        LABELS SURVIVE, which is the whole point. a tile's filename is derived from its recording
        and index, so re-cutting overwrites each file in place and _labels.json - which is keyed by
        NAME - still points at the same plate. changing the padding therefore costs a re-cut, never
        a re-draw of 234 rectangles.
        """
        labels_root = Path(self.bases["labels"])
        done, failed = [], []
        for path in sorted(labels_root.glob("*.drawn-*.candidates.jsonl")):
            tag = path.name.split(".drawn-")[0]
            boxes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            boxes = [b for b in boxes if not b.get("negative")]
            if not boxes:
                continue
            frames_dir = _session_frames_for(tag)
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

    def pool_source_state(self, tag: str) -> dict:
        """what closing this source would cost, so the page can ask before it does anything.

        `judged` is the answer to "were there changes": a tile carrying a class or marked "not a
        class" is work someone did, and discarding it throws that away while the tiles themselves
        stay put.
        """
        records = self.pool_records()
        tiles = judged = 0
        for path in self.tile_paths():
            if (_tile_tag(path.stem) or "untagged") != tag:
                continue
            tiles += 1
            record = records.get(path.stem) or {}
            if record.get("label") or record.get("excluded"):
                judged += 1
        return {"tag": tag, "tiles": tiles, "judged": judged}

    def close_pool_source(self, tag: str, discard: bool = False) -> dict:
        """stop showing a source. NOTHING MOVES.

        THIS USED TO ARCHIVE, and archiving is not closing. It moved every tile out to
        the archive and dropped their label records, so "close" silently cost the work: on
        7 Sep it emptied a 6,133 tile pool and left `_labels.json` holding `{}`, which is a tracked
        file. Reopening then meant cutting every tile again from the candidates.

        Closing is a VIEW decision now - the tag is remembered as hidden, the tiles stay on disk and
        keep their judgements, and opening it again is instant. `discard` is the other half of the
        question the page asks: it drops the judgements for this source and leaves the tiles, which
        is the only destructive thing here and is never the default.
        """
        if not tag:
            return {"error": "no source given"}
        state = self.pool_source_state(tag)
        with self.lock:
            if discard:
                records = self.pool_records()
                for path in self.tile_paths():
                    if (_tile_tag(path.stem) or "untagged") == tag:
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

    def training_sets(self) -> dict:
        """what the cnn may train on: the durable sets promote_to_training writes, never a raw
        candidates queue. a queue is unreviewed and unlabelled by definition - offering one here
        invites training on boxes nobody has judged, which is the opposite of what promote is for.
        """
        # NAMED IN paths, not spelled out here. This was `<labels>/../datasets`, which quietly
        # became `training/datasets` the moment the boxes moved under training/ - the sets were
        # still on disk and the tab listed none of them
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

    @staticmethod
    def _synthetic_negatives(
        boxes: list[dict], frame_size: tuple[int, int], seed: int = 0
    ) -> list[dict]:
        """one background box per drawn box, placed where no drawn box is.

        RESTS ON ONE ASSUMPTION, and it is Balthazar Fitzpatrick's own: he marked every plate that was visible in
        the frame. Given that, anywhere else in that frame IS background - which makes a balanced
        negative set free, where hard negatives otherwise only arrive once a model has fired and
        been corrected.

        the RECTANGLES must not overlap, though the training crop taken around them may - a crop is
        a wide window and demanding those be disjoint too would leave nowhere to stand on a busy
        frame. seeded, so promoting twice gives the same set rather than quietly growing it.
        """
        width, height = frame_size
        rng = random.Random(seed)
        placed = [(b["left"], b["top"], b["width"], b["height"]) for b in boxes]
        out = []
        for box in boxes:
            bw, bh = box["width"], box["height"]
            for _ in range(40):  # a bounded search; a frame this full simply yields fewer
                left = rng.randrange(0, max(1, width - bw))
                top = rng.randrange(0, max(1, height - bh))
                if any(
                    left < px + pw and left + bw > px and top < py + ph and top + bh > py
                    for px, py, pw, ph in placed
                ):
                    continue
                placed.append((left, top, bw, bh))
                out.append(
                    {"path": box["path"], "left": left, "top": top, "width": bw, "height": bh}
                )
                break
        return out

    @staticmethod
    def _candidates_for_tag(labels_root: Path, tag: str) -> Path | None:
        """the candidates file a tile with this tag actually came from.

        THE BUG THIS FIXES, found 2026-09-02: this used to take the first glob match, and a sweep's
        file sorts before a drawn one - ".cnn-" < ".drawn-". So hand-drawn tiles tagged
        "<recording>" resolved against an EMPTY cnn sweep of the same recording, every index fell
        out of range, and promote wrote zero rows while reporting success.

        A sweep tile carries its stamp in its own tag ("<recording>_cnn-<stamp>"), so it names its
        file exactly. A bare tag is drawn work and must never match a sweep. Where several drawn
        passes exist, the one holding the index is the right one - re-saving writes a fresh stamped
        sibling, so the newest is not automatically the one a given tile was cut from.
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
        """the boxes for one tag, and the file they came from.

        THE PATH IS RETURNED because the decisions sidecar sits beside the candidates file and
        must be the sidecar of the file actually chosen - several drawn passes can exist for one
        tag, and reading corrections from the wrong pass applies one box's correction to another.
        """
        exact = self._candidates_for_tag(labels_root, tag)
        options = [exact] if exact else sorted(labels_root.glob(f"{tag}.drawn-*.candidates.jsonl"))
        best: list[dict] = []
        best_path: Path | None = None
        for path in options:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if indexes and max(indexes) < len(rows):
                return rows, path
            if len(rows) > len(best):
                best, best_path = rows, path
        return best, best_path

    @staticmethod
    def _corrections_beside(candidates_path: Path | None) -> dict:
        """the decisions sidecar's per-index records, or nothing.

        Cheap and often absent: a sweep writes no sidecar at all, and a drawn pass only grows one
        once something has been dragged or re-centred.
        """
        if candidates_path is None:
            return {}
        sidecar = candidates_path.with_suffix(".decisions.json")
        try:
            value = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    # what makes two rows the same box: the recording, the frame and the rect. NOT the source tag -
    # the same plate re-proposed by a later sweep is still the same plate, and merging must not
    # keep both copies of it
    ROW_KEY = ("recording", "frame", "left", "top", "width", "height")

    @staticmethod
    def _row_key(row: dict) -> tuple:
        return tuple(row.get(field) for field in ReviewState.ROW_KEY)

    def promote_to_training(self, name: str = "plates", mode: str = "new") -> dict:
        """every kept, labelled tile written into ONE durable training set.

        WHY A SEPARATE FILE rather than training straight off the candidates. a candidates file is
        a review QUEUE - it belongs to one recording and one mining run, and there is a new one
        every time anything is drawn or swept. the training set is the accumulating answer: draw
        boxes on a fresh recording, confirm some CNN proposals from another, and both land here
        under the same schema. train() then reads one file instead of being handed a list of
        queues nobody remembered to combine.

        a tile is named "<dataset tag>_k<candidate index>", which is what lets a row be resolved
        back to the box it came from - the frame, the rect and the recording it belongs to.

        THE SET IS NAMED, and a name that already exists is a decision rather than an overwrite.
        mode "new" refuses and reports {"exists": true} so the page can ask; "overwrite" replaces
        the file whole; "merge" keeps every row the current pool does not speak for - see
        _merge_rows. A sidecar <name>.meta.json records which source contributed what, because a
        merged set is several days of judging and "where did this row come from" has to survive.
        """
        records = self.pool_records()

        # TWO DEFINITIONS IN ONE SET IS NOT A SET. the channel map is built from the labels present,
        # so "hostile / target" judged under one carve-up and "hostile / rogue / target" judged
        # under another would become two unrelated channels describing the same thing - and nothing
        # downstream could tell. Refused by name, the same way a set mixing capture resolutions is.
        seen_defs = {
            record["classdef"]
            for record in records.values()
            if record.get("classdef") and record.get("label")
        }
        if len(seen_defs) > 1:
            return {
                "error": (
                    "this pool holds tiles judged under different class definitions - "
                    + ", ".join(sorted(seen_defs))
                    + ". Close the datasets belonging to one of them and promote separately."
                )
            }

        by_tag: dict[str, list[tuple[str, int, str, bool]]] = {}
        for tile, record in records.items():
            excluded = bool(record.get("excluded"))
            # AN EXCLUDED SWEPT CANDIDATE IS A HARD NEGATIVE, not merely absent. the model fired
            # there and was told no, which is the one kind of background worth sampling
            # deliberately - a random window almost never contains the model's own mistake. an
            # excluded DRAWN box is different: nothing proposed it, so dropping it is enough
            if not excluded and not record.get("label"):
                continue
            tag = _tile_tag(tile)
            if tag:
                by_tag.setdefault(tag, []).append(
                    (tile, _tile_index(tile), record.get("label"), excluded)
                )

        rows, missing = [], 0
        labels_root = Path(self.bases["labels"])
        for tag, entries in by_tag.items():
            found = self._candidates_for_tag(labels_root, tag)
            if not found:
                missing += len(entries)
                continue
            candidates, candidates_path = self._resolve_candidates(
                labels_root, tag, [index for _, index, _, _ in entries]
            )
            corrections = self._corrections_beside(candidates_path)
            frames_dir = _session_frames_for(tag)
            for _, index, label, excluded in entries:
                if index >= len(candidates) or frames_dir is None:
                    missing += 1
                    continue
                box = _corrected_box(candidates[index], corrections.get(str(index)))
                rows.append(
                    {
                        "recording": str(frames_dir.parent.relative_to(paths.SESSIONS_DIR)),
                        "frame": box["path"],
                        "left": box["left"],
                        "top": box["top"],
                        "width": box["width"],
                        "height": box["height"],
                        "label": None if excluded else label,
                        "negative": excluded,
                        "source": tag,
                    }
                )

        # SYNTHETIC NEGATIVES, one per drawn box, from the "I marked everything visible"
        # assumption. generated here rather than stored so they follow the current keep set

        synthetic = 0
        positives_by_frame: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            if not row.get("negative"):
                positives_by_frame.setdefault((row["recording"], row["frame"]), []).append(row)
        for (recording, frame), frame_rows in positives_by_frame.items():
            image_path = paths.SESSIONS_DIR / recording / "frames" / frame
            try:
                with Image.open(image_path) as handle:
                    frame_size = handle.size
            except (OSError, ValueError):
                continue
            for negative in self._synthetic_negatives(
                [dict(r, path=frame) for r in frame_rows],
                frame_size,
                # crc32, NOT hash(): Python randomises str hashing per process, so this drew a
                # different set of negatives on every promote of the same frame - and a merge
                # keys rows on their rect, so no negative ever matched its own previous run
                seed=zlib.crc32(frame.encode()) & 0xFFFF,
            ):
                rows.append(
                    {
                        "recording": recording,
                        "frame": frame,
                        "left": negative["left"],
                        "top": negative["top"],
                        "width": negative["width"],
                        "height": negative["height"],
                        "label": None,
                        "negative": True,
                        "source": "synthetic",
                    }
                )
                synthetic += 1

        out = paths.DATASETS_DIR / f"{_set_filename(name)}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read_training_set(out)

        # A PROMOTE THAT WROTE NOTHING MUST NOT DESTROY THE LAST ONE. This rewrites the file whole,
        # so a run that produced no rows wrote an EMPTY training set over a good one, twice, for
        # two different reasons that need two different fixes:
        #
        #   - the recordings stopped being findable (a restart reset the sessions base). the labels
        #     were all still there; promote simply could not see the frames.
        #   - nothing in the pool was labelled YET. Balthazar Fitzpatrick promoted at 19:56:37 and the pool's
        #     labels were not written until 19:58:29, so every tile hit the skip above and the
        #     old guard - which only fired on `missing` - let a rows=0, missing=0 run straight
        #     through and report success.
        #
        # so the guard is on `not rows`, and it names which of the two it was.
        if not rows:
            # COUNTED OFF DISK, not off `records`. a decision record only exists once a tile has
            # been judged, so an unjudged pool of 4000 tiles reads as {} here - and "the pool is
            # empty" is the one answer that would send Balthazar Fitzpatrick looking in the wrong place
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
                    "tag them in select, then promote"
                )
            return {
                "error": f"nothing to promote: {why}. "
                f"the existing {len(existing)}-row training set was left alone",
                "out": str(out),
                "rows": 0,
                "unresolved": missing,
            }

        if existing and mode == "new":
            # THE NAME COLLIDES, and only Balthazar Fitzpatrick can say what that means - the page asks
            return {
                "exists": True,
                "name": out.stem,
                "out": str(out),
                "existing_rows": len(existing),
                "would_add": len(rows),
            }

        merged = self._merge_rows(existing, rows) if mode == "merge" and existing else rows
        # rewritten whole, not appended: promoting twice must not double every row, and the pool
        # is the source of truth for what is currently kept
        out.write_text("".join(json.dumps(r) + "\n" for r in merged))
        meta = self._write_set_meta(
            out,
            merged,
            labels_root,
            mode if existing else "new",
            {row.get("source") for row in rows},
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
        }

    @staticmethod
    def _read_training_set(path: Path) -> list[dict]:
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

    def _merge_rows(self, existing: list[dict], fresh: list[dict]) -> list[dict]:
        """add this pool's rows to a set without disturbing what other pools put there.

        THE POOL SPEAKS ONLY FOR ITS OWN SOURCES. a merge that kept every existing row would leave
        the rows of a source that has since been re-judged sitting beside their replacements, and
        one that dropped them all would silently discard days of work done from another pool. So a
        row survives unless this promote covers its source or re-states its exact box.

        synthetic negatives are never carried over: they are generated per promote from the current
        positives, so an old one describes a frame's positives as they used to be.
        """
        replaced = {row["source"] for row in fresh} | {"synthetic"}
        keys = {self._row_key(row) for row in fresh}
        kept = [
            row
            for row in existing
            if row.get("source") not in replaced and self._row_key(row) not in keys
        ]
        return kept + fresh

    def _write_set_meta(
        self,
        out: Path,
        rows: list[dict],
        labels_root: Path,
        mode: str,
        written_sources: set[str],
    ) -> dict:
        """the sidecar that says where each row in the set came from.

        Balthazar Fitzpatrick: "keep the integrity of where a label comes from so that we can still manage that".
        the rows carry their own source tag, but nothing said which candidates FILE a tag was read
        out of, and that file is the only thing that can re-resolve a row to its box. a merged set
        outlives the queue it was built from, so the provenance has to be written down rather than
        rediscovered.
        """
        previous = {}
        meta_path = out.with_suffix(".meta.json")
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
            # a source this promote did not write keeps the file and timestamp it came in with, so
            # a merge does not restamp work done days ago as if it had happened now
            entry["candidates"] = str(found) if found else was.get("candidates")
            entry["promoted"] = stamp if source in written_sources else was.get("promoted", stamp)

        meta = {
            "name": out.stem,
            "written": stamp,
            "mode": mode,
            "rows": len(rows),
            "sources": sorted(by_source.values(), key=lambda e: e["source"]),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        return meta

    def set_excluded(self, name: str, excluded: bool) -> bool | None:
        """DECLARATIVE, not a toggle: state what this tile should be and it becomes that.

        Reversible, not a delete - the tile stays on disk and stays visible in its cluster
        (dimmed, with an X), so a mismatch found by eye can be flagged and un-flagged just as
        easily. Returns the resulting state, or None if this tile doesn't exist.

        IT USED TO FLIP, and that was wrong on a multi-selection, which is the only way the button
        is ever used. Balthazar Fitzpatrick: "its declarative, if i select any and assign it a class or not a
        class, it is supposed to be stateless, and apply my will to all." Flipping meant a
        selection where some were already judged came out INVERTED - the judged ones re-included,
        the rest excluded - which is the opposite of what the press said. Nothing needs the old
        behaviour: un-judging is assigning a class, the exact opposite press.

        The audit line is written only on a REAL change. A no-op press is a press, not history,
        and logging it would forge a record of a decision nobody made.
        """
        with self.lock:
            if self.tile_path(name) is None:
                return None
            # POOL-WIDE like the label, and for the same reason: a decision record is keyed by
            # candidate index, and index 5 exists in every dataset
            before = bool(self.pool_records().get(name, {}).get("excluded", False))
            if before == excluded:
                return excluded
            self.set_pool_excluded(name, excluded)

            log_path = self.unsorted_dir / "_exclusions.jsonl"
            entry = {
                "name": name,
                "index": _tile_index(name),
                "at": datetime.now(UTC).isoformat(),
                "action": "excluded" if excluded else "re-included",
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            return excluded

    def _pool_labels_path(self, pool: Path | None = None) -> Path:
        """one per pool - see _read_pool_file for why they are not merged into a single file"""
        return (pool or self.pools[0]) / "_labels.json"

    def pool_records(self) -> dict:
        """tile name -> {"label": str|None, "excluded": bool}.

        POOL-WIDE, because clusters() reads every tile in the pool regardless of what is bound
        while a decision record is keyed by CANDIDATE INDEX - and index 5 exists in every dataset.
        both the label and the exclusion had that bug: judging a tile from one recording while
        another was bound wrote onto the other recording's index 5.
        """
        raw = self._read_pool_file()
        out = {}
        for name, value in raw.items():
            # older files stored a bare label string; keep reading them
            out[name] = {"label": value, "excluded": False} if isinstance(value, str) else value
        return out

    def _read_pool_file(self) -> dict:
        """every pool's judgements, merged, primary winning a clash.

        ONE FILE PER POOL, NOT ONE FOR ALL OF THEM. The primary's `_labels.json` is tracked in git
        because a human's class assignments cannot be regenerated from anything; a generated pool's
        labels reproduce from `wt-synth-frames --seed` and must not land in that file. Each
        directory keeping its own is what makes both true at once.
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
        """back to the pool each record came from, so a generated set's labels stay in its own
        directory and the tracked file keeps only what a human decided"""
        split: dict[Path, dict] = {pool: {} for pool in self.pools}
        for name, record in records.items():
            split.setdefault(self.pool_of(name), {})[name] = record
        for pool, subset in split.items():
            # a pool that has never held a record and still holds none is left without a file
            path = pool / "_labels.json"
            if not subset and not path.is_file():
                continue
            with contextlib.suppress(OSError):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(subset, indent=1, sort_keys=True))

    def set_pool_excluded(self, name: str, excluded: bool) -> None:
        records = self.pool_records()
        record = records.setdefault(name, {"label": None, "excluded": False})
        record["excluded"] = excluded
        # a record claiming nothing is not a record - tidied the same way set_pool_label tidies,
        # so un-judging an unlabelled tile leaves no {"label": null, "excluded": false} behind
        if not record.get("label") and not excluded:
            records.pop(name, None)
        self._write_pool_records(records)

    def pool_labels(self) -> dict:
        """tile name -> the class Balthazar Fitzpatrick put it in.

        POOL-WIDE, NOT PER-DATASET, and that is a correctness fix rather than a convenience. these
        used to live on the bound dataset's decision record keyed by CANDIDATE INDEX - but
        clusters() reads every tile in the pool regardless of what is bound, and index 5 exists
        in every dataset. labelling a tile from one recording while another was bound wrote the
        label onto the other recording's index 5, overwriting one label and losing the other.
        clustering three drawn sets together would have corrupted labels silently.
        """
        return {n: r["label"] for n, r in self.pool_records().items() if r.get("label")}

    def set_pool_label(self, name: str, label: str | None) -> None:
        records = self.pool_records()
        record = records.setdefault(name, {"label": None, "excluded": False})
        record["label"] = label
        if not record["label"] and not record.get("excluded"):
            records.pop(name, None)
        self._write_pool_records(records)

    def set_pool_field(self, name: str, key: str, value) -> None:
        """one extra fact on a tile's record - today only which definition judged it.

        A RECORD WITH NO LABEL AND NO EXCLUSION IS NOT A RECORD, so this refuses to create one on
        its own: a definition name attached to a tile nobody judged would make the tile look judged
        to anything counting records.
        """
        records = self.pool_records()
        if name not in records:
            return
        records[name][key] = value
        self._write_pool_records(records)

    def pool_definitions(self) -> dict:
        """tile name -> the class definition it was judged under, for tiles that record one.

        A TILE WITHOUT ONE IS NOT BROKEN, it is older than definitions - judged when the fifteen
        classes were hardcoded. Its label still means what it meant; it simply cannot be mixed
        into a set with a newer one.
        """
        return {n: r["classdef"] for n, r in self.pool_records().items() if r.get("classdef")}

    def apply_manual_label(self, name: str, label: str, definition: str) -> str | None:
        """write the class the human picked, exactly as picked.

        WHAT THIS REPLACES, and it is the whole reason class definitions can exist at all: the
        previous version took a SHORTLIST of plausible identities and chose between them by nearest
        measured RGB. That only works while every member is one of fourteen hardcoded names with a
        reference colour sampled from real nameplates. A dimension a person invents - "player
        class", "fish species" - has no colour to be near, so the shortlist was silently filtered
        against the hardcoded table and an all-unknown pick returned nothing.

        The tile also records WHICH DEFINITION it was judged under. Nothing rewrites an older
        label when a definition changes; instead promotion refuses a set that mixes two, naming
        both. A judgement made in March means what it meant in March.
        """
        if self.tile_path(name) is None or not label:
            return None

        with self.lock:
            self.set_pool_label(name, label)
            self.set_pool_field(name, "classdef", definition)
            # ASSIGNING A CLASS IS THE OPPOSITE PRESS TO "not a class", so it un-judges. calling a
            # crop a plate IS calling it a plate, and leaving `excluded` set would keep it a hard
            # negative that also carries a class - two contradictory claims about one tile. this is
            # what gives the declarative model its undo without a toggle or a second button
            self.set_pool_excluded(name, False)
        return label

    def _load_promotion_counts(self) -> dict[str, int]:
        path = self.library / "_promotion_counts.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _save_promotion_counts(self, counts: dict[str, int]) -> None:
        (self.library / "_promotion_counts.json").write_text(json.dumps(counts, indent=0))

    def promote_to_library(self, name: str, slug_override: str | None = None) -> str | None:
        # NO LONGER REACHABLE FROM THE PAGE. the labelling loop ends at a training set now - classes
        # are applied and promoted to training, and nothing promotes a tile into the template
        # library. Kept because the library itself is still live: wt-fit-camera, wt-pitch-scale and
        # plate_track.locate_by_template all match against training/library

        """merges one curated tile into the REAL template library find_matches searches with -
        not a straight copy: if a template for this identity already exists there, the new
        tile is AVERAGED into it (weighted by how many have been merged so far, tracked in
        _promotion_counts.json since PlateTemplate itself carries no count) rather than
        replacing it, so the template converges toward a clean composite instead of just being
        whatever the last-promoted instance happened to look like. Mask is intersected (AND),
        not unioned - a pixel only stays part of the correlation region if EVERY merged instance
        agreed it was real, which is the conservative direction for a region that's used to
        judge a match, not just for display.

        uses the manual_label if this tile has one (highest confidence - Balthazar Fitzpatrick's own
        classification), falling back to the automatic cluster guess otherwise.
        """
        path = self.tile_path(name)
        if path is None:
            return None
        # BY NAME, from the pool records. this read the bound dataset's decision record keyed by
        # CANDIDATE INDEX, which is where labels USED to live - so after they moved pool-wide it
        # found nothing and fell through to the colour guess, quietly promoting a hand-labelled
        # tile under whatever hue happened to be nearest. the decisions fallback is only there
        # for records written before the move
        label = self.pool_labels().get(name)
        if not label:
            label = self.decisions.get(str(_tile_index(name)), {}).get("manual_label")
        # NO LABEL IS AN ANSWER, not a prompt to invent one. This fell back to guessing an identity
        # from the tile's mean colour, which is how a tile could be promoted into the library under
        # whatever hue happened to be nearest. The guess is gone; a tile nobody has classed is not
        # ready to be promoted.
        if not label:
            return None
        new_template = load_template(path)
        # an explicit override from the popup's editable slug field wins outright - Balthazar Fitzpatrick's own
        # choice of destination filename, not the auto-resolved one
        slug = _slugify(slug_override) if slug_override else _slugify(label)

        with self.lock:
            counts = self._load_promotion_counts()
            target_path = self.library / f"{slug}.npz"
            new_rgb, new_mask = new_template.rgb, new_template.mask
            if target_path.is_file():
                from PIL import Image

                existing = load_template(target_path)
                if existing.rgb.shape != new_rgb.shape:
                    h, w = existing.rgb.shape[:2]
                    new_rgb = np.asarray(Image.fromarray(new_rgb).resize((w, h), Image.NEAREST))
                    new_mask = (
                        np.asarray(
                            Image.fromarray(new_mask.astype(np.uint8) * 255).resize(
                                (w, h), Image.NEAREST
                            )
                        )
                        > 127
                    )
                n = counts.get(slug, 1)
                merged_rgb = ((existing.rgb.astype(np.float64) * n + new_rgb) / (n + 1)).round()
                merged_rgb = merged_rgb.astype(np.uint8)
                merged_mask = existing.mask & new_mask
                counts[slug] = n + 1
            else:
                merged_rgb, merged_mask = new_rgb, new_mask
                counts[slug] = 1

            save_template(PlateTemplate(name=slug, rgb=merged_rgb, mask=merged_mask), self.library)
            self._save_promotion_counts(counts)
        return slug

    # WHAT THE GRID IS FOR, and what it is not. This used to k-means the unjudged tiles by mean
    # colour and present each group as a guess at its identity. Balthazar Fitzpatrick: "it should
    # just sort the tiles statically after classes and last comes not a class."
    #
    # A COLOUR SORT ANSWERS A QUESTION NOBODY ASKED. The grid exists to show what has been judged
    # and what has not, and a guessed grouping obscures exactly that: two tiles sat together
    # because they were a similar green, which says nothing about whether either had been decided.
    # It also reshuffled on every relabel, so a tile moved between loads without being touched.
    UNJUDGED = "unjudged"
    NOT_A_CLASS = "not a class"

    def clusters(self) -> dict:
        """every tile grouped by the class it was given, in a fixed order.

        Assigned classes first, in the definition's own order (primitive, then state - see
        _label_order); then everything nobody has decided; then "not a class" last, because it is
        the one group you are finished with. Within a group, by name - so a tile only ever moves
        when its judgement does.
        """
        names = self.unsorted_names()
        if not names:
            return {"clusters": []}

        records = self.pool_records()

        def item_of(name: str) -> dict:
            return {
                "name": name,
                "excluded": records.get(name, {}).get("excluded", False),
                # THE TILE'S OWN CLASS, which is now also its group - the status dot and the tile's
                # position say the same thing, which is the point of sorting this way
                "assigned": records.get(name, {}).get("label"),
                # which dataset this tile was curated from - the pool spans several sources and
                # needs to be filterable by them
                "source": _tile_tag(name) or "untagged",
            }

        by_label: dict[str, list[str]] = {}
        unjudged: list[str] = []
        excluded: list[str] = []
        for name in sorted(names):
            record = records.get(name, {})
            if record.get("excluded"):
                excluded.append(name)
            elif record.get("label"):
                by_label.setdefault(record["label"], []).append(name)
            else:
                unjudged.append(name)

        clusters = [
            {"label": label, "items": [item_of(n) for n in group]}
            for label, group in sorted(by_label.items(), key=_label_order)
        ]
        # the two that are not classes go last, in this order: what still needs a decision before
        # what has already had one
        for label, group in ((self.UNJUDGED, unjudged), (self.NOT_A_CLASS, excluded)):
            if group:
                clusters.append({"label": label, "items": [item_of(n) for n in group]})
        return {"clusters": clusters}

    def tile_bytes(self, index: int, rect: dict | None = None) -> bytes | None:
        """the actual matching tile this candidate/alignment would produce, magenta = masked out.

        this is what find_matches would use, not a preview for its own sake - showing the masked
        corners and the left-half cut is the point, so a bad trim (background bleed, a cut that
        clips into the bar) is visible before Keep, not after. takes the current drag alignment
        rect when given, so it updates live as the image is dragged into place.
        """

        candidate = self.candidates[index]
        try:
            if rect:
                template = self._rect_template(
                    candidate, rect, "preview", self._effective_height(index)
                )
            else:
                frame = self._frame(candidate["path"])
                template = extract_template(self._sub_image(frame, candidate), "preview")
        except TemplateError:
            return None
        vis = template.rgb.copy()
        vis[~template.mask] = [255, 41, 214]
        buf = BytesIO()
        Image.fromarray(vis).save(buf, format="PNG")
        return buf.getvalue()
