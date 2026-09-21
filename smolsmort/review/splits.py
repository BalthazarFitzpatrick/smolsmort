"""train / val / test over a training set, assigned BY FRAME and never by row.

WHY BY FRAME. a crop window cut out of a full frame routinely contains several boxes and their
surroundings. Split row-wise and the same pixels land in train and in val: the object next to the
held-out one trained the model, and the holdout reports a number nobody can act on. A frame is the
smallest unit that does not leak.

WHY IT IS WRITTEN INTO THE SET rather than decided at training time. The set on disk is what a
checkpoint is judged against months later; a split re-drawn per run makes two runs incomparable
and quietly moves the holdout under a model that has already seen it.

The assignment is DETERMINISTIC from the frame's own name and the run's seed - no shuffle, no
walk of a shared random stream - so the same set always splits the same way, in any process, at
any worker count.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from smolsmort.review.find_state import write_json_atomic, write_jsonl_atomic

SPLITS = ("train", "val", "test")
DEFAULT_RATIO = (70, 20, 10)


def parse_ratio(text: str) -> tuple[int, int, int]:
    """ "70/20/10" -> (70, 20, 10)"""
    parts = text.replace(",", "/").split("/")
    if len(parts) != len(SPLITS):
        raise ValueError(f"--split wants three numbers like 70/20/10, got {text!r}")
    try:
        ratio = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(f"--split wants three numbers like 70/20/10, got {text!r}") from None
    if min(ratio) < 0 or sum(ratio) == 0:
        raise ValueError(f"--split wants non-negative numbers that are not all zero, got {text!r}")
    return ratio  # type: ignore[return-value]


def _shares(total: int, ratio: Sequence[int]) -> list[int]:
    """how many frames each split gets - largest remainder, so the counts sum to total exactly.

    a plain round() per split loses or invents a frame at most sizes, and the arithmetic then
    disagrees with the file it describes.
    """
    weight = sum(ratio)
    exact = [total * r / weight for r in ratio]
    counts = [int(e) for e in exact]
    order = sorted(range(len(ratio)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1

    # A SPLIT ASKED FOR AND LEFT EMPTY IS WORSE THAN A SMALL ONE - a holdout of zero frames reports
    # a perfect score. borrow from the largest split instead, while there is anything to borrow.
    for i, want in enumerate(ratio):
        if want and not counts[i]:
            donor = max(range(len(counts)), key=lambda j: counts[j])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[i] += 1
    return counts


def assign(frames: Iterable[str], ratio: Sequence[int] = DEFAULT_RATIO, seed: int = 0) -> dict:
    """frame name -> split name.

    the order is the frame's crc32 under the run's seed, not its index, so which frames land in
    the holdout does not track when they were generated.
    """
    names = sorted(set(frames))
    counts = _shares(len(names), ratio)
    ordered = sorted(names, key=lambda name: (zlib.crc32(f"{seed}:{name}".encode()), name))
    out: dict[str, str] = {}
    at = 0
    for split, take in zip(SPLITS, counts, strict=True):
        for name in ordered[at : at + take]:
            out[name] = split
        at += take
    return out


def partition(
    items: Iterable,
    key: Callable[[Any], str],
    ratio: Sequence[int] = DEFAULT_RATIO,
    seed: int = 0,
) -> dict[str, list]:
    """split any items into train / val / test lists, by the frame name `key` gives each one.

    items sharing a key stay together, so nothing leaks between splits; the assignment is `assign`'s
    (deterministic in key and seed), and `ratio` sets the sizes - `(80, 20, 0)` asks for no test
    split at all. the default ratio is the same one `assign` and `write_into` use.
    """
    items = list(items)
    by_name = assign((key(item) for item in items), ratio, seed)
    out: dict[str, list] = {split: [] for split in SPLITS}
    for item in items:
        out[by_name[key(item)]].append(item)
    return out


def of_rows(rows: Iterable[dict]) -> dict[tuple[str, str], str]:
    """(recording, frame) -> split, read back off a set's own rows.

    a row written before splits existed reads as `train`: the whole set was trained on, which is
    what those checkpoints actually did.
    """
    return {(r["recording"], r["frame"]): r.get("split", "train") for r in rows}


def keep(examples: list, rows: Iterable[dict], sessions_dir: Path, split: str) -> list:
    """the examples belonging to one split, matched back to their rows through the frame path"""
    if split not in SPLITS:
        raise ValueError(f"no such split {split!r} - one of {SPLITS}")
    lookup = of_rows(rows)
    wanted = []
    for example in examples:
        frames_dir = example.path.parent
        recording = str(frames_dir.parent.relative_to(sessions_dir))
        if lookup.get((recording, example.path.name), "train") == split:
            wanted.append(example)
    return wanted


def write_into(set_path: Path, ratio: Sequence[int] = DEFAULT_RATIO, seed: int = 0) -> dict:
    """stamp a split onto a set already on disk, and record it in the set's meta.

    for sets generated before splits existed. RE-SPLITTING A SET IS REFUSED - a checkpoint may
    already have been judged against its holdout, and moving it would silently invalidate that.
    """
    import json

    rows = [json.loads(line) for line in set_path.read_text().splitlines() if line.strip()]
    if any("split" in r for r in rows):
        raise ValueError(f"{set_path.name} already carries a split - refusing to redraw it")
    by_frame = assign((f"{r['recording']}/{r['frame']}" for r in rows), ratio, seed)
    counts = {"ratio": list(ratio), "rows": dict.fromkeys(SPLITS, 0), "frames": {}}
    for row in rows:
        row["split"] = by_frame[f"{row['recording']}/{row['frame']}"]
        counts["rows"][row["split"]] += 1
    counts["frames"] = {name: list(by_frame.values()).count(name) for name in SPLITS}
    write_jsonl_atomic(set_path, rows)

    meta_path = set_path.with_suffix(".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["split"] = counts
        write_json_atomic(meta_path, meta, indent=1)
    return counts


def main(argv: list[str] | None = None) -> int:
    """give an existing training set a train/val/test split, by frame"""
    import argparse

    from smolsmort.review import paths

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("sets", nargs="+", help="set names under training/sets, without .jsonl")
    parser.add_argument("--split", default="/".join(str(r) for r in DEFAULT_RATIO))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        ratio = parse_ratio(args.split)
    except ValueError as exc:
        parser.error(str(exc))

    failed = 0
    for name in args.sets:
        path = paths.DATASETS_DIR / f"{name}.jsonl"
        try:
            counts = write_into(path, ratio, args.seed)
        except (OSError, ValueError) as exc:
            print(f"{name}: {exc}")
            failed += 1
            continue
        frames, rows = counts["frames"], counts["rows"]
        print(f"{name}: " + "  ".join(f"{s} {frames[s]}f/{rows[s]}r" for s in SPLITS))
    return 1 if failed else 0
