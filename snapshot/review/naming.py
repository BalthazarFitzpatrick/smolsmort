"""turning a recording or tile name into its parts.

Pure string work, and deliberately its own module: every other module in the review package
needs at least one of these, and leaving them beside the classes meant a class could not move
without dragging its neighbours along.
"""

from __future__ import annotations


def _flat(name: str) -> str:
    """a session name as ONE filename component.

    a nested recording is named by its path from sessions/ - "nameplate_pipeline_test/1" - and
    that slash would otherwise make labels/nameplate_pipeline_test/1.candidates.jsonl, a directory
    nobody created. flattening keeps every sidecar beside its siblings in labels/.
    """
    return name.replace("/", "__")


def _tile_tag(name: str) -> str:
    """the dataset a saved tile came from, off its own filename: "classic_seeds_k00005" ->
    "classic_seeds". empty for a bare "k00005" written before prefixes existed, which cannot be
    attributed to any dataset and must not be guessed into one.
    """
    stem = name.rsplit("k", 1)[0]
    return stem[:-1] if stem.endswith("_") else stem


def _tile_index(name: str) -> int:
    """candidate index encoded in a saved tile's own filename - "k00005" or, once a session
    prefix is in play (see ReviewState.session_tag), "sessiontag_k00005". always the digits
    after the LAST "k", never a fixed slice offset, since the prefix's own length varies.
    """
    return int(name.rsplit("k", 1)[-1])
