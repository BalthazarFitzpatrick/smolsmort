"""the review tool's own layer over `smolsmort.backends`: naming a saved run, and reading back what
produced it.

`smolsmort.backends.sidecar` already writes a json beside a checkpoint that names which
`ModelBackend` wrote it, plus that backend's own facts (its classes, its box size) - see
`smolsmort/backends.py`. That file is the BACKEND's metadata. This module writes a second, review-side
file beside a NAMED save: the run that produced it - which training set, which options - because a
checkpoint picked by name off a list is meaningless without knowing what it was trained on and when.
This is the same problem the parent project's `save_checkpoint` solved with a `.classes.json` file
next to the weights; this is that idea, generalised to any backend.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROVENANCE_SUFFIX = ".provenance.json"


def provenance_path(path: Path) -> Path:
    """the json beside a named checkpoint that says what produced it -
    weights/a.pt -> weights/a.pt.provenance.json"""
    return path.with_name(path.name + PROVENANCE_SUFFIX)


def save_named(backend, weights, path: Path, **provenance) -> dict:
    """save `weights` under `path` through `backend`, then a provenance file beside it.

    `backend.save` already writes the backend/classes sidecar `smolsmort.backends` reads back by
    value; this adds the facts the seam itself does not carry - the training set, the settings the
    run actually used, when it happened - stamped with the wall-clock time unless a caller passes
    its own (tests do, so a run is reproducible in its assertions rather than racing the clock).
    """
    backend.save(weights, path)
    stamp = provenance.pop("saved", None) or datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"saved": stamp, **provenance}
    provenance_path(path).write_text(json.dumps(meta, indent=2))
    return {"name": path.name, "bytes": path.stat().st_size, **meta}


def load_provenance(path: Path) -> dict:
    """the provenance beside a named checkpoint, or empty when it predates this - the same
    "no sidecar is not an error" stance `smolsmort.backends.backend_of` takes for LEGACY checkpoints"""
    meta = provenance_path(path)
    if not meta.is_file():
        return {}
    try:
        return json.loads(meta.read_text())
    except json.JSONDecodeError:
        return {}
