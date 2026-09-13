"""what a class IS, defined by the person judging rather than hardcoded in five places.

WHAT THIS REPLACES. The fifteen nameplate classes were five primitives crossed with three states,
written out as literals in `review/state.py` AND again in `review_ui/app.js`, with four regexes
parsing the state back out of a label string. Nothing connected the two copies, so adding a class
meant finding all of them. Worse, assignment did not take the class you picked: you ticked a
SHORTLIST of plausible identities and the server chose between them by nearest measured RGB, which
cannot mean anything for a dimension a person invents.

THE SHAPE. A definition is up to three DIMENSIONS, each holding any number of MEMBERS. A class is
one member from each - the full cross-product - so three primitives, three player classes and three
states is twenty-seven classes. Nothing downstream cares how many: smolsmort derives its channel map
from the labels actually present, sorted, so the head is sized by the data rather than by a
constant.

`not a class` IS NOT A MEMBER, and that is deliberate. It is already a boolean on the pool record,
mutually exclusive with having a class - assigning one clears it and vice versa. Modelling it as a
member of some dimension would make "hostile and not a class" expressible, which is not a thing.

DEFINITIONS ARE VERSIONED AND OLD LABELS STAND. A pool record remembers the definition it was judged
under. Editing a definition never rewrites a judgement already made; instead a training set that
mixes two definitions is refused at promotion, naming both - the same guard that already refuses a
set mixing capture resolutions. The alternative, rewriting past labels to match a new carve-up,
quietly changes what you decided months ago.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from parent.config.profile import PROFILES_ROOT

# beside the screen and character profiles, because it is the same kind of thing: a named,
# hand-made description of how this project sees the world
DEFS_DIR = PROFILES_ROOT / "classes"

# three is what a person can hold in a right-click menu as columns, and the point at which the
# cross-product stops being readable - four dimensions of three is eighty-one classes
MAX_DIMENSIONS = 3

# what a tile is when it is not any class. never a member; see the module docstring
NOT_A_CLASS = "not a class"

# the separator between members in a written label. " / " rather than the old " (suffix)" form,
# which could only ever express one extra dimension and needed a regex to read back
JOIN = " / "

# WHAT AN UNANSWERED DIMENSION WRITES. a dimension with no members yet, or one nobody picked from,
# still occupies its position in the label - because `labelParts` splits the joined string
# POSITIONALLY, so a skipped value would shift every later one into the wrong dimension and a
# tile's size would read as its state. A placeholder keeps the form intact and says plainly that
# the axis was not answered, which is a different claim from any of its members.
UNSET = "n/a"


class ClassDefError(Exception):
    pass


def _slug(name: str) -> str:
    """a filename that cannot escape its directory or collide by punctuation alone"""
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not cleaned:
        raise ClassDefError(f"{name!r} has no usable characters in it")
    return cleaned


@dataclass
class Dimension:
    name: str
    members: list[str] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"name": self.name, "members": list(self.members)}


@dataclass
class ClassDef:
    name: str
    dimensions: list[Dimension] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return _slug(self.name)

    def classes(self) -> list[str]:
        """every class the cross-product produces, in a stable order.

        SORTED AT THE END, because the channel map downstream is built from the labels present
        sorted - so producing them in definition order here and letting them be re-sorted there
        would put the two out of step for no reason.
        """
        combos = [[]]
        for dimension in self.dimensions:
            if not dimension.members:
                continue
            combos = [combo + [member] for combo in combos for member in dimension.members]
        return sorted(JOIN.join(combo) for combo in combos if combo)

    def label_for(self, picked: dict[str, str]) -> str:
        """the label for one member per dimension, in the definition's own order.

        EVERY DIMENSION KEEPS ITS POSITION, answered or not. An unanswered one - no members yet, or
        none picked - writes `UNSET` rather than being skipped, which is what lets a definition be
        used while it is still being built. Skipping it instead was the old behaviour and it could
        not survive contact with `labelParts`, which splits positionally: drop a middle value and
        every later one lands in the wrong dimension.

        STILL REFUSES A LABEL THAT ANSWERS NOTHING. All-unset is not a judgement, it is the absence
        of one, and writing it would put a class in the channel map that says only "a tile exists".
        """
        parts = []
        for dimension in self.dimensions:
            member = picked.get(dimension.name)
            if not dimension.members or member is None:
                parts.append(UNSET)
                continue
            if member not in dimension.members:
                raise ClassDefError(f"{member!r} is not a member of {dimension.name!r}")
            parts.append(member)
        if not parts:
            raise ClassDefError("this definition has no dimensions")
        if all(part == UNSET for part in parts):
            raise ClassDefError("nothing was picked - a label that answers nothing is not a class")
        return JOIN.join(parts)

    def as_json(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "dimensions": [d.as_json() for d in self.dimensions],
            "classes": self.classes(),
        }

    def as_toml(self) -> str:
        lines = [
            "# a class definition: dimensions, and the members of each. a class is one member from",
            "# every dimension - see snapshot/review/classdefs.py for why.",
            f'name = "{self.name}"',
            "",
        ]
        for dimension in self.dimensions:
            members = ", ".join(f'"{m}"' for m in dimension.members)
            lines += ["[[dimension]]", f'name = "{dimension.name}"', f"members = [{members}]", ""]
        return "\n".join(lines)


def parse(text: str) -> ClassDef:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ClassDefError(f"not readable as toml: {exc}") from exc
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ClassDefError("a definition needs a name")
    dimensions = []
    for entry in raw.get("dimension", []):
        dim_name = str(entry.get("name") or "").strip()
        if not dim_name:
            raise ClassDefError("a dimension needs a name")
        members = [str(m).strip() for m in entry.get("members", []) if str(m).strip()]
        if len(set(members)) != len(members):
            raise ClassDefError(f"{dim_name!r} lists the same member twice")
        dimensions.append(Dimension(name=dim_name, members=members))
    if len(dimensions) > MAX_DIMENSIONS:
        raise ClassDefError(
            f"{len(dimensions)} dimensions; the most that reads is {MAX_DIMENSIONS}"
        )
    if len({d.name for d in dimensions}) != len(dimensions):
        raise ClassDefError("two dimensions share a name")
    return ClassDef(name=name, dimensions=dimensions)


def save(definition: ClassDef, directory: Path | None = None) -> Path:
    root = directory or DEFS_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{definition.slug}.toml"
    path.write_text(definition.as_toml())
    return path


def load(name: str, directory: Path | None = None) -> ClassDef:
    root = (directory or DEFS_DIR).resolve()
    path = (root / f"{_slug(name)}.toml").resolve()
    # RESOLVE THEN CONTAIN, never a blocklist on ".." - the name reaches here from an http route
    if root not in path.parents:
        raise ClassDefError(f"{name!r} is not inside the definitions directory")
    if not path.is_file():
        raise ClassDefError(f"no definition called {name!r}")
    return parse(path.read_text())


def available(directory: Path | None = None) -> list[str]:
    root = directory or DEFS_DIR
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.toml"))
