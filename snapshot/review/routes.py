"""every http route the review tool answers, and nothing else.

THIN ON PURPOSE. a route parses its request, calls one method on a state object, and serialises
what comes back. Logic that crept in here could only be tested through a socket; in a state class
it is testable directly, which is why fifty of this tool's tests need no server at all.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from parent.config.profile import PROFILES_DIR
from parent.imitation.record import resolve_session_paths
from snapshot.review import classdefs, housekeeping, maps, nav, paths
from snapshot.review.camera import CameraState
from snapshot.review.control import ControlState
from snapshot.review.naming import _flat
from snapshot.review.nav import NavState
from snapshot.review.playback_state import PlaybackState
from snapshot.review.state import ReviewState, _session_frames_for
from snapshot.review.train import TrainState
from snapshot.review.verify import VerifyState
from snapshot.vision.plate_templates import TemplateError

# where a working set goes before it has been told which screen it belongs to. NOT a screen, so it
# must never be counted as one when deciding what to open
UNASSIGNED_SCREEN = "unassigned"


def _startup_screen() -> str:
    """always the holding name: the tab opens bound to NOTHING until a screen is picked.

    IT USED TO GUESS, and a wrong guess is indistinguishable from lost work. It opened whichever
    directory it decided on and adopted that screen's only character - so landing on a holding
    directory whose state had nothing marked showed an empty element list beside two screens
    holding 7 and 24 marks, and read as the work having been destroyed.

    Balthazar Fitzpatrick, after seeing exactly that: "Fix it to open on nothing, until I use the dropdowns."
    Nothing selected is a state the page can state plainly. A guess is not.
    """
    return UNASSIGNED_SCREEN


def index_html() -> str:
    """the page skeleton, read fresh from review_ui/index.html on every request.

    NOT CACHED ON PURPOSE. it used to be a 3,472-line string literal in this file, which meant no
    editor help and no syntax checking - and every change was a text substitution that could
    silently no-op or break the surrounding structure. Reading per request costs one file read and
    means a markup edit shows up on the next refresh, the same as the css and js beside it.
    """
    return (paths.REVIEW_UI_DIR / "index.html").read_text(encoding="utf-8")


def _interface_names(interface, payload: dict) -> None:
    """set realm/character/screen, and move the working set to match.

    ORDER IS THE WHOLE FUNCTION. every path derives WHERE to write from these three names, so the
    outgoing working set must be saved BEFORE any of them moves. Saving afterwards files it under
    the incoming names - both halves of that were observed on Balthazar Fitzpatrick's real working set: the old
    screen's file stamped with the new screen's name, and a character's marks shown to the next
    character.
    """
    wants = {
        field: (payload[field] or "").strip()
        for field in ("realm", "character", "profile_name")
        if field in payload
    }
    changed = [field for field, value in wants.items() if value != getattr(interface, field)]
    # persisted the moment they are picked, NOT only when a profile is generated - a name that
    # lives in the browser alone is lost on reload
    if changed and interface.has_work:
        interface.save()
    for field, value in wants.items():
        setattr(interface, field, value)

    # SCREENSHOTS AND MARKS BELONG TO A SCREEN. picking a different screen profile swaps the whole
    # working set - a screen with nothing marked yet shows no screenshots and no cycle buttons,
    # rather than the previous screen's captures with rects that cannot mean anything on them
    target = (
        paths.INTERFACE_SHOTS / interface.profile_name
        if interface.profile_name
        else interface.shots_dir
    )
    if target != interface.shots_dir:
        interface.rebind(target, save_first=False)  # already saved above, under the outgoing names
    elif "realm" in changed or "character" in changed:
        # SAME SCREEN, DIFFERENT CHARACTER. rebind returns early on an unchanged directory, so
        # without this branch the incoming character keeps looking at the outgoing one's marks
        interface.reload_for_character()


def _interface_generate(interface, payload: dict) -> dict:
    """write the two profiles the tab has enough information for.

    THE SCREEN PROFILE IS BARELY WRITTEN AT ALL, and never overwritten. It holds the resolution
    and the [readout] block, and the readout costs a live gameplay session to calibrate - so an
    existing one is left exactly alone and only a genuinely new screen gets a stub. Everything
    positional goes to the server_character profile, per Balthazar Fitzpatrick: "screen will have screen name,
    resolution, and carry the addon location. EVERYTHING else goes into the server_character
    profile."

    the resolution comes from the SCREENSHOT the rects were marked on, not from anything typed - a
    rect measured on a 3420x2224 capture means nothing against a differently sized profile, and
    asking a human to retype the number invites the one typo that invalidates every rect.
    """
    from PIL import Image

    interface.realm = (payload.get("realm") or "").strip()
    interface.character = (payload.get("character") or "").strip()
    interface.profile_name = (payload.get("profile_name") or "").strip()
    if not interface.profile_name:
        return {"error": "a screen profile name is needed"}
    if not (interface.realm and interface.character):
        return {"error": "a server and a character are needed - the rects belong to them"}

    marked = [e for e in interface.elements if e.marked]
    if not marked:
        return {"error": "nothing marked yet"}

    shot = payload.get("screenshot") or marked[0].screenshot
    if not shot:
        return {"error": "no screenshot to take the resolution from"}
    path = interface.shots_dir / Path(shot).name
    if not path.is_file():
        return {"error": f"no screenshot named {shot!r}"}
    with Image.open(path) as handle:
        resolution = handle.size

    written = []
    screen_path = PROFILES_DIR / f"{interface.profile_name}.toml"
    if screen_path.exists():
        # THE GUARD THAT WAS MISSING, and its absence cost a whole profile. nothing checked that
        # the screen picked matches the capture marked on, so a session with the picker left on
        # mac-m4-msi wrote RETINA rects into an MSI profile - and since that profile also carried
        # retina's resolution, nothing downstream could tell either. both were deleted 2026-09-01
        try:
            declared = tomllib.loads(screen_path.read_text()).get("reference_resolution")
        except (OSError, tomllib.TOMLDecodeError):
            declared = None
        if declared and tuple(declared) != tuple(resolution):
            return {
                "error": (
                    f"{interface.profile_name} is a {declared[0]}x{declared[1]} screen but "
                    f"{shot} is {resolution[0]}x{resolution[1]}. a rect measured on one means "
                    "nothing on the other - pick the screen this was captured on, or name a new one"
                )
            }
        # LEFT ALONE. it carries the readout, which this tab cannot reproduce from a screenshot
        written.append(f"{screen_path.name} (existing, untouched)")
    else:
        screen_path.write_text(interface.profile_toml(resolution))
        written.append(screen_path.name)

    # A CHARACTER IS A FOLDER, holding its toml and the screenshots its rects were measured on.
    # every rect was drawn on a particular frame, and a rect whose frame is gone can be re-checked
    # by nobody - which matters most exactly when something reads wrong months later.
    chars_dir = PROFILES_DIR.parent / "characters"
    # THE SCREEN IS PART OF THE KEY. every rect, and every ability's brightness cut, was measured
    # against one screen - and Balthazar Fitzpatrick arranges the ui differently on different screen estate, so
    # they do not even scale between screens linearly. keying on realm-character alone gave one
    # slot for all of them, so marking a second screen would have overwritten the first. Balthazar Fitzpatrick:
    # "I am totally fine with maintaining two profiles for two screens INCLUDING all the abilities
    # and spells and interface items"
    key = f"{interface.realm}-{interface.character}-{interface.profile_name}".lower()
    char_dir = chars_dir / key
    (char_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    char_path = char_dir / "character.toml"
    char_path.write_text(interface.character_toml())
    written.append(f"{key}/character.toml")

    # copy ONLY the screenshots something actually references. the tab's upload area is a working
    # space and usually holds more than the finished profile depends on
    referenced = set()
    for element in interface.elements:
        if element.screenshot:
            referenced.add(element.screenshot)
        for example in element.examples.values():
            if example.get("screenshot"):
                referenced.add(example["screenshot"])
    copied = 0
    for shot in sorted(referenced):
        source = interface.shots_dir / Path(shot).name
        if source.is_file():
            shutil.copy2(source, char_dir / "screenshots" / source.name)
            copied += 1
    if copied:
        written.append(f"{copied} screenshots")

    # a partial profile is a legitimate intermediate, so this reports rather than refuses - but it
    # must never be silent about which core rects are absent
    missing = interface.missing_required()
    warning = f" STILL MISSING (required): {', '.join(missing)}." if missing else ""
    readout = (
        ""
        if screen_path.exists() and "[readout]" in screen_path.read_text()
        else f" no [readout] yet - run wt-calibrate-addon-readout --profile "
        f"{interface.profile_name}"
    )
    return {
        "message": (
            f"wrote {', '.join(written)} from {len(marked)} marked elements at "
            f"{resolution[0]}x{resolution[1]}.{warning}{readout}"
        )
    }


def make_handler(
    state: ReviewState,
    trainer: TrainState,
    verifier: VerifyState,
    interface=None,
    playback_state: PlaybackState | None = None,
    nav_state: NavState | None = None,
    camera_state: CameraState | None = None,
    control_state: ControlState | None = None,
):
    playback_state = playback_state or PlaybackState()
    nav_state = nav_state or NavState()
    camera_state = camera_state or CameraState()
    control_state = control_state or ControlState()
    from snapshot.tools.interface_state import InterfaceError, InterfaceState

    if interface is None:
        interface = InterfaceState(shots_dir=paths.INTERFACE_SHOTS / _startup_screen())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _interface_post(self):
            """the Interface tab's four writes, all under /api/interface/.

            every one returns the WHOLE state rather than a delta, so the page cannot drift out of
            step with the server after a failed request - there is one small state object and
            re-sending it costs nothing.
            """
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            action = self.path.split("?")[0][len("/api/interface/") :]

            try:
                if action == "shot":
                    name = unquote(parse_qs(urlparse(self.path).query).get("name", ["shot"])[0])
                    # normalise whatever was uploaded to png so the browser can always show it
                    from io import BytesIO

                    from PIL import Image

                    buf = BytesIO()
                    Image.open(BytesIO(raw)).convert("RGB").save(buf, format="PNG")
                    saved = interface.save_screenshot(name, buf.getvalue())
                    self._json({"name": saved})
                    return

                payload = json.loads(raw) if raw else {}
                with interface.lock:
                    if action == "add":
                        interface.add(
                            payload.get("name", ""),
                            payload.get("mode", "rect"),
                            payload.get("keybind", ""),
                            bool(payload.get("ability")),
                        )
                    elif action == "remove":
                        interface.remove(payload.get("name", ""))
                    elif action == "mark":
                        interface.mark(
                            payload.get("name", ""),
                            payload.get("shape") or {},
                            payload.get("screenshot"),
                            payload.get("state") or None,
                        )
                    elif action == "names":
                        _interface_names(interface, payload)
                    elif action == "clear-marks":
                        interface.clear_marks()
                    elif action == "keybind":
                        interface.set_keybind(payload.get("name", ""), payload.get("keybind", ""))
                    elif action == "add-state":
                        interface.add_state(
                            payload.get("name", ""),
                            payload.get("state", ""),
                            bool(payload.get("required")),
                        )
                    elif action == "remove-shot":
                        orphaned = interface.remove_screenshot(payload.get("name", ""))
                        interface.save()
                        data = interface.to_json()
                        data["orphaned"] = orphaned
                        self._json(data)
                        return
                    elif action == "generate":
                        self._json(_interface_generate(interface, payload))
                        return
                    else:
                        self._json({"error": f"unknown action {action!r}"}, status=404)
                        return
                interface.save()
                self._json(interface.to_json())
            except (InterfaceError, ValueError, OSError) as exc:
                self._json({"error": str(exc)}, status=400)

        def _no_store(self):
            """nothing this server sends is ever worth caching - the page and its stylesheet
            change on every restart while the tool is being worked on, and a browser is free to
            reuse a response with no cache headers at all. that is what made a reload show a
            stale page while a fresh profile always looked correct"""
            self.send_header("Cache-Control", "no-store, must-revalidate")

        def _json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._no_store()
            self.end_headers()
            self.wfile.write(body)

        def _png(self, data: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            # THE IMAGES NEED THIS AS MUCH AS THE JSON DOES, and it was missing here alone. a
            # tile is re-cut in place under an unchanged url, so a browser free to reuse its
            # cached copy shows the OLD crop after a successful realign - the save worked, the
            # picture did not move, and it reads as "fix alignment does not save reliably"
            self._no_store()
            self.end_headers()
            self.wfile.write(data)

        # SEVEN ROUTES THAT WERE THE SAME SEVEN LINES. each took a name out of the path, asked
        # something for bytes, sent 404 when there were none, and sent the bytes otherwise - copied
        # out once per prefix, which is seven places for one behaviour to drift.
        IMAGE_ROUTES = {
            "/draw-frame/": (lambda name: state.frame_bytes(name), "image/jpeg"),
            "/interface-shot/": (lambda name: interface.screenshot_bytes(name), "image/png"),
            "/library-thumb/": (lambda name: state.library_thumb_bytes(name), "image/png"),
            "/unsorted-thumb/": (lambda name: state.unsorted_thumb_bytes(name), "image/png"),
            "/tile/": (lambda name: state.pool_tile_bytes(name), "image/png"),
            "/crop/": (lambda name: state.crop_bytes(int(name)), "image/png"),
            "/guide/": (lambda name: state.guide_bytes(name), "image/png"),
            "/zone-map/": (lambda name: nav.zone_map_bytes(name), "image/jpeg"),
            "/map-layer/": (lambda name: maps.layer_bytes(name), "image/png"),
        }

        def _post_pool_decisions(self) -> bool:
            """judgements about tiles already in the pool: size, review state, labels, undo.

            These are the routes the discard/promote tab fires while a human works through crops,
            and they share nothing with training or binding beyond living in the same handler.
            Returns True when it handled the path.
            """
            if self.path == "/api/tile-size":
                payload = self._payload()
                state.uniform_height = max(
                    4, state.uniform_height + int(payload.get("height_delta", 0))
                )
                self._json({"width": state.uniform_width, "height": state.uniform_height})
                return True
            if self.path == "/api/height-delta":
                payload = self._payload()
                delta = state.set_height_delta(payload["index"], int(payload.get("delta", 0)))
                self._json(
                    {
                        "height_delta": delta,
                        "effective_height": state._effective_height(payload["index"]),
                    }
                )
                return True
            if self.path == "/api/smart-center":
                payload = self._payload()
                names = [str(n) for n in payload.get("names", [])]
                if not names:
                    self._json({"error": "nothing selected"})
                    return True
                self._json(state.smart_center(names))
                return True
            if self.path == "/api/reviewed":
                payload = self._payload()
                state.mark_reviewed(payload["index"])
                self._json({"ok": True})
                return True
            if self.path == "/api/nav-keys":
                # the bindings the recorder will pick up next time it starts - see record.save_nav_keys
                from parent.imitation.record import save_nav_keys

                try:
                    self._json({"ok": True, "keys": save_nav_keys(self._payload())})
                except ValueError as exc:
                    self._json({"error": str(exc)}, status=400)
                return True
            if self.path == "/api/nav-keys/listen":
                # the recorder's own listener, for a button the browser never receives - see
                # input_listener. blocks until an input, a cancel or `seconds` run out
                from parent.imitation import input_listener

                payload = self._payload()
                if payload.get("cancel"):
                    input_listener.cancel_listening()
                    self._json({"ok": True})
                    return True
                try:
                    seconds = min(max(float(payload["seconds"]), 1.0), 15.0)
                except (KeyError, TypeError, ValueError):
                    self._json({"error": "seconds is required"}, status=400)
                    return True
                self._json(input_listener.listen_for_combo(seconds))
                return True

            if self.path == "/api/nav-edit":
                # the lines for ONE map of one session, replaced wholesale - see nav.write_edits
                payload = self._payload()
                root = paths.SESSIONS_DIR.resolve()
                try:
                    asked = (root / payload["session"]).resolve()
                    asked.relative_to(root)
                except (ValueError, OSError, KeyError):
                    self._json({"error": "session is not inside the recordings directory"}, 400)
                    return True
                stored = nav.write_edits(asked, int(payload["map_id"]), payload.get("lines", []))
                self._json({"ok": True, "maps": sorted(stored.get("maps", {}))})
                return True

            if self.path == "/api/exclude":
                payload = self._payload()
                # DECLARED, NOT TOGGLED - see state.set_excluded. a caller that omits the field is
                # a stale page still expecting a flip, and answering it by guessing would invert
                # half of whatever it sent; say what is missing instead
                if "excluded" not in payload:
                    self._json(
                        {"error": "exclude needs an 'excluded' boolean - it no longer toggles"}, 400
                    )
                    return True
                # A SELECTION IS ONE PRESS, SO IT IS ONE REQUEST. the page used to fan out a
                # POST per tile: eighty tiles meant eighty round trips through a browser that
                # opens six connections, and one rejection left the grid unrepainted while the
                # writes that DID land stayed - which read as "the button does nothing" until a
                # refresh showed them marked.
                names = payload.get("names") or [payload.get("name")]
                wanted = bool(payload["excluded"])
                done, unknown = [], []
                for one in names:
                    if one is None:
                        continue
                    if state.set_excluded(one, wanted) is None:
                        unknown.append(one)
                    else:
                        done.append(one)
                if not done and unknown:
                    self._json({"error": "no such tile", "unknown": unknown}, 404)
                    return True
                self._json({"ok": True, "excluded": wanted, "done": done, "unknown": unknown})
                return True
            if self.path == "/api/manual-label":
                payload = self._payload()
                # ONE MEMBER PER DIMENSION, and the definition says which those are. this used to
                # take a shortlist and pick between its entries by nearest measured RGB, which
                # cannot mean anything for a dimension the user invented
                try:
                    definition = classdefs.load(payload["definition"])
                    label = definition.label_for(payload.get("picked", {}))
                except classdefs.ClassDefError as exc:
                    self._json({"error": str(exc)}, 400)
                    return True
                # same press, same reason - see /api/exclude above
                names = payload.get("names") or [payload.get("name")]
                written = None
                for one in names:
                    if one is not None:
                        written = state.apply_manual_label(one, label, definition.slug)
                if written is None:
                    self._json({"error": f"no tile called {payload['name']}"}, 400)
                    return True
                self._json({"ok": True, "label": written})
                return True
            if self.path == "/api/classdef-save":
                payload = self._payload()
                try:
                    definition = classdefs.ClassDef(
                        name=str(payload.get("name", "")),
                        dimensions=[
                            classdefs.Dimension(name=d["name"], members=list(d.get("members", [])))
                            for d in payload.get("dimensions", [])
                        ],
                    )
                    classdefs.parse(definition.as_toml())  # same validation a loaded one gets
                    classdefs.save(definition)
                except (classdefs.ClassDefError, KeyError, TypeError) as exc:
                    self._json({"error": str(exc)}, 400)
                    return True
                self._json({"ok": True, **definition.as_json()})
                return True
            return False

        def _post_training(self) -> bool:
            """the train and sweep routes: everything that touches the model or the pool it eats.

            SEVEN ROUTES THAT BELONG TOGETHER, lifted out of a 235-line do_POST so the training
            half can be read without scrolling past binding, playback and pool decisions. Returns
            True when it handled the path, so do_POST stays a list of what handles what.
            """
            if self.path == "/api/load-checkpoint":
                payload = self._payload()
                self._json(trainer.load_checkpoint(str(payload.get("name", ""))))
                return True
            if self.path == "/api/save-checkpoint":
                payload = self._payload()
                self._json(
                    trainer.save_checkpoint(payload.get("name"), str(payload.get("folder") or ""))
                )
                return True
            if self.path == "/api/sweep-send":
                payload = self._payload()
                self._json(trainer.send_sweep(float(payload.get("min_score", 0.5))))
                return True
            if self.path == "/api/sweep-start":
                payload = self._payload()
                self._json(
                    trainer.start_sweep(
                        str(payload.get("recording", "")),
                        max(1, min(100, int(payload.get("percent", 100)))),
                        float(payload.get("min_score", 0.5)),
                    )
                )
                return True
            if self.path == "/api/train-bind":
                payload = self._payload()
                self._json(trainer.bind(str(payload.get("name", ""))))
                return True
            if self.path == "/api/crop-settings":
                payload = self._payload()
                self._json(state.set_crop_settings(payload.get("pad_x"), payload.get("pad_y")))
                return True
            if self.path == "/api/train-abort":
                self._json(trainer.abort())
                return True
            if self.path == "/api/train-start":
                payload = self._payload()
                epochs = max(1, min(2000, int(payload.get("epochs", 200))))
                batch = max(1, min(64, int(payload.get("batch", 8))))
                crop = payload.get("crop")
                # CLAMPED LIKE THE REST. a learning rate is the one hyperparameter that can waste a
                # long run outright - too high and the loss diverges in the first epochs - so the
                # box refuses a value outside the range that ever makes sense rather than trusting
                # what was typed
                rate = payload.get("learning_rate")
                rate = None if rate in (None, "") else max(1e-6, min(1e-1, float(rate)))
                seed = payload.get("seed")
                seed = None if seed in (None, "") else int(seed)
                self._json(
                    trainer.start(
                        epochs,
                        batch,
                        None if crop is None else int(crop),
                        learning_rate=rate,
                        seed=seed,
                    )
                )
            return False

        def _payload(self) -> dict:
            """the request's json body, as a dict.

            TWENTY ROUTES OPENED WITH THE SAME TWO LINES - read Content-Length, json.loads the
            body - which is twenty chances to forget one and twenty places to fix a bug in the
            other. An empty body is {} rather than an error: several routes legitimately take none,
            and they were each writing their own guard for it.
            """
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length)) or {}
            except json.JSONDecodeError:
                return {}

        def _serve_image(self, path: str) -> bool:
            """serve one of the byte routes, or return False so the caller keeps looking.

            the getters raise as readily as they return None - guide_bytes goes to disk - so both
            are treated as "not there", which is what a 404 means to the page either way.
            """
            for prefix, (getter, content_type) in self.IMAGE_ROUTES.items():
                if not path.startswith(prefix):
                    continue
                try:
                    data = getter(unquote(path[len(prefix) :]))
                except (TemplateError, OSError, ValueError, IndexError, KeyError):
                    # IndexError is not hypothetical: /crop/<n> indexes the bound candidate list,
                    # and /crop/999999 used to raise straight out of the handler and kill the
                    # connection - curl saw no response at all rather than a 404. Nothing in the
                    # page asks for an out-of-range crop, which is why it survived; a stale tab
                    # after rebinding does exactly that.
                    data = None
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return True
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self._no_store()
                self.end_headers()
                self.wfile.write(data)
                return True
            return False

        def _serve_ui_asset(self, name: str) -> None:
            """serves one file: this tool's own from review_ui/, the shared ones from ui_base.

            THE SHARED FILES ARE A DEPENDENCY NOW, not copies. base.css, menu.js, shell.js and
            align.js used to be copied in with a drift test guarding them, because ui_base had no
            remote and a path dependency cannot resolve inside ci-parity's tracked-files-only export
            - which is what broke the push. ui_base is published and pinned in pyproject.toml, so
            the copies and the drift test are gone: a drift test is a workaround for not having a
            remote, and the remote is the real fix.

            ui_base is tried FIRST so a stale local copy of a shared file can never shadow the
            pinned one. read_asset does its own containment check; the local branch keeps ours,
            where resolve() collapses any "../" before the check so an encoded traversal cannot
            escape the directory however it is spelled in the url.
            """
            data = None
            try:
                from ui_base import UiBaseError, read_asset
            except ImportError:  # the package is a dependency, but a dev checkout may lack it
                pass
            else:
                try:
                    data = read_asset(name)
                except UiBaseError:
                    # not one of ui_base's - this tool's own file, handled below. read_asset
                    # raises rather than returning None, and catching the wrong exception here
                    # would take the handler down on every request for app.js
                    data = None
            if data is None:
                target = (paths.REVIEW_UI_DIR / name).resolve()
                if paths.REVIEW_UI_DIR.resolve() in target.parents and target.is_file():
                    data = target.read_bytes()
            if data is None:
                self.send_response(404)
                self.end_headers()
                return
            content_type = {".css": "text/css", ".js": "application/javascript"}.get(
                Path(name).suffix, "text/plain"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self._no_store()
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            """dispatch, with the same guard do_POST has and for the same reason.

            MEASURED: /api/playback-frame?t=abc, /api/draw-frames?percent=abc and
            /api/clusters?k=abc each parse a query parameter with float() or int() and answered
            NOTHING AT ALL - the socket closed. A query string is the easiest thing in the whole
            tool for a stale tab or a hand-typed url to get wrong.
            """
            try:
                self._dispatch_get()
            except (KeyError, TypeError, ValueError) as exc:
                self._json({"error": f"bad request for {self.path}: {exc}"}, 400)

        def _dispatch_get(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = (
                    index_html()
                    .replace("{page_size}", str(paths.PAGE_SIZE))
                    .replace("{margin_x}", str(paths.MARGIN_X))
                    .replace("{margin_y}", str(paths.MARGIN_Y))
                    .replace("{library}", str(state.library))
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self._no_store()
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path.startswith("/ui/"):
                self._serve_ui_asset(unquote(parsed.path[len("/ui/") :]))
            elif parsed.path == "/api/bindable-recordings":
                self._json(state.bindable_recordings())
            elif parsed.path == "/api/dir-list":
                self._json(state.dir_list())
            elif parsed.path == "/api/source-state":
                tag = parse_qs(parsed.query).get("tag", [""])[0]
                self._json(state.pool_source_state(tag))
            elif parsed.path == "/api/dir-tree":
                under = parse_qs(parsed.query).get("under", [""])[0]
                self._json(state.dir_tree(under or None))
            elif parsed.path == "/api/script-list":
                self._json(state.script_list())
            elif parsed.path == "/api/page":
                q = parse_qs(parsed.query)
                page = int(q.get("page", ["0"])[0])
                size = int(q.get("size", [str(paths.PAGE_SIZE)])[0])
                hide_done = q.get("hide_done", ["0"])[0] == "1"
                self._json(state.page(page, size, hide_done))
            elif parsed.path == "/api/tile-size":
                self._json({"width": state.uniform_width, "height": state.uniform_height})
            elif parsed.path == "/api/clusters":
                query = parse_qs(parsed.query)
                # NO `k` ANY MORE. It set the number of colour clusters, and the grid sorts by
                # class now - there is nothing to choose. A stale tab still sending one is simply
                # ignored rather than refused.
                #
                # which tiles to show: "unsorted" the kept-but-unpromoted pool (the default),
                # "library" the promoted templates grouped by their own label, "none" an empty
                # grid for reviewing a fresh dataset unobstructed
                source = query.get("source", ["unsorted"])[0]
                if source == "library":
                    self._json(state.library_clusters())
                elif source == "none":
                    self._json({"clusters": [], "source": "none"})
                else:
                    self._json(state.clusters())
            elif parsed.path == "/api/housekeeping-recordings":
                orphaned_sets = housekeeping.orphaned_sets()
                self._json(
                    {
                        "recordings": [vars(r) for r in housekeeping.all_recordings()],
                        "orphaned_sets": sorted(orphaned_sets),
                    }
                )
            elif parsed.path == "/api/housekeeping-assets":
                tag = parse_qs(parsed.query).get("tag", [""])[0]
                self._json(housekeeping.assets_for(tag))
            elif parsed.path == "/api/interface":
                self._json(interface.to_json())
            elif parsed.path == "/api/openable-datasets":
                self._json(state.openable_datasets())
            elif parsed.path == "/api/pool-box":
                self._json(state.pool_box_info(parse_qs(parsed.query).get("name", [""])[0]))
            elif parsed.path == "/api/pool-sources":
                self._json(state.pool_sources())
            elif parsed.path == "/api/training-sets":
                self._json(state.training_sets())
            elif parsed.path == "/api/draw-frames":
                query = parse_qs(parsed.query)
                percent = query.get("percent", [None])[0]
                self._json(
                    state.draw_frames(
                        int(query.get("sample", ["40"])[0]),
                        int(percent) if percent else None,
                    )
                )
            elif self._serve_image(parsed.path):
                pass  # handled - see IMAGE_ROUTES
            elif parsed.path == "/api/map-manifest":
                # one manifest places every layer, so the page asks once and then only for images
                self._json(maps.manifest_json(parse_qs(parsed.query).get("map", [""])[0]))
            elif parsed.path == "/api/map-markers":
                self._json(maps.markers_json(parse_qs(parsed.query).get("map", [""])[0]))
            elif parsed.path == "/api/maps":
                self._json(maps.maps_json())
            elif parsed.path == "/api/nav-heat":
                query = parse_qs(parsed.query)
                names = [n for n in query.get("sessions", [""])[0].split(",") if n]
                wanted_map = query.get("map", [""])[0]
                yards = query.get("yards", [""])[0]
                self._json(
                    nav_state.heat(
                        names,
                        int(wanted_map) if wanted_map else None,
                        float(yards) if yards else nav.DEFAULT_CELL_YARDS,
                    )
                )
            elif parsed.path == "/api/nav-keys":
                from parent.imitation.nav_bindings import DPI_BUTTON
                from parent.imitation.record import DEFAULT_NAV_KEYS, load_nav_keys

                self._json(
                    {
                        "keys": load_nav_keys(),
                        "defaults": DEFAULT_NAV_KEYS,
                        "dpi_button": DPI_BUTTON,
                    }
                )
            elif parsed.path == "/api/nav-sessions":
                self._json(nav_state.sessions())
            elif parsed.path == "/api/nav-latest":
                self._json({"name": nav.latest_session()})
            elif parsed.path == "/api/nav":
                # no session named means the live one, which is what the tab wants on first load
                wanted = parse_qs(parsed.query).get("session", [""])[0] or nav.latest_session()
                self._json(nav_state.walk(wanted) if wanted else {"maps": [], "size": 0})

            elif parsed.path == "/api/classdefs":
                wanted = parse_qs(parsed.query).get("name", [""])[0]
                names = classdefs.available()
                if not wanted:
                    self._json({"definitions": names, "active": names[0] if names else None})
                    return
                try:
                    self._json(classdefs.load(wanted).as_json())
                except classdefs.ClassDefError as exc:
                    self._json({"error": str(exc)}, 400)

            elif parsed.path == "/api/camera":
                # no session named means the live one, same as /api/nav - the tab is watched while
                # the sitting is being recorded, so that is the case worth defaulting to
                wanted = parse_qs(parsed.query).get("session", [""])[0] or camera_state.latest()
                self._json(camera_state.sitting(wanted) if wanted else {"cycles": [], "size": 0})
            elif parsed.path == "/api/control":
                # nothing to name: one sink file, written by whichever wt-overlay-shadow process
                # is running - see tools/review/control.py for why the server never captures this
                # itself
                self._json(control_state.snapshot())
            elif parsed.path == "/api/playback-sessions":
                self._json(playback_state.sessions())
            elif parsed.path == "/api/playback-frame":
                at = float(parse_qs(parsed.query).get("t", ["0"])[0])
                self._json(playback_state.frame(at))
            elif parsed.path == "/api/vlm-info":
                self._json(verifier.info())
            elif parsed.path == "/api/vlm-status":
                self._json(verifier.status())
            elif parsed.path.startswith("/vlm-crop/"):
                # 204 rather than 404 on purpose: the page polls this while the vlm works, and a
                # missing crop means "not yet", not "wrong url"
                index = int(unquote(parsed.path[len("/vlm-crop/") :].split("?")[0]))
                data = verifier.crop_bytes(index)
                if data is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                self._png(data)
            elif parsed.path == "/api/crop-settings":
                self._json(state.crop_settings())
            elif parsed.path == "/api/window-floor":
                self._json(trainer.window_floor())
            elif parsed.path == "/api/train-info":
                self._json(trainer.info())
            elif parsed.path == "/api/train-status":
                self._json(trainer.status())
            elif parsed.path == "/api/peak-distribution":
                self._json(trainer.peak_distribution())
            elif parsed.path == "/api/sweep-recordings":
                self._json(trainer.sweep_recordings())
            elif parsed.path == "/api/sweep-status":
                self._json(trainer.sweep_status())
            elif parsed.path == "/api/saved-checkpoints":
                self._json(trainer.saved_checkpoints())
            elif parsed.path == "/api/checkpoint-folders":
                under = parse_qs(parsed.query).get("under", [""])[0]
                self._json(trainer.checkpoint_folders(under))
            elif parsed.path == "/api/train-classes":
                self._json(trainer.class_names())
            elif parsed.path == "/api/train-frames":
                self._json({"frames": trainer.frames()})
            elif parsed.path.startswith("/train-overlay/"):
                index = int(unquote(parsed.path[len("/train-overlay/") :].split("?")[0]))
                data = trainer.overlay_bytes(
                    index, int(parse_qs(parsed.query).get("channel", ["-1"])[0])
                )
                if data is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                self._png(data)
            elif parsed.path.startswith("/tile/"):
                index = int(unquote(parsed.path[len("/tile/") :].split("?")[0]))
                q = parse_qs(parsed.query)
                rect = None
                if all(k in q for k in ("left", "top", "width", "height")):
                    rect = {k: int(float(q[k][0])) for k in ("left", "top", "width", "height")}
                data = state.tile_bytes(index, rect)
                if data is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                self._png(data)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            """dispatch, with one guard so a bad request cannot kill the connection.

            A ROUTE THAT RAISES ANSWERS NOTHING AT ALL - not a 500, no response, the socket just
            closes and the page sees a network error it cannot explain. Measured on four routes
            (/api/height-delta, /api/reviewed, /api/exclude, /api/manual-label): each does
            payload["index"] and a request without one raises KeyError straight out of the handler.
            Nothing in the page sends a malformed body, which is why it survived; a stale tab, a
            retried request or anything poking the api by hand does exactly that.

            Deliberately narrow: a bad REQUEST becomes a 400 that names the problem, while a
            genuine fault inside the tool still propagates and is visible in the log rather than
            being dressed up as the caller's mistake.
            """
            try:
                self._dispatch_post()
            except (KeyError, TypeError, ValueError) as exc:
                self._json({"error": f"bad request for {self.path}: {exc}"}, 400)

        def _dispatch_post(self):
            if self.path.startswith("/api/interface/"):
                self._interface_post()
                return
            if self.path == "/api/vlm-start":
                payload = self._payload()
                frames = max(1, min(50, int(payload.get("frames", 3))))
                peaks = max(1, min(20, int(payload.get("peaks", 4))))
                self._json(verifier.start(frames, peaks))
                return
            if self._post_training():
                return
                return
            if self.path == "/api/set-bases":
                payload = self._payload()
                self._json({"ok": True, "bases": state.set_bases(payload)})
                return
            if self.path == "/api/playback-open":
                payload = self._payload()
                self._json(playback_state.open(payload.get("name", "")))
                return
            if self.path == "/api/unbind-recording":
                self._json(state.unbind())
                return
            if self.path == "/api/bind-recording":
                payload = self._payload()
                kind, name = payload.get("kind"), payload.get("name", "")
                try:
                    if kind == "dataset":
                        candidates_path = paths.LABELS_DIR / f"{_flat(name)}.candidates.jsonl"
                        if not candidates_path.exists():
                            raise FileNotFoundError(candidates_path)
                        candidates = [
                            json.loads(line)
                            for line in candidates_path.read_text().splitlines()
                            if line.strip()
                        ]
                        frames_dir = _session_frames_for(name)
                        if frames_dir is None:
                            raise FileNotFoundError(paths.SESSIONS_DIR / f"{name}/frames")
                        decisions_path = candidates_path.with_suffix(".decisions.json")
                        state.rebind(frames_dir, candidates, decisions_path, name)
                    elif kind == "session":
                        _, frames_dir = resolve_session_paths(paths.SESSIONS_DIR / name)
                        if not frames_dir.is_dir():
                            raise FileNotFoundError(frames_dir)
                        decisions_path = (
                            paths.LABELS_DIR / f"{_flat(name)}.candidates.decisions.json"
                        )
                        state.rebind(frames_dir, [], decisions_path, name)
                    else:
                        self._json({"error": "unknown kind"}, 400)
                        return
                except FileNotFoundError as exc:
                    self._json({"error": f"not found: {exc}"}, 404)
                    return
                self._json(
                    {
                        "ok": True,
                        "session_tag": state.session_tag,
                        "frame_count": len(state.candidates),
                    }
                )
                return
            if self.path == "/api/open-dataset":
                payload = self._payload()
                self._json(state.open_dataset(str(payload.get("name", ""))))
                return
            if self.path == "/api/realign-pool-tile":
                payload = self._payload()
                self._json(
                    state.realign_pool_tile(
                        str(payload.get("name", "")),
                        float(payload.get("left", paths.MARGIN_X)),
                        float(payload.get("top", paths.MARGIN_Y)),
                    )
                )
                return
            if self.path == "/api/recut-pool":
                self._json(state.recut_pool())
                return
            if self.path == "/api/close-source":
                payload = self._payload()
                self._json(
                    state.close_pool_source(
                        payload.get("tag", ""), discard=bool(payload.get("discard"))
                    )
                )
                return
            if self.path == "/api/open-source":
                payload = self._payload()
                self._json(state.open_pool_source(payload.get("tag", "")))
                return
            if self.path == "/api/promote-training":
                payload = self._payload()
                self._json(
                    state.promote_to_training(
                        payload.get("name") or "plates", payload.get("mode") or "new"
                    )
                )
                return
            if self.path == "/api/find-run":
                # find is drawing only - the boxes are the ones he drew, nothing is mined
                payload = self._payload()
                self._json(
                    state.save_drawn(
                        payload.get("boxes", []), negatives=payload.get("negatives") or []
                    )
                )
                return
            if self.path == "/api/housekeeping-archive":
                payload = self._payload()
                try:
                    self._json(housekeeping.archive_paths(list(payload.get("paths", []))))
                except housekeeping.HousekeepingError as exc:
                    self._json({"error": str(exc)}, 400)
                return
            if self.path == "/api/housekeeping-delete":
                payload = self._payload()
                try:
                    self._json(housekeeping.delete_paths(list(payload.get("paths", []))))
                except housekeeping.HousekeepingError as exc:
                    self._json({"error": str(exc)}, 400)
                return
            if self._post_pool_decisions():
                return
            self.send_response(404)
            self.end_headers()

    return Handler
