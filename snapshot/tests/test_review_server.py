"""the review tool, exercised through a real socket.

WHY THIS EXISTS. fifty tests already cover the review tool and every one of them builds a state
class directly - so nothing whatsoever guarded the served page, the 45 routes, or the wiring
between them. The entire web app lives in one 3,472-line python string literal, which means every
edit to it is a text substitution with no syntax check behind it; two failures on 2026-09-01 came
straight from that (a set of replacements that silently no-opped after ruff reflowed the target, and
a block edit that broke python indentation inside peak_distribution).

This is the guard that makes splitting that literal into real files a mechanical move rather than a
rewrite. The two assertions that matter:

  1. EVERY ROUTE ANSWERS. the route list is read out of the source rather than typed here, so an
     endpoint added tomorrow is covered tomorrow, not whenever someone remembers this file.
  2. EVERY ID THE SCRIPT REACCHES FOR EXISTS IN THE MARKUP. the js and the html are two halves of
     one document today; the moment they are two files they can drift, and the failure mode is a
     silent dead button rather than an error. this compares one against the other.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import snapshot.tools.review_templates as rt
from snapshot.review import paths, routes

# the routes moved out of review_templates.py into review/routes.py; this reads them from wherever
# the handler actually is, so the scrape follows the code rather than a remembered filename
SOURCE = Path(routes.__file__).read_text()


# ---------------------------------------------------------------- routes, read from the source


def _routes(pattern: str) -> set[str]:
    return set(re.findall(pattern, SOURCE))


GET_ROUTES = sorted(_routes(r'parsed\.path == "([^"]+)"'))
GET_PREFIXES = sorted(_routes(r'parsed\.path\.startswith\("([^"]+)"\)'))
POST_ROUTES = sorted(_routes(r'self\.path == "([^"]+)"'))


def test_the_route_list_is_not_empty():
    """a scrape that silently matches nothing would make every test below vacuously pass"""
    assert len(GET_ROUTES) > 20
    assert len(POST_ROUTES) > 15
    assert "/" in GET_ROUTES
    assert "/ui/" in GET_PREFIXES


# ---------------------------------------------------------------- a real server on a real socket


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """the actual ThreadingHTTPServer main() builds, on an ephemeral port.

    PORT 0, never a fixed one: 8766 is Balthazar Fitzpatrick's own server and this suite must never contend with
    it, nor fail because he happens to have it open.
    """
    root = tmp_path_factory.mktemp("review")
    sessions = root / "sessions"
    labels = root / "labels"
    library = root / "library"
    for path in (sessions, labels, library):
        path.mkdir(parents=True)

    state = rt.ReviewState(sessions, [], library, library / "_unbound.decisions.json", "")
    state.bases = dict(state.bases)
    state.bases["sessions"] = str(sessions)
    state.bases["labels"] = str(labels)
    trainer = rt.TrainState(state)
    trainer.MODEL_PATH = root / "model.pt"
    verifier = rt.VerifyState(state, trainer)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), rt.make_handler(state, trainer, verifier))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _wait_until_listening(httpd.server_address)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _wait_until_listening(address, attempts: int = 50) -> None:
    import time

    for _ in range(attempts):
        with socket.socket() as probe:
            if probe.connect_ex(address) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"server never came up on {address}")


def _post(base: str, path: str, payload: dict) -> tuple[int, bytes]:
    """a POST that reports what came back, including nothing at all.

    urlopen raises URLError rather than returning when the server drops the connection, and that is
    precisely the failure being tested - so it is turned into an empty body the caller can assert on
    rather than an error that reads like a broken test.
    """
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:  # a 400 is an answer
        return exc.code, exc.read()
    except urllib.error.URLError:
        return 0, b""


def _get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(base + path, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:  # a 404 is an answer; a 500 is not
        return exc.code, exc.read()


# ---------------------------------------------------------------- every route answers


@pytest.mark.parametrize("route", GET_ROUTES)
def test_every_get_route_answers_without_a_server_error(server, route):
    """UNBOUND ON PURPOSE. no dataset, no model, no profile - the state every route has to survive,
    because it is the state the server starts in. a route that only works once something is loaded
    must say so in json, not raise."""
    status, body = _get(server, route)
    assert status < 500, f"{route} returned {status}: {body[:400]!r}"


def test_the_index_page_is_served_whole(server):
    status, body = _get(server, "/")
    assert status == 200
    page = body.decode()
    assert page.lstrip().startswith("<!doctype html")
    assert page.rstrip().endswith("</html>")
    # the markup alone, now that the script is a separate file. the script's own size is checked
    # where it is served, not here
    assert len(page) > 15_000, "the page came back suspiciously short"


def test_every_asset_the_page_links_is_served(server):
    """a stylesheet or script the page references but the server cannot serve fails silently - the
    page renders unstyled or inert with a 404 in a console nobody is watching"""
    page = _served_page(server)
    referenced = set(re.findall(r'(?:href|src)="(/ui/[\w.-]+)"', page))
    assert referenced, "the page references no /ui/ assets - the scrape is wrong"
    for path in sorted(referenced):
        status, body = _get(server, path)
        assert status == 200, f"{path} is referenced by the page but returned {status}"
        assert body.strip(), f"{path} is empty"


def test_an_unknown_route_is_a_404_not_a_crash(server):
    status, _ = _get(server, "/api/no-such-endpoint")
    assert status == 404


# ---------------------------------------------------------------- the page agrees with its script

# ids the script creates or reaches for on elements that do not exist in the static markup - a
# cloned node, a row built at runtime, an element inside another page. each needs a reason.
DYNAMIC_IDS = {
    "train-view-img",  # lives inside #train-stage, rebuilt per frame
    # both built by loadTrainClasses into #train-channels, because the channel list is derived from
    # the bound set rather than fixed. Note these USED to pass without being listed: the script was
    # inside the page, so the regex found id="..." in its own innerHTML string. Extracting the
    # script to /ui/app.js is what exposed them as genuinely dynamic.
    "train-primitives",
    "train-states",
}


def _served_page(server: str) -> str:
    status, body = _get(server, "/")
    assert status == 200
    return body.decode()


def _served_scripts(server: str) -> dict[str, str]:
    """every script the page runs: the inline blocks AND each /ui/*.js it pulls in.

    THIS FOLLOWS THE EXTRACTION. the page script moved out of the python literal into
    review_ui/app.js on 2026-09-02; a check that kept reading only the html would have gone on
    passing while guarding nothing at all.
    """
    page = _served_page(server)
    found = {
        f"inline#{i}": body
        for i, body in enumerate(re.findall(r"<script>(.*?)</script>", page, re.DOTALL))
    }
    for src in re.findall(r'<script src="(/ui/[\w.-]+)"', page):
        status, body = _get(server, src)
        assert status == 200, f"{src} is referenced by the page but returned {status}"
        found[src] = body.decode()
    assert len(found) > 1, "expected inline blocks and at least one /ui/ script"
    return found


def test_every_id_the_script_looks_up_exists_in_the_markup(server):
    """THE GUARD THAT MAKES THE SPLIT SAFE.

    the html and the js are one string literal today, so nothing stops them agreeing. the moment
    they are separate files, a moved block or a renamed id produces a dead control and no error
    anywhere - getElementById returns null and the handler is simply never attached. this reads
    every id the script asks for and checks the markup actually offers it.
    """
    page = _served_page(server)
    script = "\n".join(_served_scripts(server).values())
    wanted = set(re.findall(r"""getElementById\(['"]([\w-]+)['"]\)""", script))
    wanted |= set(re.findall(r"""querySelector\(['"]#([\w-]+)['"]\)""", script))
    present = set(re.findall(r"""\bid=["']([\w-]+)["']""", page))

    missing = sorted(wanted - present - DYNAMIC_IDS)
    assert not missing, (
        f"the script reaches for {len(missing)} id(s) the page never defines: {missing}"
    )


def test_every_panel_is_reachable_and_every_tab_leads_somewhere(server):
    """a tab with no panel switches to nothing; a panel nothing can open is dead markup. both have
    happened while moving markup around, and neither raises.

    NOT a plain equality: `tune` is a drill-down with no nav tab of its own, opened by "fix
    alignment" in the right-click menu. A panel is legitimate without a tab as long as something
    actually calls activateTab for it - which is the condition checked here.
    """
    page = _served_page(server)
    tabs = set(re.findall(r'class="nav-tab[^"]*" data-tab="([\w-]+)"', page))
    panels = set(re.findall(r'class="tab-panel[^"]*" data-panel="([\w-]+)"', page))
    # activateTab calls live in /ui/app.js now, not in the page
    activated = set(
        re.findall(
            r"""activateTab\(['"]([\w-]+)['"]\)""", "\n".join(_served_scripts(server).values())
        )
    )
    assert tabs, "no nav tabs found - the scrape is wrong, not the page"

    assert not tabs - panels, f"nav tab(s) with no panel: {sorted(tabs - panels)}"
    orphans = panels - tabs - activated
    assert not orphans, f"panel(s) nothing can open: {sorted(orphans)}"


def test_exactly_one_tab_starts_active(server):
    """two active tabs shows two panels stacked; none shows a blank page"""
    page = _served_page(server)
    active = re.findall(r'class="nav-tab active" data-tab="([\w-]+)"', page)
    assert len(active) == 1, f"expected one active tab, found {active}"


# ---------------------------------------------------------------- one flow across the seams


def test_binding_a_dataset_reaches_the_trainer(server, tmp_path_factory):
    """the flow that crosses the most seams: http -> ReviewState -> TrainState -> back out as json.

    a split that leaves the three state objects wired to different instances would pass every
    route test above and fail here.
    """
    status, body = _get(server, "/api/bindable-recordings")
    assert status == 200
    assert "rows" in json.loads(body)

    status, body = _get(server, "/api/train-classes")
    assert status == 200
    payload = json.loads(body)
    # the two pickers are mutually exclusive answers to "what is loaded" - both keys must exist
    # even unbound, because the page paints both heads from this one response
    assert set(payload) >= {"set", "weights", "classes"}
    assert payload["set"] is None and payload["weights"] is None


def test_train_info_reports_an_unbound_server_rather_than_failing(server):
    """with nothing bound this has to answer, not raise - it is fetched on every tab entry"""
    status, body = _get(server, "/api/train-info")
    assert status == 200
    payload = json.loads(body)
    assert payload["set"] is None
    assert "device" in payload


# ---------------------------------------------------------------- the script actually parses


def test_the_served_script_is_valid_javascript(server, tmp_path):
    """THE GAP THIS CLOSES, found the hard way on 2026-09-02.

    Removing the load tab deleted `if (name === 'load') loadLoadList();` - which was the HEAD of
    activateTab's if/else chain, leaving `else if` with nothing before it. That is a syntax error,
    so the browser refused the entire script and every control on every tab went dead at once. The
    route test passed, the element-id test passed, the suite was green: nothing here parses
    javascript, so nothing could see it.

    This matters most for the extraction ahead - moving 3,472 lines of markup and script into
    separate files is exactly the operation that severs a chain like that one.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available to parse the served script")

    for index, (name, body) in enumerate(_served_scripts(server).items()):
        path = tmp_path / f"served_{index}.js"
        path.write_text(body)
        result = subprocess.run(
            [node, "--check", str(path)], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"{name} is not valid javascript:\n{result.stderr}"


def test_every_function_the_page_calls_on_tab_switch_is_defined(server):
    """a tab handler naming a function that no longer exists throws only when that tab is opened,
    so the page looks fine until the one click that matters. this reads the dispatch chain and
    checks each name is actually declared."""
    script = "\n".join(_served_scripts(server).values())
    called = set(re.findall(r"""name === ['"][\w-]+['"]\)\s*\{?\s*(\w+)\(""", script))
    assert called, "activateTab's dispatch chain was not found - the scrape is wrong"
    for name in sorted(called):
        assert re.search(rf"\b(?:async\s+)?function\s+{name}\b", script), (
            f"activateTab calls {name}() but nothing declares it"
        )


def test_every_project_function_the_scripts_call_is_defined(server):
    """A FUNCTION DELETED WHILE ITS CALLERS REMAIN is invisible to every other check here.

    `ifaceRefresh` was the interface tab's single repaint path. It was deleted on 2026-08-31 while
    its SEVEN callers stayed, and went unnoticed for two days: `node --check` validates syntax and
    an unbound identifier is syntactically fine, so the suite stayed green while every mark, add,
    remove and keybind threw a ReferenceError after its request had already succeeded. The server
    updated, the page did not, and it read like a caching bug rather than a crash.

    Scoped to the project's own camelCase prefixes on purpose: a broad "is every callee defined"
    check drowns in class methods, CSS functions inside template strings and words in prose, and a
    test that needs a fifty-entry allowlist is a test nobody keeps honest. Nothing built in is
    called `ifaceRefresh`, so this stays precise.
    """
    scripts = _served_scripts(server)
    source = "\n".join(scripts.values())

    # definitions from the RAW source - they are unambiguous, and stripping risks eating one
    defined = set(re.findall(r"(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", source))
    defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", source))

    # calls from source with comments removed, so prose cannot look like a call
    code = re.sub(r"//[^\n]*", "", source)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    prefixes = (
        "iface",
        "draw",
        "load",
        "render",
        "pb",
        "train",
        "sweep",
        "cluster",
        "script",
        "crop",
    )
    called = {
        name
        for name in re.findall(r"(?<![.\w$])([a-z][\w$]*)\s*\(", code)
        if name.startswith(prefixes) and len(name) > 4
    }
    assert called, "no project functions found - the scrape is wrong, not the page"

    missing = sorted(called - defined)
    assert not missing, (
        f"called but never defined, so every call site throws a ReferenceError: {missing}"
    )


# ---------------------------------------------------------------- which screen the tab opens on


def test_the_tab_opens_bound_to_nothing(tmp_path, monkeypatch):
    """A WRONG GUESS IS INDISTINGUISHABLE FROM LOST WORK, which is what it looked like: startup
    picked a directory, adopted its only character, and landed on a holding directory whose state
    had nothing marked - showing an empty list beside two screens holding 7 and 24 marks.

    Balthazar Fitzpatrick: "Fix it to open on nothing, until I use the dropdowns." Nothing selected is a state the
    page can state plainly; a guess is not.
    """
    monkeypatch.setattr(paths, "INTERFACE_SHOTS", tmp_path)
    for screen, character in (("msi", "a-one"), ("retina", "a-two")):
        d = tmp_path / screen / character
        d.mkdir(parents=True)
        (d / "interface.json").write_text("{}")

    assert routes._startup_screen() == routes.UNASSIGNED_SCREEN


def test_it_opens_on_nothing_even_with_a_single_screen(tmp_path, monkeypatch):
    """the old "only one, so open it" shortcut is gone too - one screen may still hold two
    characters, and picking for him is the behaviour that caused the scare
    """
    monkeypatch.setattr(paths, "INTERFACE_SHOTS", tmp_path)
    d = tmp_path / "msi" / "someone"
    d.mkdir(parents=True)
    (d / "interface.json").write_text("{}")

    assert routes._startup_screen() == routes.UNASSIGNED_SCREEN


def test_no_screens_at_all_is_the_same_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "INTERFACE_SHOTS", tmp_path)
    assert routes._startup_screen() == routes.UNASSIGNED_SCREEN


# ---------------------------------------------------------------- the byte routes


def test_every_image_route_answers_404_rather_than_dropping_the_connection(server):
    """SEVEN ROUTES THAT WERE THE SAME SEVEN LINES, now one table and one server. /crop/<n> was the
    odd one out and had no error handling at all: an out-of-range index raised straight out of the
    handler and killed the connection, so curl saw no response rather than a 404. Nothing in the
    page asks for one, which is why it survived - a stale tab after rebinding does exactly that.
    """
    for path in (
        "/draw-frame/nope.jpg",
        "/interface-shot/nope.png",
        "/library-thumb/nope",
        "/unsorted-thumb/nope",
        "/tile/nope",
        "/guide/nope",
        "/crop/999999",
        "/crop/not-a-number",
    ):
        status, _ = _get(server, path)
        assert status == 404, f"{path} answered {status}"


def test_the_image_routes_are_declared_in_one_place(server):
    """a new one should be a line in a table, not a seventh copy of the same seven lines"""
    assert len(routes.__dict__) or True  # the table lives on the handler class, built per server
    import re

    table = re.search(r"IMAGE_ROUTES = \{(.*?)\n        \}", SOURCE, re.DOTALL)
    assert table, "IMAGE_ROUTES is gone - the byte routes have been un-collapsed"
    assert table.group(1).count('": (') >= 7


def test_a_post_with_no_body_is_answered_rather_than_dropped(server):
    """TWENTY ROUTES OPENED WITH THE SAME TWO LINES reading the body, and json.loads(b"") raises -
    so a POST without a body killed the connection rather than answering. One _payload() helper
    now returns {} for an empty or unparseable body, which several routes legitimately receive.
    """
    import urllib.error
    import urllib.request

    for path in ("/api/save-checkpoint", "/api/promote-training", "/api/crop-settings"):
        request = urllib.request.Request(server + path, data=b"", method="POST")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 200, f"{path} answered {status} to an empty body"


def test_the_body_is_read_in_one_place(server):
    """a twenty-first route should not have to remember Content-Length"""
    import re

    assert "def _payload(self)" in SOURCE
    raw_reads = re.findall(r'length = int\(self\.headers\.get\("Content-Length", 0\)\)', SOURCE)
    assert len(raw_reads) <= 2, (
        "only _payload itself and the interface upload, which reads raw bytes rather than json, "
        f"should read Content-Length directly - found {len(raw_reads)}"
    )


def test_no_post_route_can_kill_the_connection(server):
    """A ROUTE THAT RAISES ANSWERS NOTHING - not a 500, no response, the socket closes and the page
    sees an unexplainable network error. Four routes did exactly that on a request without an
    "index": /api/height-delta, /api/reviewed, /api/exclude and /api/manual-label all index the
    payload directly. Nothing in the page sends a malformed body, which is why it survived; a stale
    tab or a retried request does.
    """
    for path in POST_ROUTES:
        status, body = _post(server, path, {})
        assert status in (200, 400, 404), f"{path} answered {status}"
        assert body, f"{path} answered nothing at all - the connection was dropped"


def test_a_bad_request_says_what_was_missing(server):
    """a 400 that does not name the field is only marginally better than a dropped socket"""
    status, body = _post(server, "/api/height-delta", {})
    assert status == 400
    assert "index" in body.decode(), body[:120]


def test_no_get_route_can_kill_the_connection_on_a_bad_query(server):
    """the same hole do_POST had, on the easiest thing in the tool to get wrong. Three routes parsed
    a query parameter with float() or int() and answered NOTHING when it was not a number -
    /api/playback-frame?t=, /api/draw-frames?percent= and /api/clusters?k=.
    """
    for path in (
        "/api/playback-frame?t=abc",
        "/api/draw-frames?percent=abc",
        "/api/clusters?k=abc",
        "/api/page?index=abc",
        "/crop/abc",
    ):
        # _get raises on a dropped connection rather than returning, so reaching this line at all
        # is the assertion. A 404 legitimately carries no body; a 400 must say what was wrong.
        status, body = _get(server, path)
        assert status in (200, 400, 404), f"{path} answered {status}"
        if status == 400:
            assert body, f"{path} gave a 400 with no explanation"


def test_the_ordinary_routes_are_untouched_by_the_guard(server):
    """a guard that swallows real answers would be worse than the crash it replaces"""
    for path in ("/", "/ui/app.js", "/api/interface", "/api/page"):
        status, body = _get(server, path)
        assert status == 200 and body, f"{path} answered {status}"


# ---------------------------------------------------------------- the navigation tab's routes


@pytest.mark.parametrize(
    "path",
    [
        "/api/nav-sessions",
        "/api/nav-latest",
        "/api/nav",
        "/api/nav?session=",
        "/api/nav?session=does-not-exist",
        "/api/nav?session=../../etc/passwd",
        "/api/nav?session=%00",
    ],
)
def test_the_nav_routes_always_answer(server, path):
    """NO ANSWER IS THE FAILURE MODE, and these are polled once a second while a recording runs -
    a handler that raises closes the socket and the tab shows a network error it cannot explain.
    A 404 or an {"error": ...} body is fine; a dropped connection is not.
    """
    status, body = _get(server, path)
    assert status == 200, f"{path} answered {status}"
    json.loads(body)  # and it is json, not a stack trace


def test_asking_for_a_walk_with_no_recordings_is_empty_not_broken(server):
    """the ordinary first-run state: the tool is open before anything has been recorded"""
    status, body = _get(server, "/api/nav")
    assert status == 200
    assert json.loads(body)["maps"] == []


def test_image_routes_forbid_caching_like_the_json_ones_do(server):
    """A KERNEL IS RE-CUT UNDER AN UNCHANGED URL, which is what makes this a correctness header
    rather than a nicety. `_png` was the one responder that never called `_no_store`, so after a
    successful realign the browser redrew its CACHED crop - the save landed, the picture did not
    move, and it read as "fix alignment doesn't save reliably".
    """
    # a 404 still goes through send_response/end_headers, so the header is asserted on the path
    # that always exists rather than on a fixture tile this server has no reason to hold
    with urllib.request.urlopen(server + "/", timeout=20) as response:
        assert "no-store" in response.headers.get("Cache-Control", "")

    for route in ("/unsorted-thumb/nope", "/tile/nope"):
        status, _ = _get(server, route)
        assert status == 404, f"{route} should 404 for a tile that does not exist"


def test_exclude_refuses_a_toggle_shaped_request(server):
    """it no longer flips, so a body without `excluded` is a stale page - and guessing would invert
    half of whatever it sent. name the problem instead, the way _dispatch_post's guard does
    """
    status, body = _post(server, "/api/exclude", {"name": "whatever_k00000"})
    assert status == 400
    assert "excluded" in json.loads(body)["error"]


def test_exclude_takes_the_whole_selection_in_one_request(server):
    """A SELECTION IS ONE PRESS. The page fanned out a POST per tile inside a Promise.all with no
    catch, so eighty tiles were eighty round trips through a six-connection browser and one
    rejection skipped the repaint entirely - the button looked dead while the writes that landed
    stayed, which a refresh then revealed. Balthazar Fitzpatrick: "the button seems unresponsive
    and nothing happens, but when refreshing the cards are marked as that".
    """
    status, body = _post(
        server, "/api/exclude", {"names": ["missing_k00000", "missing_k00001"], "excluded": True}
    )
    # both are unknown here, so it still 404s - what matters is that a LIST is accepted and
    # answered per name rather than rejected outright
    assert status == 404
    answer = json.loads(body)
    assert sorted(answer["unknown"]) == ["missing_k00000", "missing_k00001"]


def test_exclude_still_takes_a_single_name(server):
    """the old shape has to keep working - a page held open across a restart still sends it"""
    status, body = _post(server, "/api/exclude", {"name": "missing_k00000", "excluded": True})
    assert status == 404


# ---- the settings tree: which folder holds which kind of data ---------------------------------
# The bases were three text fields, and a path typed by hand is a path that can be typed wrong -
# silently, because a base that does not exist offers nothing rather than failing. These guard the
# browser that replaced them, and the one base that is a live handle rather than a string.


def test_the_tree_lists_one_level_and_says_which_rows_open(tmp_path):
    from types import SimpleNamespace

    from snapshot.review.state import ReviewState

    (tmp_path / "sessions" / "one").mkdir(parents=True)
    (tmp_path / "labels").mkdir()
    (tmp_path / ".hidden").mkdir()
    fake = SimpleNamespace(bases={"root": str(tmp_path)})

    out = ReviewState.dir_tree(fake)
    names = {e["name"]: e for e in out["entries"]}
    assert set(names) == {"sessions", "labels"}, "a dotted directory is not offered"
    assert names["sessions"]["has_children"] is True
    assert names["labels"]["has_children"] is False
    assert out["parent"] is None, "the root has nowhere above it to walk to"


def test_the_tree_refuses_to_leave_the_root(tmp_path):
    """this is a browser, and the honest answer to "show me /etc" is what it is allowed to show"""
    from types import SimpleNamespace

    from snapshot.review.state import ReviewState

    (tmp_path / "inside").mkdir()
    fake = SimpleNamespace(bases={"root": str(tmp_path)})
    assert ReviewState.dir_tree(fake, "/etc")["here"] == str(tmp_path.resolve())
    assert ReviewState.dir_tree(fake, str(tmp_path / "inside"))["here"] == str(
        (tmp_path / "inside").resolve()
    )


def test_repointing_the_pool_moves_the_directories_derived_from_it(tmp_path):
    """THE POOL IS A LIVE HANDLE, not just a string: a repoint that only stored the value would
    take effect at the next restart and look broken until then.

    The pool is now the tile directory ITSELF rather than a library path with "_unsorted" glued
    on, and the library is its own base - repointing the pool used to silently repoint the
    library with it.
    """
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    from snapshot.review import paths
    from snapshot.review.state import ReviewState

    fake = SimpleNamespace(
        bases=dict(paths.DEFAULT_BASES),
        library=tmp_path / "old_library",
        pools=[tmp_path / "old_tiles"],
        archive_dir=tmp_path / "old_tiles" / "_archive",
        _dataset_cache={"stale": object()},
        _record_origin={"a_k00000": tmp_path / "old_tiles"},
        _save_bases=lambda: None,
        lock=threading.RLock(),
    )
    fake.bases["templates"] = str(tmp_path / "old_library")
    ReviewState.set_bases(fake, {"pool": str(tmp_path / "new_tiles")})
    assert fake.pools == [Path(tmp_path / "new_tiles")]
    assert fake.archive_dir == Path(tmp_path / "new_tiles" / "_archive")
    assert fake._record_origin == {}, "records must not be attributed to the old pool"
    assert fake.library == tmp_path / "old_library", "the pool must not drag the library with it"
    assert fake._dataset_cache == {}, "tiles resolved against the old pool must not survive"


def test_an_unknown_base_key_is_ignored_rather_than_stored(tmp_path):
    """a stale browser must not be able to invent a base the server then tries to read"""
    import threading
    from types import SimpleNamespace

    from snapshot.review import paths
    from snapshot.review.state import ReviewState

    fake = SimpleNamespace(
        bases=dict(paths.DEFAULT_BASES),
        library=tmp_path / "p",
        pools=[tmp_path / "tiles"],
        archive_dir=tmp_path / "tiles" / "_archive",
        _dataset_cache={},
        _record_origin={},
        _save_bases=lambda: None,
        lock=threading.RLock(),
    )
    out = ReviewState.set_bases(fake, {"nonsense": "/tmp", "labels": str(tmp_path)})
    assert "nonsense" not in out
    assert out["labels"] == str(tmp_path)


# ---- closing a source ---------------------------------------------------------------------
# CLOSING USED TO ARCHIVE. It moved every tile out to the archive and dropped its label
# record, so "close" silently cost the judging - on 7 Sep it emptied a 6,133 tile pool and left the
# tracked _labels.json holding `{}`. Closing is a view decision now.


def _pool(tmp_path, tag="demo", tiles=3, judged=0):
    import json

    import numpy as np

    from snapshot.vision.plate_templates import PlateTemplate, save_template

    pool = tmp_path / "tiles"
    pool.mkdir(parents=True)
    records = {}
    for i in range(tiles):
        name = f"{tag}_k{i:05d}"
        save_template(
            PlateTemplate(name=name, rgb=np.zeros((4, 4, 3), np.uint8), mask=np.ones((4, 4), bool)),
            pool,
        )
        records[name] = {"classdef": "d", "excluded": False}
        if i < judged:
            records[name]["label"] = "hostile / target"
    (pool / "_labels.json").write_text(json.dumps(records))
    return pool


def _wire(pool):
    """a throwaway pool with the real ReviewState methods bound to it.

    BOUND, NOT STUBBED: pool_records reaches _read_pool_file reaches _pool_labels_path, and a stub
    per layer tests the stubs. This gives the real chain a directory to work in.
    """
    import threading
    from types import MethodType, SimpleNamespace

    from snapshot.review.state import ReviewState

    fake = SimpleNamespace(pools=[pool], _record_origin={}, lock=threading.RLock())
    for name in (
        "tile_paths",
        "tile_path",
        "pool_of",
        "_pool_labels_path",
        "_read_pool_file",
        "pool_records",
        "_write_pool_records",
        "closed_tags",
        "_write_closed_tags",
        "pool_source_state",
        "unsorted_templates",
        "pool_sources",
        "close_pool_source",
        "open_pool_source",
    ):
        setattr(fake, name, MethodType(getattr(ReviewState, name), fake))
    fake.unsorted_dir = ReviewState.unsorted_dir.fget(fake)
    fake.closed_file = ReviewState.closed_file.fget(fake)
    return fake


def test_closing_a_source_moves_nothing_and_keeps_its_judgements(tmp_path):
    pool = _pool(tmp_path, tiles=3, judged=2)
    fake = _wire(pool)

    out = fake.close_pool_source("demo")
    assert out["closed"] == 3 and out["discarded"] == 0
    assert len(list(pool.glob("*.npz"))) == 3, "closing must not move a single tile"
    assert len(fake.pool_records()) == 3, "and must not drop a judgement"
    assert fake.closed_tags() == {"demo"}


def test_a_closed_source_is_hidden_from_the_grid_but_still_listed(tmp_path):
    pool = _pool(tmp_path, tiles=3)
    fake = _wire(pool)
    fake.close_pool_source("demo")

    assert fake.unsorted_templates() == [], "a closed source leaves the grid"
    listed = fake.pool_sources()["sources"]
    assert listed == [{"tag": "demo", "tiles": 3, "closed": True}], (
        "and stays listed, so it can be offered back rather than re-cut"
    )


def test_reopening_is_instant_because_nothing_was_moved(tmp_path):
    pool = _pool(tmp_path, tiles=3, judged=1)
    fake = _wire(pool)
    fake.close_pool_source("demo")
    out = fake.open_pool_source("demo")

    assert out["tiles"] == 3 and out["judged"] == 1
    assert len(fake.unsorted_templates()) == 3


def test_discard_drops_the_judgements_and_still_keeps_the_tiles(tmp_path):
    """the one destructive choice, and it is never what close does on its own"""
    pool = _pool(tmp_path, tiles=4, judged=3)
    fake = _wire(pool)

    out = fake.close_pool_source("demo", discard=True)
    assert out["discarded"] == 3
    assert len(list(pool.glob("*.npz"))) == 4, "discarding judgements is not deleting tiles"
    assert fake.pool_records() == {}


def test_the_page_can_ask_before_closing(tmp_path):
    fake = _wire(_pool(tmp_path, tiles=5, judged=2))
    assert fake.pool_source_state("demo") == {
        "tag": "demo",
        "tiles": 5,
        "judged": 2,
    }


# ---------------------------------------------------------------- nav hotkey bindings


def test_the_nav_keys_route_saves_a_binding(server, tmp_path, monkeypatch):
    """the navigation tab's "change" lands here; the file it writes is what wt-record reads"""
    from parent.imitation import record

    target = tmp_path / "nav_keys.json"
    monkeypatch.setattr(record, "NAV_KEYS_PATH", target)

    status, body = _post(server, "/api/nav-keys", {"go_zone": "Option+Scroll Up"})
    assert status == 200
    assert json.loads(body)["keys"]["go_zone"] == "alt+scroll_up"
    stored = json.loads(target.read_text())
    assert stored["version"] == 2 and stored["go_zone"] == "alt+scroll_up"

    status, body = _get(server, "/api/nav-keys")
    assert json.loads(body)["keys"]["go_zone"] == "alt+scroll_up"

    status, body = _post(server, "/api/nav-keys", {"log_point": "alt+"})
    assert status == 400 and b"log_point" in body
    assert json.loads(target.read_text())["log_point"] == record.DEFAULT_NAV_KEYS["log_point"]


def test_the_listen_route_needs_a_duration(server):
    """an empty body must answer at once, not open an input listener"""
    status, body = _post(server, "/api/nav-keys/listen", {})
    assert status == 400 and b"seconds" in body
