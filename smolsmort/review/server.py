"""the entry point: build the state, wire the routes, serve on stdlib http.server.

no web framework and no new dependency. `serve` returns the (unstarted) server so a caller - the
command line, or a test on an ephemeral port - owns its lifecycle: `serve_forever` on a thread,
`shutdown` to stop.
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from smolsmort.review import paths
from smolsmort.review.playback_state import PlaybackState
from smolsmort.review.routes import App, Tab, make_handler
from smolsmort.review.state import ReviewState
from smolsmort.review.train import TrainState
from smolsmort.review.train_api import TrainApi


def build_app(
    *,
    backend: str = "heatmap",
    renderer=None,
    guesser=None,
    playback: PlaybackState | None = None,
    tabs: list[Tab] | None = None,
    ui_dir: Path | None = None,
    ui_fallbacks: tuple[Path, ...] = (),
    pool: Path | None = None,
    bases_file: Path | None = None,
    crop_file: Path | None = None,
) -> App:
    """the whole app around the four seams: `renderer` and `guesser` here, the model backend by
    name (see smolsmort.backends), and the class scheme through `App.load_scheme`. the example
    source is the find tab itself (human-drawn boxes) and the trained backend's sweep."""
    state = ReviewState(
        paths.SESSIONS_DIR,
        [],
        (pool or paths.TILES_DIR) / "_unbound.decisions.json",
        "",
        pool,
        renderer=renderer,
        guesser=guesser,
        bases_file=bases_file,
        crop_file=crop_file,
    )
    state.load_saved_bases()
    trainer = TrainApi(state, TrainState(backend))
    return App(
        state=state,
        trainer=trainer,
        playback=playback,
        tabs=tabs or [],
        ui_dir=ui_dir if ui_dir is not None else paths.REVIEW_UI_DIR,
        ui_fallbacks=tuple(ui_fallbacks),
    )


def serve(app: App, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """a server bound to `port` (0 picks a free one), not yet serving"""
    return ThreadingHTTPServer((host, port), make_handler(app))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the review web tool: find, select, train")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--backend", default="heatmap", help="model backend name")
    args = parser.parse_args(argv)
    # a settings file beside the tool remembers repointed bases and the crop rule across restarts
    settings = paths.REVIEW_UI_DIR.parent
    app = build_app(
        backend=args.backend,
        bases_file=settings / "review_bases.json",
        crop_file=settings / "review_crop.json",
    )
    server = serve(app, port=args.port)
    print(f"review tool on http://127.0.0.1:{server.server_address[1]} - Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
