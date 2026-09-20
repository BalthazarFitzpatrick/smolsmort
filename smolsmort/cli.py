"""`uv run smolsmort` - starts the review web tool (find, select, train) and blocks until Ctrl-C.

this is the real review server from smolsmort.review.server, serving the page in review_ui. the
hyperparams menu the standalone page used to be is the train tab's config menu.
"""

from __future__ import annotations

import argparse

from smolsmort.review import paths
from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC
from smolsmort.review_ui.tab import hyperparams_tab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smolsmort", description="the review web tool")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
    parser.add_argument("--backend", default="heatmap", help="model backend name")
    args = parser.parse_args(argv)

    # settings beside the training data remember repointed bases and the crop rule
    settings = paths.LABELS_DIR.parent
    app = build_app(
        backend=args.backend,
        ui_dir=STATIC,
        tabs=[hyperparams_tab()],
        bases_file=settings / "review_bases.json",
        crop_file=settings / "review_crop.json",
    )
    server = serve(app, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"smolsmort review on {url} - Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
