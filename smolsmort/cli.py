"""`uv run smolsmort` - starts the hyperparams review UI on localhost and blocks until Ctrl-C.

no train_state_for hook is passed here: this entrypoint has no real examples to bind a TrainState
to, so /api/train-start only resolves and reports hyperparams, same as server.py's own docstring
describes for the no-hook case. a future train tab wires the hook where it has real examples.
"""

from __future__ import annotations

import argparse

from smolsmort.review_ui.server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smolsmort", description="the hyperparams review UI")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
    args = parser.parse_args(argv)

    server = serve(port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"smolsmort review UI on {url} - Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0
