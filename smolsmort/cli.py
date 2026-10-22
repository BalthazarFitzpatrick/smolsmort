"""`uv run smolsmort` - starts the review web tool (find, select, train) and blocks until Ctrl-C.

this is the real review server from smolsmort.review.server, serving the page in review_ui. the
hyperparams menu the standalone page used to be is the train tab's config menu.
"""

from __future__ import annotations

import argparse

from smolsmort.forecast.tab import forecast_tab
from smolsmort.review import paths
from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC
from smolsmort.review_ui.tab import hyperparams_tab


def _parse_threads(value: str) -> list[int]:
    try:
        threads = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threads must be comma-separated positive integers"
        ) from exc
    if any(count < 1 for count in threads):
        raise argparse.ArgumentTypeError("threads must be comma-separated positive integers")
    return threads


def _parse_positive_int(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smolsmort", description="the review web tool")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
    parser.add_argument("--backend", default="heatmap", help="model backend name")
    commands = parser.add_subparsers(dest="command")
    forecast = commands.add_parser("forecast", help="forecast tools")
    forecast_commands = forecast.add_subparsers(dest="forecast_command", required=True)
    calibrate = forecast_commands.add_parser("calibrate", help="measure machine parallelism")
    calibrate.add_argument(
        "--threads", type=_parse_threads, default=[1, 2, 3, 5, 10], help="thread counts to measure"
    )
    calibrate.add_argument(
        "--max-families",
        type=_parse_positive_int,
        default=3,
        help="maximum useful parallel families (default 3: xgboost, lightgbm, ridge)",
    )
    calibrate.add_argument("--write", action="store_true", help="save the machine forecast config")
    args = parser.parse_args(argv)

    if args.command == "forecast":
        from smolsmort.forecast.calibration import run_calibration

        try:
            return run_calibration(args.threads, max_families=args.max_families, write=args.write)
        except (ValueError, OSError) as exc:
            calibrate.error(str(exc))

    # settings beside the training data remember repointed bases and the crop rule
    settings = paths.LABELS_DIR.parent
    app = build_app(
        backend=args.backend,
        ui_dir=STATIC,
        tabs=[hyperparams_tab(), forecast_tab()],
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
