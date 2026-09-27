"""`uv run smolsmort` - starts the review web tool (find, select, train) and blocks until Ctrl-C.

this is the real review server from smolsmort.review.server, serving the page in review_ui. the
hyperparams menu the standalone page used to be is the train tab's config menu.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smolsmort.forecast.tab import forecast_tab
from smolsmort.review import paths
from smolsmort.review.server import build_app, serve
from smolsmort.review_ui.server import STATIC
from smolsmort.review_ui.tab import hyperparams_tab


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smolsmort", description="the review web tool")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default 8080)")
    parser.add_argument("--backend", default="heatmap", help="model backend name")
    commands = parser.add_subparsers(dest="command")
    crops = commands.add_parser("crops", help="train and evaluate crop classifiers")
    crop_commands = crops.add_subparsers(dest="crops_command", required=True)
    train = crop_commands.add_parser("train", help="train crop classifier weights")
    train.add_argument("--set", required=True, type=Path)
    train.add_argument("--heads", required=True, type=Path)
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=20)
    evaluate = crop_commands.add_parser("eval", help="evaluate crop classifier weights")
    evaluate.add_argument("--set", required=True, type=Path)
    evaluate.add_argument("--weights", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "crops":
        from smolsmort.crops.dataset import load_heads, load_set, split_groups
        from smolsmort.crops.train import evaluate as evaluate_crops
        from smolsmort.crops.train import load, save
        from smolsmort.crops.train import train as train_crops

        examples = load_set(args.set)
        if args.crops_command == "train":
            heads = load_heads(args.heads, examples)
            training, held_out = split_groups(examples)
            model, history = train_crops(training, heads, epochs=args.epochs)
            save(model, heads, args.out)
            print(f"saved {args.out} after {len(history)} epochs")
            if held_out:
                print(evaluate_crops(model, heads, held_out))
        else:
            model, heads = load(args.weights)
            print(evaluate_crops(model, heads, examples))
        return 0

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
