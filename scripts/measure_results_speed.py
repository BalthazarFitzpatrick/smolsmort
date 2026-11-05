"""measure saved-run readers without starting a server"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from smolsmort.forecast import tab, tiles, views


def main():
    folder = Path(sys.argv[1])
    if len(sys.argv) > 3:
        spec = importlib.util.spec_from_file_location("original_tiles", sys.argv[3])
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        tiles.read_tiles = original.read_tiles
        tiles.ensemble_rows = original.ensemble_rows
        tiles.read_ensemble_rows = original.ensemble_rows
    request = views.read_request(folder)
    app = SimpleNamespace(state=SimpleNamespace(bases={"forecast": folder.parent.parent}))
    calls = {
        "read_tiles": lambda: tiles.read_tiles(folder),
        "view_plain": lambda: views.view(folder, request, {}),
        "view_breakdown": lambda: views.view(folder, request, {"breakdown": "client_group"}),
        "view_ensemble": lambda: views.view(folder, request, {"model": "ensemble"}),
        "run_tiles": lambda: tab._run_tiles(app, {"run": folder.name}),
    }
    for name, call in calls.items():
        start = time.perf_counter()
        call()
        print(name, round(time.perf_counter() - start, 6), flush=True)
    if len(sys.argv) > 2:
        rows = tiles.ensemble_rows(folder)
        golden = [
            {
                "tile_id": row["tile_id"],
                "metrics": {key: row["metrics"][key] for key in ("wape", "mae", "mape")},
            }
            for row in rows
        ]
        Path(sys.argv[2]).write_text(json.dumps(golden), encoding="utf-8")


if __name__ == "__main__":
    main()
