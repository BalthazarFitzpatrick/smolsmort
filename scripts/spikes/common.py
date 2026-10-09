"""prepare the read-only bike source with the production workspace code"""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from smolsmort.forecast.pipeline import load_workspace
from smolsmort.forecast.prep import prepare
from smolsmort.forecast.spec import SERIES, STEP, Column, PrepSpec
from smolsmort.forecast.tables import read_table

SOURCE = Path("/Users/fabianzimmer/Documents/dev/ai_lab/smolsmort/sample_data/bikes/day.csv")


@contextmanager
def bike_workspace(source=SOURCE, unit="day", horizon=30):
    spec = PrepSpec(
        source=str(source),
        mode="series",
        task="regression",
        columns=(Column("dteday", "time"), Column("cnt", "target", "sum")),
        step=unit,
        horizon=horizon,
    )
    with TemporaryDirectory(prefix="multifamily-prep-", dir="/private/tmp") as folder:
        prepared = prepare(spec, Path(folder))
        workspace = load_workspace(spec, prepared.folder, prepared.summary)
        panel = read_table(prepared.folder / "panel.parquet", order_by=f"{SERIES}, {STEP}")
        data = pd.DataFrame(
            {"unique_id": panel[SERIES], "ds": pd.to_datetime(panel[STEP]), "y": panel["cnt"]}
        )
        yield workspace, data
