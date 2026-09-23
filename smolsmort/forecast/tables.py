"""prepared parquet tables as numpy columns: text as object arrays, dates as datetime64[D]"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_table(path: Path, order_by: str | None = None) -> dict[str, np.ndarray]:
    import duckdb

    order = f" ORDER BY {order_by}" if order_by else ""
    columns = duckdb.sql(f"SELECT * FROM read_parquet('{path}'){order}").fetchnumpy()
    out = {}
    for name, values in columns.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.datetime64):
            array = array.astype("datetime64[D]")
        elif array.dtype.kind in "fiu":
            array = array.astype(float)
        out[name] = array
    return out


def is_text(values: np.ndarray) -> bool:
    return values.dtype == object or values.dtype.kind in "US"


def write_table(path: Path, columns: dict[str, np.ndarray]) -> Path:
    """numpy columns to parquet through duckdb; dates go in as nanosecond timestamps, the one
    datetime width duckdb's numpy scan accepts"""
    import duckdb

    data = {}
    for name, values in columns.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.datetime64):
            array = array.astype("datetime64[ns]")
        data[name] = array
    con = duckdb.connect()
    con.register("frame", data)
    con.execute(f"COPY frame TO '{path}' (FORMAT PARQUET)")
    con.close()
    return path
