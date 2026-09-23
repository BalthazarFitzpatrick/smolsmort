"""prepared parquet tables as numpy columns: text as object arrays, dates as datetime64[D]"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def connect():
    """a duckdb connection, importing duckdb only now: the vision half and the cli must run on a
    plain install without the forecast extra"""
    import duckdb

    return duckdb.connect()


def read_table(path: Path, order_by: str | None = None) -> dict[str, np.ndarray]:
    import duckdb

    order = f" ORDER BY {order_by}" if order_by else ""
    columns = duckdb.sql(f"SELECT * FROM read_parquet('{path}'){order}").fetchnumpy()
    return {name: _unmask(values) for name, values in columns.items()}


def _unmask(values) -> np.ndarray:
    """a plain array with nulls as nan, nat or none. fetchnumpy marks nulls with a mask, and
    np.asarray drops the mask and keeps whatever the buffer held - an open row's target became
    a number that way"""
    mask = np.ma.getmaskarray(values)
    array = np.ma.getdata(values)
    if np.issubdtype(array.dtype, np.datetime64):
        out = array.astype("datetime64[D]")
        return np.where(mask, np.datetime64("NaT", "D"), out)
    if array.dtype.kind in "fiub":
        return np.where(mask, np.nan, array.astype(float))
    out = array.astype(object)
    out[mask] = None
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
