"""prepared parquet tables as numpy columns: text as object arrays, dates as datetime64[D]"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

AUTO = "auto"
SCAN_ROWS = 1000

# "1.234,5" / "0,6" and "1,234.5" - a value fitting both (plain "1,234") decides nothing
DECIMAL_COMMA = re.compile(r"[+-]?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?")
THOUSANDS_COMMA = re.compile(r"[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?")
NUMBER_SQL = {
    "decimal_comma": "TRY_CAST(replace(replace(trim({q}), '.', ''), ',', '.') AS DOUBLE)",
    "thousands_comma": "TRY_CAST(replace(trim({q}), ',', '') AS DOUBLE)",
}


def resolve_encoding(data: bytes, requested: str | None) -> str:
    """the codec to decode a source with: the requested one, or a guess when it is auto or empty.
    a bom is certain; without one, utf-8 if the bytes decode, else cp1252 (excel's default)"""
    if requested and requested.lower() != AUTO:
        return requested
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "cp1252"
    return "utf-8"


def number_format(values: list) -> str | None:
    """decimal_comma or thousands_comma when every value fits exactly that one format, else none"""
    texts = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not texts:
        return None
    decimal = all(DECIMAL_COMMA.fullmatch(t) for t in texts)
    thousands = all(THOUSANDS_COMMA.fullmatch(t) for t in texts)
    if decimal and not thousands:
        return "decimal_comma"
    if thousands and not decimal:
        return "thousands_comma"
    return None


def type_source(con, read_expr: str) -> str:
    """the source with text columns holding locale-formatted numbers cast to double, judged on
    the first SCAN_ROWS rows; later values that do not parse become null"""
    described = con.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
    text_cols = [name for name, kind, *_ in described if kind.upper() == "VARCHAR"]
    if not text_cols:
        return read_expr
    quoted = {name: '"' + name.replace('"', '""') + '"' for name in text_cols}
    rows = con.execute(
        f"SELECT {', '.join(quoted.values())} FROM {read_expr} LIMIT {SCAN_ROWS}"
    ).fetchall()
    replaced = []
    for i, name in enumerate(text_cols):
        found = number_format([row[i] for row in rows])
        if found:
            q = quoted[name]
            replaced.append(f"{NUMBER_SQL[found].format(q=q)} AS {q}")
    if not replaced:
        return read_expr
    return f"(SELECT * REPLACE ({', '.join(replaced)}) FROM {read_expr})"


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
