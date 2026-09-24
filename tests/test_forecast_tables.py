"""prepared tables come back with their nulls as nulls, never as whatever the buffer held"""

from __future__ import annotations

import numpy as np
import pytest

from smolsmort.forecast.tables import read_table, resolve_encoding, write_table

duckdb = pytest.importorskip("duckdb")


@pytest.mark.parametrize(
    ("data", "requested", "expected"),
    [
        ("a,b\n".encode("utf-16"), "auto", "utf-16"),
        (b"\xfe\xff\x00a", "", "utf-16"),
        ("a,b\n".encode("utf-8-sig"), None, "utf-8-sig"),
        ("Müller\n".encode(), "auto", "utf-8"),
        ("Müller\n".encode("cp1252"), "auto", "cp1252"),
        ("Müller\n".encode("cp1252"), "latin-1", "latin-1"),
    ],
)
def test_resolve_encoding_trusts_a_bom_then_utf8_then_cp1252(data, requested, expected):
    assert resolve_encoding(data, requested) == expected


def test_nulls_read_back_as_nan_none_and_nat(tmp_path):
    path = tmp_path / "t.parquet"
    duckdb.sql(
        "COPY (SELECT * FROM (VALUES (1.0, 'a', 7, DATE '2026-01-05'), (NULL, NULL, NULL, NULL),"
        " (3.0, 'c', 9, DATE '2026-01-19')) t(x, s, n, d)) TO '" + str(path) + "' (FORMAT PARQUET)"
    )
    table = read_table(path)
    assert table["x"][0] == 1.0 and np.isnan(table["x"][1]) and table["x"][2] == 3.0
    assert table["s"].tolist() == ["a", None, "c"]
    assert np.isnan(table["n"][1]) and table["n"][2] == 9.0
    assert np.isnat(table["d"][1]) and str(table["d"][0]) == "2026-01-05"


def test_a_table_round_trips(tmp_path):
    columns = {
        "x": np.array([1.5, np.nan]),
        "s": np.array(["a", None], dtype=object),
        "d": np.array(["2026-01-05", "NaT"], dtype="datetime64[D]"),
    }
    back = read_table(write_table(tmp_path / "r.parquet", columns))
    assert back["x"][0] == 1.5 and np.isnan(back["x"][1])
    assert back["s"].tolist() == ["a", None]
    assert str(back["d"][0]) == "2026-01-05" and np.isnat(back["d"][1])
