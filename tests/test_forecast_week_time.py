"""iso year-week time fields keep calendar boundaries and reject invalid weeks"""

from datetime import date

import duckdb
import pytest

from smolsmort.forecast.prep import _infer_step, _time_expression, prepare
from smolsmort.forecast.spec import Column, PrepSpec


@pytest.mark.parametrize("kind", ["INTEGER", "VARCHAR"])
def test_year_weeks_become_iso_mondays(kind):
    with duckdb.connect() as con:
        con.execute(f"CREATE TABLE source (swo_week {kind})")
        con.execute("INSERT INTO source VALUES (202601), (202643), (202653)")
        expression = _time_expression(con, "source", "swo_week", "TRUE")
        assert con.execute(f"SELECT {expression} FROM source ORDER BY swo_week").fetchall() == [
            (date(2025, 12, 29),),
            (date(2026, 10, 19),),
            (date(2026, 12, 28),),
        ]


@pytest.mark.parametrize("value", [202600, 202654, 202553, 12345, 1000001, 101])
def test_bad_year_week_names_column(tmp_path, value):
    source = tmp_path / "weeks.csv"
    source.write_text(f"swo_week,units\n{value},1\n")
    spec = PrepSpec(
        str(source),
        "timeseries",
        "regression",
        (Column("swo_week", "time"), Column("units", "target", aggregation="sum")),
        step="week",
    )
    with pytest.raises(ValueError, match="invalid year-week in column 'swo_week'"):
        prepare(spec, tmp_path / "cache")


def test_week_inference_and_preparation(tmp_path):
    source = tmp_path / "weeks.csv"
    source.write_text("swo_week,units\n202601,1\n202602,2\n202603,3\n")
    spec = PrepSpec(
        str(source),
        "timeseries",
        "regression",
        (Column("swo_week", "time"), Column("units", "target", aggregation="sum")),
    )
    prepared = prepare(spec, tmp_path / "cache")
    with duckdb.connect() as con:
        assert _infer_step(con, f"read_csv('{source}')", "swo_week", "TRUE") == "week"
        rows = con.execute(
            f"SELECT __step, units FROM '{prepared.folder / 'panel.parquet'}' ORDER BY __step"
        ).fetchall()
    assert [(row[0].date(), row[1]) for row in rows] == [
        (date(2025, 12, 29), 1),
        (date(2026, 1, 5), 2),
        (date(2026, 1, 12), 3),
    ]


@pytest.mark.parametrize("kind", ["DATE", "TIMESTAMP", "VARCHAR"])
def test_date_columns_keep_their_dates(kind):
    with duckdb.connect() as con:
        con.execute(f"CREATE TABLE source (day {kind})")
        con.execute("INSERT INTO source VALUES ('2026-10-21'), (NULL)")
        expression = _time_expression(con, "source", "day", "TRUE")
        assert con.execute(f"SELECT {expression} FROM source").fetchall() == [
            (date(2026, 10, 21),),
            (None,),
        ]
