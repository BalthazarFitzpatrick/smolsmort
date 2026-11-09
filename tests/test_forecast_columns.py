"""source metadata and filter domains through the forecast routes"""

from types import SimpleNamespace

import pytest

from smolsmort.forecast.tab import forecast_tab
from smolsmort.review.routes import RequestError

pytest.importorskip("duckdb")


@pytest.fixture
def source_app(tmp_path):
    return SimpleNamespace(state=SimpleNamespace(bases={"forecast": str(tmp_path)}))


def test_columns_return_date_and_number_bounds(source_app, tmp_path):
    (tmp_path / "bounds.csv").write_text(
        "day,amount,region\n2023-04-03,-3,west\n2026-10-25,19,east\n,,west\n",
        encoding="utf-8",
    )
    result = forecast_tab().post["/api/forecast-columns"](source_app, {"source": "bounds.csv"})
    columns = {column["name"]: column for column in result["columns"]}
    assert columns["day"]["kind"] == "date"
    assert (columns["day"]["min"], columns["day"]["max"]) == ("2023-04-03", "2026-10-25")
    assert (columns["amount"]["min"], columns["amount"]["max"]) == (-3, 19)
    assert "min" not in columns["region"]
    assert result["row_count"] == 3


@pytest.fixture
def domain_query(source_app, tmp_path):
    (tmp_path / "domain.csv").write_text(
        '"odd""field",amount\nwest,1\nwest,2\nwest,3\nEast,4\nEast,5\no\'brien,6\n,7\n,8\n',
        encoding="utf-8",
    )

    def fetch(**query):
        return forecast_tab().get["/api/forecast-column-values"](
            source_app, {"source": "domain.csv", "column": 'odd"field', **query}
        )

    return fetch


def test_column_values_return_counts_nulls_and_order(domain_query):
    result = domain_query()
    assert result["values"] == [
        {"value": "west", "count": 3},
        {"value": "East", "count": 2},
        {"value": "o'brien", "count": 1},
        {"value": None, "count": 2},
    ]
    assert result["null_count"] == 2


def test_column_values_search_is_bound_case_insensitive_and_literal(domain_query):
    assert domain_query(q="EAsT")["values"] == [
        {"value": "East", "count": 2},
        {"value": None, "count": 2},
    ]
    assert domain_query(q="o'b")["values"][0] == {"value": "o'brien", "count": 1}
    assert domain_query(q="' OR TRUE --")["values"] == [{"value": None, "count": 2}]
    assert domain_query(q="%")["values"] == [{"value": None, "count": 2}]


def test_column_values_limit_and_validation(domain_query):
    assert domain_query(limit="1")["values"] == [
        {"value": "west", "count": 3},
        {"value": None, "count": 2},
    ]
    with pytest.raises(RequestError, match="limit must be an integer"):
        domain_query(limit="bad")
    with pytest.raises(RequestError, match="unknown column"):
        domain_query(column='odd"field" OR TRUE')
    with pytest.raises(RequestError, match="outside the forecast root"):
        domain_query(source="../outside.csv")


def test_column_values_caps_limit_and_handles_utf16(source_app, tmp_path):
    rows = "region\n" + "\n".join(f"region-{index:03}" for index in range(220)) + "\n"
    (tmp_path / "encoded.csv").write_bytes(rows.encode("utf-16"))
    result = forecast_tab().get["/api/forecast-column-values"](
        source_app,
        {"source": "encoded.csv", "column": "region", "limit": "999", "encoding": "auto"},
    )
    assert len(result["values"]) == 200
    assert result["encoding"] == "utf-16"
    assert result["null_count"] == 0
