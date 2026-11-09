from types import SimpleNamespace

from smolsmort.forecast import tab


def app_for(root):
    return SimpleNamespace(state=SimpleNamespace(bases={"forecast": root}))


def test_sources_list_data_files_and_skip_hidden_folders(tmp_path):
    for name in (
        "units.csv",
        "exports/weekly.csv",
        "exports/spec.json",
        ".forecast-runs/run-1/request.json",
        ".forecast-runs-archive-20261104/run-2/result.json",
        ".forecast-cache/prepared.csv",
        ".warm-seeds-total/lightgbm.json",
        ".venv/lib/pkg/data.json",
        "nested/.hidden/notes.csv",
        "notes.txt",
        ".hidden-file.csv",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("a,b\n1,2\n")
    listing = tab._sources(app_for(tmp_path), {})
    assert [item["path"] for item in listing["files"]] == [
        "exports/spec.json",
        "exports/weekly.csv",
        "units.csv",
    ]
    assert listing["root"] == str(tmp_path)


def test_sources_empty_for_a_missing_root(tmp_path):
    assert tab._sources(app_for(tmp_path / "missing"), {})["files"] == []
