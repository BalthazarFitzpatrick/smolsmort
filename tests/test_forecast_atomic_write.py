"""concurrent readers of one run write its route cursors without clobbering each other"""

import json
import threading

from smolsmort.forecast.views import write_json_atomic


def test_many_writers_leave_one_valid_file_and_no_temp_files(tmp_path):
    target = tmp_path / "route-cursors.json"
    errors = []

    def write(n):
        try:
            for i in range(40):
                write_json_atomic(target, {"writer": n, "i": i, "pad": "x" * 2000})
        except OSError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert json.loads(target.read_text(encoding="utf-8"))["i"] == 39
    assert [p.name for p in tmp_path.iterdir()] == ["route-cursors.json"]
