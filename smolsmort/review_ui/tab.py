"""the hyperparams menu's endpoint, registered as a host tab so the real review server serves it.

the review server has no hyperparams route of its own; the train tab's config menu reads
/api/hyperparams, so this hands it over without touching smolsmort/review.
"""

from __future__ import annotations

from smolsmort.review.routes import RequestError, Tab
from smolsmort.review_ui import logic


def _menu_options(app, query: dict) -> dict:
    try:
        return logic.menu_options(query.get("backend", "heatmap"))
    except logic.RequestError as exc:
        raise RequestError(str(exc)) from exc


def hyperparams_tab() -> Tab:
    return Tab(name="hyperparams", get={"/api/hyperparams": _menu_options})
