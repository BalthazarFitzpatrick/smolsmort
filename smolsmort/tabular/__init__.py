"""the tabular model backend seam implementation #3: rows in, an xgboost model out.

behind the same ModelBackend seam as the two vision backends (`smolsmort/detect`, `smolsmort/boxes`)
- see docs/REVIEW_TOOL_DESIGN.md. a "frame" here is a path to a one-row json feature file rather
than an image, and `predict` answers a class + score per row rather than a box.
"""
