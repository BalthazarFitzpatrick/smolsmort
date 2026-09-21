"""suite-wide pytest setup.

SET BEFORE ANY TEST MODULE IMPORTS. torch (the heatmap/box backends) bundles its own openmp
runtime; xgboost's homebrew libomp aborts or segfaults the process on macos if it finds one
already initialised. conftest.py loads before every test module in the session, which is the
only point early enough to matter - setting this inside smolsmort/tabular/backend.py was too
late, because torch tests earlier in the same run had already initialised their openmp first.
"""

import os
import sys

# macos only: the clash is torch's bundled openmp against homebrew's libomp. linux (the ci image)
# has one runtime, and single-threading every torch op there just made the suite slower
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # single-threaded: allowing torch's openmp pool and xgboost's to both spin up threads
    # segfaults the process even with the duplicate-lib check disabled above
    os.environ.setdefault("OMP_NUM_THREADS", "1")
