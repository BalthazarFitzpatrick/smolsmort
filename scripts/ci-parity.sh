#!/usr/bin/env bash
# run exactly what a clean checkout would run, against ONLY the files git actually has.
#
# WHY THIS EXISTS. The suite kept passing locally and failing on a fresh clone. The cause is not
# flaky tests: this working tree holds ~12 MB of gitignored screenshots, the curated kernel pools
# and recorded sessions, and a clean checkout has none of it. A test that quietly reads one of
# those passes here and fails there, and nothing local ever shows the difference.
#
# `git archive` is the honest reproduction: it writes the tracked files at a commit into a temp
# directory and touches NOTHING in the repo. This used to use `git worktree add`, which was a
# mistake twice over - it writes registrations into .git/, and when the cleanup failed to remove
# one it left a stale entry pointing at a deleted temp dir. Worse, pre-commit saw the repo change
# underneath it and failed the hook with "files were modified by this hook" even though the tests
# had all passed. A check that reports failure for touching the thing it is checking is worse than
# no check.
#
# The one difference this CANNOT catch is the platform: a Linux runner is not macOS. Anything
# importing Quartz/pynput/mss, or depending on path casing or line endings, still needs one.
#
#   ./scripts/ci-parity.sh            # the current HEAD commit
#   ./scripts/ci-parity.sh <ref>      # any ref, e.g. what is actually pushed:
#                                     #   ./scripts/ci-parity.sh origin/my-branch
set -euo pipefail

ref="${1:-HEAD}"
repo_root="$(git rev-parse --show-toplevel)"
scratch="$(mktemp -d)"

cleanup() { rm -rf "$scratch"; }
trap cleanup EXIT

echo "==> exporting $ref (tracked files only, repo untouched)"
git -C "$repo_root" archive --format=tar "$ref" | tar -x -C "$scratch"
cd "$scratch"

# uncommitted work is the other way local and a clean checkout diverge, so say so rather than let
# a green run here imply the dirty tree is fine
if ! git -C "$repo_root" diff --quiet || ! git -C "$repo_root" diff --cached --quiet; then
    echo "    NOTE: $repo_root has uncommitted changes, which are NOT in this run"
fi

echo "==> uv sync --dev"
uv sync --dev >/dev/null

echo "==> ruff check ."
uv run ruff check .

echo "==> ruff format --check ."
uv run ruff format --check .

echo "==> pytest"
uv run pytest -q

echo
echo "PASSED parity for $ref (macOS - the linux-only differences are still untested)"
