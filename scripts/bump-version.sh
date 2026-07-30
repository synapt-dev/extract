#!/bin/bash
# Bump the synapt-extract version across every place it is written.
#
# Usage:
#   ./scripts/bump-version.sh 0.7.0
#   ./scripts/bump-version.sh patch     # 0.6.0 -> 0.6.1
#   ./scripts/bump-version.sh minor     # 0.6.0 -> 0.7.0
#   ./scripts/bump-version.sh major     # 0.6.0 -> 1.0.0
#
# WHY THIS EXISTS
#
# The version is written in FOUR places: the two package manifests and the two
# runtime constants that let a consumer read the version instead of hand-copying
# it. Four hand-edited numbers is precisely the shape that drifts, and a stale
# version is not a cosmetic problem here -- it is recorded into downstream
# provenance as evidence of what produced a document.
#
# `tests/python/test_version.py` and `packages/ts/tests/test_version.ts` fail
# when any one of the four drifts. This script is how you avoid tripping them.
#
# It deliberately does NOT commit, tag, or push. Under the dev/main branching
# model a tag belongs to the release ceremony on `main`, not to whatever branch
# happens to be checked out when someone bumps a number.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYPROJECT="$REPO_ROOT/packages/python/pyproject.toml"
INIT_PY="$REPO_ROOT/packages/python/src/synapt/extract/__init__.py"
TS_PKG="$REPO_ROOT/packages/ts/package.json"
TS_VERSION="$REPO_ROOT/packages/ts/src/version.ts"

for f in "$PYPROJECT" "$INIT_PY" "$TS_PKG" "$TS_VERSION"; do
    [ -f "$f" ] || { echo "Error: missing $f" >&2; exit 1; }
done

# --- Read current version (pyproject is the reference) ---
CURRENT=$(grep '^version = ' "$PYPROJECT" | head -1 | sed 's/version = "\(.*\)"/\1/')
if [ -z "$CURRENT" ]; then
    echo "Error: could not read version from $PYPROJECT" >&2
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

ARG="${1:-}"
if [ -z "$ARG" ]; then
    echo "Current version: $CURRENT"
    echo ""
    echo "Usage: $0 <version|patch|minor|major>"
    echo "  $0 patch   ->  $MAJOR.$MINOR.$((PATCH + 1))"
    echo "  $0 minor   ->  $MAJOR.$((MINOR + 1)).0"
    echo "  $0 major   ->  $((MAJOR + 1)).0.0"
    echo "  $0 0.7.0   ->  0.7.0"
    exit 0
fi

case "$ARG" in
    patch) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
    minor) NEW="$MAJOR.$((MINOR + 1)).0" ;;
    major) NEW="$((MAJOR + 1)).0.0" ;;
    *)
        if ! echo "$ARG" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
            echo "Error: '$ARG' is not a bare semver triple (x.y.z) or patch/minor/major." >&2
            echo "A range or a specifier here would be written into consumer provenance." >&2
            exit 1
        fi
        NEW="$ARG"
        ;;
esac

echo "Bumping $CURRENT -> $NEW"

# --- Apply. Each pattern is anchored so it cannot match a dependency's version.
python3 - "$NEW" "$PYPROJECT" "$INIT_PY" "$TS_PKG" "$TS_VERSION" <<'PY'
import re
import sys

new, pyproject, init_py, ts_pkg, ts_version = sys.argv[1:6]

def sub(path, pattern, replacement):
    text = open(path, encoding="utf-8").read()
    updated, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise SystemExit(f"Error: expected exactly 1 version match in {path}, found {n}")
    open(path, "w", encoding="utf-8").write(updated)

sub(pyproject,  r'^version = "[^"]+"',            f'version = "{new}"')
sub(init_py,    r'^__version__ = "[^"]+"',        f'__version__ = "{new}"')
sub(ts_pkg,     r'^(  "version": )"[^"]+"',       rf'\g<1>"{new}"')
sub(ts_version, r'^export const VERSION = "[^"]+"', f'export const VERSION = "{new}"')
PY

# --- Verify by fruit. The point of this script is that four numbers agree, so
# --- it checks that they do rather than reporting success for having run.
echo ""
echo "Verifying all four locations:"
FAIL=0
check() {
    local label="$1" actual="$2"
    printf "  %-34s %s" "$label" "$actual"
    if [ "$actual" = "$NEW" ]; then echo "  ok"; else echo "  MISMATCH (expected $NEW)"; FAIL=1; fi
}
check "pyproject.toml"        "$(grep '^version = ' "$PYPROJECT" | head -1 | sed 's/version = "\(.*\)"/\1/')"
check "python __init__.py"    "$(grep '^__version__ = ' "$INIT_PY" | head -1 | sed 's/__version__ = "\(.*\)"/\1/')"
check "ts package.json"       "$(grep '^  "version": ' "$TS_PKG" | head -1 | sed 's/.*: "\(.*\)".*/\1/')"
check "ts src/version.ts"     "$(grep '^export const VERSION = ' "$TS_VERSION" | sed 's/.*"\(.*\)".*/\1/')"

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "Error: the four locations do not agree. Nothing was committed; fix before proceeding." >&2
    exit 1
fi

echo ""
echo "All four agree at $NEW. Next:"
echo "  1. REINSTALL the python package first:  pip install -e packages/python"
echo "     (test_installed_distribution_agrees... compares importlib.metadata against"
echo "      the source constant, so it correctly reports RED until the installed"
echo "      distribution is rebuilt at $NEW. That is the check doing its job, not a bug.)"
echo "  2. run the suites  (pytest tests/python  &&  cd packages/ts && npm test)"
echo "  3. PR the bump into dev"
echo "  4. release ceremony merges dev -> main, then 'gh release create v$NEW' cuts the tag"
