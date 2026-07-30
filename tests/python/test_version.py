"""The runtime version is evidence, not a claim.

WHY THIS FILE EXISTS

A consumer that records which extractor produced a document needs to obtain the
version FROM the package. Before ``__version__`` existed there was no way to, so
the only option we offered was hand-copying a string -- and a hand-copied
version is a claim about the runtime rather than evidence of it. A downstream
consumer did exactly that and its copy went stale (declared 0.5.0 against a
0.6.0 runtime) with nothing able to detect the drift.

These tests tie every place the version is written to every other place, so any
single-sided edit fails here instead of shipping.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages" / "python" / "src"))

import synapt.extract
from synapt.extract import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "packages" / "python" / "pyproject.toml"
TS_PACKAGE_JSON = REPO_ROOT / "packages" / "ts" / "package.json"

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _pyproject_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.MULTILINE)
    assert match is not None, f"no version line in {PYPROJECT}"
    return match.group(1)


def _ts_package_version() -> str:
    import json

    return json.loads(TS_PACKAGE_JSON.read_text())["version"]


def test_version_matches_the_distribution_metadata():
    assert __version__ == _pyproject_version()


def test_version_matches_the_typescript_package():
    """The two language surfaces ship as one product at one version.

    Nothing enforced this before: they were two hand-edited numbers that
    happened to agree."""
    assert __version__ == _ts_package_version()


def test_version_is_exported_from_the_package_root():
    """A consumer must reach it without knowing the internal module layout."""
    assert synapt.extract.__version__ == __version__


def test_version_is_a_bare_semver_triple():
    """Guards the shape stamped into provenance. A range (">=0.6.0") or a full
    specifier ("@synapt-dev/extract@0.6.0") are each a plausible thing to paste
    into this constant, and each would corrupt the recorded value."""
    assert _SEMVER.match(__version__), f"not a bare semver triple: {__version__!r}"


def test_installed_distribution_agrees_when_the_package_is_installed():
    """importlib.metadata reads what pip actually installed, which is a
    genuinely different source than the literal in __init__.py -- so this
    catches a stale editable install or a version-bump that never got reinstalled,
    which the other assertions here cannot see.

    Skipped rather than failed when the package is imported from a source tree
    with no installed distribution, since that is a legitimate way to use it.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("synapt-extract")
    except PackageNotFoundError:
        pytest.skip("synapt-extract is not installed as a distribution")
    assert installed == __version__
