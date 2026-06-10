"""Shared pytest fixtures for the mod3 test suite.

Key responsibilities:
  1. sys.modules stub isolation — any test that injects fake modules into
     sys.modules must restore the original state after the test (or module)
     completes.  Without this, a stub written by test_version.py's
     _stub_heavy_deps() persists for the rest of the session and causes
     ImportError / AttributeError in unrelated tests that later try to import
     the real engine, vad, or other modules.

  2. testpaths is declared in pyproject.toml [tool.pytest.ini_options] so
     pytest never walks the worktree root or vendor/ dirs during collection.
"""

from __future__ import annotations

import sys
from collections.abc import Generator

import pytest


@pytest.fixture()
def restore_sys_modules() -> Generator[None, None, None]:
    """Save and restore sys.modules around a single test.

    Usage — request this fixture in any test or fixture that injects stubs:

        def test_something(restore_sys_modules):
            sys.modules["mymod"] = fake_mod
            ...  # sys.modules is restored when the test exits

    The fixture saves the *keys* that existed before the test and removes any
    new keys added during the test.  It also restores the original object for
    any key that was present before and got overwritten, ensuring that a stub
    written by one test does not leak into the next.
    """
    snapshot: dict[str, object] = dict(sys.modules)
    yield
    # Remove keys added during the test
    added = set(sys.modules) - set(snapshot)
    for key in added:
        del sys.modules[key]
    # Restore keys that were overwritten or deleted
    for key, mod in snapshot.items():
        if sys.modules.get(key) is not mod:
            sys.modules[key] = mod
