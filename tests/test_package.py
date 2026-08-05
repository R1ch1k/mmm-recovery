"""Smoke test: the src-layout package is importable from the installed environment."""

import mmm_recovery


def test_version_is_exposed() -> None:
    assert isinstance(mmm_recovery.__version__, str)
    assert mmm_recovery.__version__
