from __future__ import annotations

import pytest


def test_get_bool_env_returns_default_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.config import _get_bool_env

    monkeypatch.delenv("SOME_BOOL", raising=False)
    assert _get_bool_env("SOME_BOOL", default=True) is True
    assert _get_bool_env("SOME_BOOL", default=False) is False


def test_squash_whitespace_handles_none_and_strips_all_whitespace() -> None:
    from service.config import _squash_whitespace

    assert _squash_whitespace(None) is None
    assert _squash_whitespace("  a b\n\t c  ") == "abc"
