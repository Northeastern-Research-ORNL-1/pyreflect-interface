from __future__ import annotations

import pytest


@pytest.mark.parametrize("user_id", [None, "", "   "])
def test_is_whitelisted_user_rejects_missing_user_id(user_id: str | None) -> None:
    from service.services.limits_access import is_whitelisted_user

    assert is_whitelisted_user(user_id=user_id, whitelist=["u"]) is False


def test_is_whitelisted_user_matches_exact_id() -> None:
    from service.services.limits_access import is_whitelisted_user

    assert is_whitelisted_user(user_id="u", whitelist=["u"]) is True
    assert is_whitelisted_user(user_id="u", whitelist=["other"]) is False


def test_get_effective_limits_local_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import limits_access

    monkeypatch.setattr(limits_access, "IS_PRODUCTION", False)
    monkeypatch.setattr(limits_access, "LOCAL_LIMITS", {"max_epochs": 1000})
    monkeypatch.setattr(limits_access, "PRODUCTION_LIMITS", {"max_epochs": 50})

    limits, granted, source = limits_access.get_effective_limits(user_id=None)
    assert limits == {"max_epochs": 1000}
    assert granted is True
    assert source == "local_dev"


def test_get_effective_limits_production_whitelisted(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import limits_access

    monkeypatch.setattr(limits_access, "IS_PRODUCTION", True)
    monkeypatch.setattr(limits_access, "LOCAL_LIMITS", {"max_epochs": 1000})
    monkeypatch.setattr(limits_access, "PRODUCTION_LIMITS", {"max_epochs": 50})
    monkeypatch.setattr(limits_access, "LIMITS_WHITELIST_USER_IDS", ["u"])  # type: ignore[attr-defined]

    limits, granted, source = limits_access.get_effective_limits(user_id="u")
    assert limits == {"max_epochs": 1000}
    assert granted is True
    assert source == "whitelist"


def test_get_effective_limits_production_non_whitelisted(monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import limits_access

    monkeypatch.setattr(limits_access, "IS_PRODUCTION", True)
    monkeypatch.setattr(limits_access, "LOCAL_LIMITS", {"max_epochs": 1000})
    monkeypatch.setattr(limits_access, "PRODUCTION_LIMITS", {"max_epochs": 50})
    monkeypatch.setattr(limits_access, "LIMITS_WHITELIST_USER_IDS", ["someone-else"])  # type: ignore[attr-defined]

    limits, granted, source = limits_access.get_effective_limits(user_id="u")
    assert limits == {"max_epochs": 50}
    assert granted is False
    assert source == "production"
