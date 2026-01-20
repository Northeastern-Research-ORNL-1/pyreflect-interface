from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_get_limits_local_dev(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import limits_access

    monkeypatch.setattr(limits_access, "IS_PRODUCTION", False)
    monkeypatch.setattr(limits_access, "LOCAL_LIMITS", {"max_epochs": 1000})
    monkeypatch.setattr(limits_access, "PRODUCTION_LIMITS", {"max_epochs": 50})

    resp = client.get("/api/limits")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["limits"] == {"max_epochs": 1000}
    assert payload["access_granted"] is True
    assert payload["limit_source"] == "local_dev"


def test_get_limits_production_uses_whitelist(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.services import limits_access

    monkeypatch.setattr(limits_access, "IS_PRODUCTION", True)
    monkeypatch.setattr(limits_access, "LOCAL_LIMITS", {"max_epochs": 1000})
    monkeypatch.setattr(limits_access, "PRODUCTION_LIMITS", {"max_epochs": 50})
    monkeypatch.setattr(limits_access, "LIMITS_WHITELIST_USER_IDS", ["u"])  # type: ignore[attr-defined]

    # no code
    r0 = client.get("/api/limits")
    assert r0.status_code == 200
    assert r0.json()["limit_source"] == "production"
    assert r0.json()["access_granted"] is False

    # whitelisted user
    r1 = client.get("/api/limits", headers={"X-User-ID": "u"})
    assert r1.status_code == 200
    assert r1.json()["limit_source"] == "whitelist"
    assert r1.json()["access_granted"] is True
    assert r1.json()["limits"] == {"max_epochs": 1000}
