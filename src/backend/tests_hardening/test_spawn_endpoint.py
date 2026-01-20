from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_queue_spawn_not_protected_in_dev(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.routers import jobs as jobs_router

    monkeypatch.setattr(jobs_router, "IS_PRODUCTION", False)
    monkeypatch.setattr(jobs_router, "ADMIN_TOKEN", None)

    class _FakeRQ:
        available = True

    monkeypatch.setattr(jobs_router, "_get_rq_or_reconnect", lambda http_request: _FakeRQ())
    monkeypatch.setattr(jobs_router, "_maybe_trigger_modal_gpu_worker", lambda rq: {"triggered": False})

    resp = client.post("/api/queue/spawn")
    assert resp.status_code == 200
    assert resp.json() == {"remote_worker": {"triggered": False}}


def test_queue_spawn_requires_token_in_production(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.routers import jobs as jobs_router

    monkeypatch.setattr(jobs_router, "IS_PRODUCTION", True)
    monkeypatch.setattr(jobs_router, "ADMIN_TOKEN", "t")

    class _FakeRQ:
        available = True

    monkeypatch.setattr(jobs_router, "_get_rq_or_reconnect", lambda http_request: _FakeRQ())
    monkeypatch.setattr(jobs_router, "_maybe_trigger_modal_gpu_worker", lambda rq: {"triggered": False})

    r0 = client.post("/api/queue/spawn")
    assert r0.status_code == 401

    r1 = client.post("/api/queue/spawn", headers={"X-Admin-Token": "wrong"})
    assert r1.status_code == 401

    r2 = client.post("/api/queue/spawn", headers={"X-Admin-Token": "t"})
    assert r2.status_code == 200


def test_queue_spawn_requires_admin_token_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.routers import jobs as jobs_router

    monkeypatch.setattr(jobs_router, "IS_PRODUCTION", True)
    monkeypatch.setattr(jobs_router, "ADMIN_TOKEN", None)

    resp = client.post("/api/queue/spawn")
    assert resp.status_code == 503
