from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_models_require_user_id_in_production(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from service.routers import models as models_router

    monkeypatch.setattr(models_router, "IS_PRODUCTION", True)

    r = client.get("/api/models/x")
    assert r.status_code == 401

    r = client.delete("/api/models/x")
    assert r.status_code == 401

    r = client.get("/api/models/x/info")
    assert r.status_code == 401


def test_model_access_denied_when_meta_mismatch(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_backend_layout: dict[str, Path]
) -> None:
    from service.routers import models as models_router

    monkeypatch.setattr(models_router, "IS_PRODUCTION", True)

    models_dir = tmp_backend_layout["models_dir"]
    (models_dir / "m.pth").write_bytes(b"test")
    (models_dir / "m.meta.json").write_text('{"user_id":"owner"}', encoding="utf-8")

    r = client.get("/api/models/m", headers={"X-User-ID": "other"})
    assert r.status_code == 403


def test_model_access_denied_when_meta_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_backend_layout: dict[str, Path]
) -> None:
    from service.routers import models as models_router

    monkeypatch.setattr(models_router, "IS_PRODUCTION", True)

    models_dir = tmp_backend_layout["models_dir"]
    (models_dir / "m2.pth").write_bytes(b"test")

    r = client.get("/api/models/m2", headers={"X-User-ID": "someone"})
    assert r.status_code == 403


def test_model_delete_works_with_owner(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_backend_layout: dict[str, Path]
) -> None:
    from service.routers import models as models_router

    monkeypatch.setattr(models_router, "IS_PRODUCTION", True)

    models_dir = tmp_backend_layout["models_dir"]
    (models_dir / "m3.pth").write_bytes(b"test")
    (models_dir / "m3.meta.json").write_text('{"user_id":"u"}', encoding="utf-8")

    r = client.delete("/api/models/m3", headers={"X-User-ID": "u"})
    assert r.status_code == 200
    assert (models_dir / "m3.pth").exists() is False
