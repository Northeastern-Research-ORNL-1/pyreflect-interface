from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Ensure `import service` resolves even if pytest changes import semantics.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture()
def tmp_backend_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Patch backend storage paths to an isolated temp dir."""

    root = tmp_path / "backend_root"
    data_dir = root / "data"
    curves_dir = data_dir / "curves"
    expt_dir = curves_dir / "expt"
    models_dir = data_dir / "models"
    settings_path = root / "settings.yml"

    expt_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("root: .\n", encoding="utf-8")

    # Patch config module constants.
    from service import config as cfg

    monkeypatch.setattr(cfg, "BACKEND_ROOT", root)
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir)
    monkeypatch.setattr(cfg, "CURVES_DIR", curves_dir)
    monkeypatch.setattr(cfg, "EXPT_DIR", expt_dir)
    monkeypatch.setattr(cfg, "MODELS_DIR", models_dir)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)

    # Patch settings_store module bindings (it imported from config).
    from service import settings_store

    monkeypatch.setattr(settings_store, "BACKEND_ROOT", root)
    monkeypatch.setattr(settings_store, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings_store, "CURVES_DIR", curves_dir)
    monkeypatch.setattr(settings_store, "EXPT_DIR", expt_dir)
    monkeypatch.setattr(settings_store, "MODELS_DIR", models_dir)
    monkeypatch.setattr(settings_store, "SETTINGS_PATH", settings_path)

    # Patch routers that imported these paths directly.
    from service.routers import models as models_router

    monkeypatch.setattr(models_router, "MODELS_DIR", models_dir)
    monkeypatch.setattr(models_router, "HF_REPO_ID", None)

    return {
        "root": root,
        "data_dir": data_dir,
        "curves_dir": curves_dir,
        "expt_dir": expt_dir,
        "models_dir": models_dir,
        "settings_path": settings_path,
    }


@pytest.fixture()
def app(tmp_backend_layout: dict[str, Path]) -> FastAPI:
    """Minimal app for router-level tests (no external integrations)."""

    from service.routers.generate import router as generate_router
    from service.routers.health import router as health_router
    from service.routers.jobs import router as jobs_router
    from service.routers.models import router as models_router
    from service.routers.upload import router as upload_router

    app = FastAPI()
    app.include_router(health_router, prefix="/api")
    app.include_router(generate_router, prefix="/api")
    app.include_router(upload_router, prefix="/api")
    app.include_router(models_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")
    return app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Test client for the FastAPI app."""
    return TestClient(app)
