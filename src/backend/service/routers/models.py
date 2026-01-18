from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi import File, Form, Header, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from ..config import HF_REPO_ID, MAX_LOCAL_MODELS, MODELS_DIR
from ..integrations.huggingface import get_remote_model_info
from ..services.local_model_limit import (
    delete_local_model,
    evict_old_models_for_user,
    models_dir_lock,
    write_model_meta,
)

router = APIRouter()

def _require_upload_token(x_model_upload_token: str | None) -> None:
    expected = os.getenv("MODEL_UPLOAD_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="MODEL_UPLOAD_TOKEN not configured on backend")
    if not x_model_upload_token or x_model_upload_token != expected:
        raise HTTPException(status_code=401, detail="Invalid upload token")


@router.post("/models/upload")
async def upload_model(
    http_request: Request,
    file: UploadFile = File(...),
    model_id: str = Form(...),
    user_id: str | None = Form(default=None),
    x_model_upload_token: str | None = Header(default=None),
):
    """
    Accept a model artifact upload from a remote worker (e.g. Modal GPU worker).

    Security: requires header `X-Model-Upload-Token` matching `MODEL_UPLOAD_TOKEN`.
    """
    _require_upload_token(x_model_upload_token)

    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_path = MODELS_DIR / f"{model_id}.pth"

    # Stream to a temp file first (avoid partial writes to final path).
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        prefix=f"{model_id}.", suffix=".pth.upload", dir=str(MODELS_DIR)
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    try:
        size = 0
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                size += len(chunk)

        with models_dir_lock(MODELS_DIR):
            # Enforce per-user local model limit by evicting oldest models for the user.
            # This keeps at most `MAX_LOCAL_MODELS` per user (default: 2).
            evicted: list[str] = evict_old_models_for_user(
                models_dir=MODELS_DIR, user_id=user_id, max_models=max(MAX_LOCAL_MODELS - 1, 0)
            )

            # Replace existing (best-effort) and write metadata.
            try:
                if final_path.exists():
                    delete_local_model(models_dir=MODELS_DIR, model_id=model_id)
            except Exception:
                pass
            tmp_path.replace(final_path)
            write_model_meta(
                models_dir=MODELS_DIR,
                model_id=model_id,
                user_id=user_id,
                model_size_mb=size / (1024 * 1024),
                source="remote_upload",
            )

        return {"status": "stored", "model_id": model_id, "size_mb": size / (1024 * 1024), "evicted": evicted}
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


@router.get("/models/{model_id}")
async def download_model(model_id: str):
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    file_path = MODELS_DIR / f"{model_id}.pth"
    if file_path.exists():
        return FileResponse(
            path=file_path,
            filename=f"pyreflect_model_{model_id[:8]}.pth",
            media_type="application/octet-stream",
        )

    if HF_REPO_ID:
        hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{model_id}.pth"
        return RedirectResponse(url=hf_url)

    raise HTTPException(status_code=404, detail="Model not found locally or on Hugging Face")


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    file_path = MODELS_DIR / f"{model_id}.pth"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found locally")

    try:
        delete_local_model(models_dir=MODELS_DIR, model_id=model_id)
        return {"status": "deleted", "model_id": model_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {exc}")


@router.get("/models/{model_id}/info")
async def get_model_info(model_id: str, http_request: Request):
    local_path = MODELS_DIR / f"{model_id}.pth"
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        return {"size_mb": size_mb, "source": "local"}

    hf = getattr(http_request.app.state, "hf", None)
    if hf and hf.repo_id:
        return get_remote_model_info(hf, model_id)

    return {"size_mb": None, "source": "unknown"}
