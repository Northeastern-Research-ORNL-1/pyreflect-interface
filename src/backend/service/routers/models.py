from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi import File, Form, Header, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from ..config import HF_REPO_ID, IS_PRODUCTION, MAX_LOCAL_MODELS, MODELS_DIR
from ..integrations.huggingface import get_remote_model_info
from ..services.local_model_limit import (
    delete_local_model,
    evict_old_models_for_user,
    models_dir_lock,
    write_model_meta,
)
from ..services.guards import require_user_id

router = APIRouter()


def _require_user_id(x_user_id: str | None) -> None:
    require_user_id(is_production=IS_PRODUCTION, x_user_id=x_user_id)


def _get_generations_collection(http_request: Request) -> Any | None:
    """Get MongoDB generations collection from app state."""
    mongo = getattr(http_request.app.state, "mongo", None)
    if mongo and getattr(mongo, "available", False):
        return getattr(mongo, "generations", None)
    return None


def _model_meta_user_id(model_id: str) -> str | None:
    meta_path = MODELS_DIR / f"{model_id}.meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    user_id = meta.get("user_id") if isinstance(meta, dict) else None
    return user_id if isinstance(user_id, str) else None


def _check_mongo_model_ownership(generations: Any, model_id: str, user_id: str) -> bool:
    """Check if user owns model according to MongoDB generations collection."""
    if generations is None:
        return False
    try:
        doc = generations.find_one(
            {"result.model_id": model_id, "user_id": user_id},
            {"_id": 1},
        )
        return doc is not None
    except Exception as exc:
        print(f"Warning: MongoDB ownership check failed: {exc}")
        return False


def _require_model_access(
    model_id: str,
    x_user_id: str | None,
    *,
    http_request: Request | None = None,
    allow_hf_fallback: bool = False,
) -> None:
    """
    Verify model access permissions.
    
    Checks ownership in order:
    1. Local metadata file (fast path for locally stored models)
    2. MongoDB generations collection (for HF-only models from Modal GPU)
    3. HF fallback (if enabled and HF is configured)
    
    Args:
        model_id: The model ID to check
        x_user_id: The requesting user's ID
        http_request: FastAPI request (needed for MongoDB access)
        allow_hf_fallback: If True, allow access when model doesn't exist locally but HF is configured.
    """
    _require_user_id(x_user_id)
    
    # Check 1: Local metadata file
    owner = _model_meta_user_id(model_id)
    local_path = MODELS_DIR / f"{model_id}.pth"
    local_exists = local_path.exists()

    # If local owner metadata exists, verify ownership
    if owner is not None:
        if owner != x_user_id:
            raise HTTPException(status_code=403, detail="Model access denied")
        return  # Owner matches, access granted

    # Check 2: MongoDB ownership (for HF-only models from Modal GPU)
    if http_request is not None and x_user_id:
        generations = _get_generations_collection(http_request)
        if _check_mongo_model_ownership(generations, model_id, x_user_id):
            return  # User owns this model according to MongoDB

    # No local or MongoDB ownership found...
    if not IS_PRODUCTION:
        return  # In dev, allow access
    
    # In production with HF fallback enabled and HF configured:
    # Allow access if the model doesn't exist locally (will check HF instead)
    if allow_hf_fallback and HF_REPO_ID and not local_exists:
        return
    
    # Deny access
    if local_exists:
        raise HTTPException(status_code=403, detail="Model access denied (no owner metadata)")
    else:
        raise HTTPException(status_code=403, detail="Model access denied")

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
async def download_model(
    model_id: str,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    # Allow HF fallback - models stored on HF won't have local metadata
    _require_model_access(model_id, x_user_id, http_request=http_request, allow_hf_fallback=True)

    file_path = MODELS_DIR / f"{model_id}.pth"
    if file_path.exists():
        return FileResponse(
            path=file_path,
            filename=f"pyreflect_model_{model_id[:8]}.pth",
            media_type="application/octet-stream",
        )

    if HF_REPO_ID:
        hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/models/{model_id}/{model_id}.pth"
        return RedirectResponse(url=hf_url)

    raise HTTPException(status_code=404, detail="Model not found locally or on Hugging Face")


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    _require_model_access(model_id, x_user_id, http_request=http_request)

    file_path = MODELS_DIR / f"{model_id}.pth"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found locally")

    try:
        delete_local_model(models_dir=MODELS_DIR, model_id=model_id)
        return {"status": "deleted", "model_id": model_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {exc}")


@router.get("/models/{model_id}/info")
async def get_model_info(
    model_id: str,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    # Check ownership via local metadata or MongoDB (for HF-only models from Modal GPU)
    _require_model_access(model_id, x_user_id, http_request=http_request, allow_hf_fallback=True)
    
    local_path = MODELS_DIR / f"{model_id}.pth"
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        return {"size_mb": size_mb, "source": "local"}

    hf = getattr(http_request.app.state, "hf", None)
    if hf and hf.repo_id:
        return get_remote_model_info(hf, model_id)

    return {"size_mb": None, "source": "unknown"}


@router.get("/models/{model_id}/training-data/{file_type}")
async def get_training_data(
    model_id: str,
    file_type: str,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    """Download training data (.npy files) for a model from HuggingFace."""
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")

    if file_type not in ("nr_train", "sld_train"):
        raise HTTPException(status_code=400, detail="Invalid file type. Use 'nr_train' or 'sld_train'")

    _require_model_access(model_id, x_user_id, http_request=http_request, allow_hf_fallback=True)

    if HF_REPO_ID:
        # Map file_type to HF filenames (nr_train -> nr_train.npy, sld_train -> sld_train.npy)
        file_name = f"{file_type}.npy"
        hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/models/{model_id}/{file_name}"
        return RedirectResponse(url=hf_url)

    raise HTTPException(status_code=404, detail="Training data not available")
