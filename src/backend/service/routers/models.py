from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from ..config import HF_REPO_ID, MODELS_DIR
from ..integrations.huggingface import get_remote_model_info

router = APIRouter()


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
        file_path.unlink()
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

