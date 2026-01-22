from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from ..config import HF_REPO_ID, MODELS_DIR
from ..integrations.huggingface import delete_model_file
from ..schemas import SaveResultRequest

router = APIRouter()

class RenameHistoryRequest(BaseModel):
    name: str | None = None


def _get_generations_collection(http_request: Request):
    mongo = getattr(http_request.app.state, "mongo", None)
    return getattr(mongo, "generations", None) if mongo else None


@router.post("/history")
async def save_history(
    request: SaveResultRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    generations = _get_generations_collection(http_request)
    if generations is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        doc = {
            "user_id": x_user_id,
            "created_at": datetime.now(timezone.utc),
            "name": request.name,
            "params": {
                "layers": [layer.model_dump() for layer in request.layers],
                "generator": request.generator.model_dump(),
                "training": request.training.model_dump(),
            },
            "result": request.result,
        }
        result = generations.insert_one(doc)
        return {"success": True, "id": str(result.inserted_id)}
    except Exception as exc:
        print(f"Error saving history: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save: {exc}")


@router.get("/history")
async def get_history(http_request: Request, x_user_id: str | None = Header(default=None)):
    generations = _get_generations_collection(http_request)
    if generations is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    cursor = (
        generations.find(
            {"user_id": x_user_id},
            {
                "result.nr": 0,
                "result.sld": 0,
                "result.training": 0,
                "result.chi": 0,
            },
        )
        .sort("created_at", -1)
        .limit(50)
    )

    local_model_ids = {p.stem for p in MODELS_DIR.glob("*.pth")}
    history: list[dict] = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        model_id = doc.get("result", {}).get("model_id")
        doc["is_local"] = model_id in local_model_ids
        if model_id and HF_REPO_ID:
            doc["hf_url"] = (
                f"https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/models/{model_id}"
            )
        history.append(doc)
    return history


@router.get("/history/{save_id}")
async def get_save(save_id: str, http_request: Request, x_user_id: str | None = Header(default=None)):
    generations = _get_generations_collection(http_request)
    if generations is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    from bson import ObjectId

    try:
        oid = ObjectId(save_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ID format")

    doc = generations.find_one({"_id": oid, "user_id": x_user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Save not found")
    doc["_id"] = str(doc["_id"])
    return doc


@router.patch("/history/{save_id}")
async def rename_save(
    save_id: str,
    request: RenameHistoryRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    generations = _get_generations_collection(http_request)
    if generations is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    from bson import ObjectId

    if not ObjectId.is_valid(save_id):
        raise HTTPException(status_code=400, detail="Invalid ID format")

    name = request.name
    if name is not None:
        name = name.strip()
        if name == "":
            name = None
        if name is not None and len(name) > 80:
            raise HTTPException(status_code=400, detail="Name too long (max 80 chars)")

    oid = ObjectId(save_id)
    result = generations.update_one(
        {"_id": oid, "user_id": x_user_id},
        {"$set": {"name": name, "updated_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Save not found or access denied")

    return {"success": True, "id": save_id, "name": name}


@router.delete("/history/{save_id}")
async def delete_save(save_id: str, http_request: Request, x_user_id: str | None = Header(default=None)):
    generations = _get_generations_collection(http_request)
    if generations is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    from bson import ObjectId

    if not ObjectId.is_valid(save_id):
        raise HTTPException(status_code=400, detail="Invalid ID format")

    try:
        oid = ObjectId(save_id)
        doc = generations.find_one({"_id": oid, "user_id": x_user_id})

        if doc and "result" in doc and "model_id" in doc["result"]:
            model_id = doc["result"]["model_id"]
            if model_id:
                try:
                    local_path = MODELS_DIR / f"{model_id}.pth"
                    if local_path.exists():
                        local_path.unlink()
                        print(f"Deleted orphan model file: {model_id}.pth")
                except Exception as exc:
                    print(f"Warning: Failed to delete model file: {exc}")

                hf = getattr(http_request.app.state, "hf", None)
                if hf and hf.repo_id:
                    deleted = delete_model_file(hf, model_id)
                    if deleted:
                        print(f"Deleted HF model file: {model_id}.pth")

        result = generations.delete_one({"_id": oid, "user_id": x_user_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Save not found or access denied")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error deleting save: {exc}")
        raise HTTPException(status_code=500, detail="Failed to delete save")
