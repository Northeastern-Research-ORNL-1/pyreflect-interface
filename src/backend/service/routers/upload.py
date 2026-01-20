from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile

from ..config import IS_PRODUCTION
from ..services.guards import require_user_id
from ..settings_store import (
    UPLOAD_ROLE_SETTINGS_MAP,
    apply_upload_to_settings,
    load_settings,
    save_settings,
    to_settings_path,
    upload_role_storage_target,
    validate_npy_payload,
)
from ..services.rate_limit import limit_upload

router = APIRouter()


@router.post("/upload")
async def upload_files(
    http_request: Request,
    files: list[UploadFile] = File(...),
    roles: list[str] | None = Form(default=None),
    x_user_id: str | None = Header(default=None),
):
    require_user_id(is_production=IS_PRODUCTION, x_user_id=x_user_id)

    limit_upload(http_request, x_user_id)

    saved: list[str] = []
    metadata: list[dict[str, Any]] = []
    settings = load_settings()
    updated_settings = False

    def load_normalization_stats(path: Path) -> dict[str, Any]:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "x" not in data or "y" not in data:
                    raise ValueError("normalization_stats .npz must contain arrays 'x' and 'y'")
                x = np.asarray(data["x"])  # type: ignore[index]
                y = np.asarray(data["y"])  # type: ignore[index]
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
                raise ValueError("normalization_stats .json must be a dict with 'x' and 'y' keys")
            x = np.asarray(payload.get("x"))
            y = np.asarray(payload.get("y"))
        else:
            raise ValueError("normalization_stats must be uploaded as .npz or .json")

        if x.dtype.kind not in {"i", "u", "f"} or y.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("normalization_stats x/y must be numeric arrays")
        return {"x": x, "y": y}

    for index, uploaded in enumerate(files):
        if not uploaded.filename:
            continue
        filename = Path(uploaded.filename).name
        role = roles[index] if roles and index < len(roles) else None
        is_settings_file = filename.endswith((".yml", ".yaml")) and filename.startswith("settings")

        if role in (None, "", "auto"):
            if not is_settings_file:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename}: missing upload role. Provide an explicit role (no filename-based inference).",
                )
            resolved_role = None
        else:
            resolved_role = role
            if resolved_role not in UPLOAD_ROLE_SETTINGS_MAP:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename}: unknown upload role '{resolved_role}'.",
                )

        target_path = upload_role_storage_target(resolved_role, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as buffer:
            while True:
                chunk = await uploaded.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        file_meta: dict[str, Any] = {"filename": filename, "role": resolved_role or role}
        if resolved_role:
            try:
                if resolved_role == "normalization_stats":
                    if target_path.suffix == ".npy":
                        raise ValueError("normalization_stats .npy uploads are not allowed (use .npz or .json)")

                    payload = load_normalization_stats(target_path)
                    # Store in the legacy .npy format expected by pyreflect, but generated
                    # by the backend (so we never unpickle untrusted input).
                    normalized_path = target_path.with_suffix(".npy")
                    np.save(normalized_path, payload)
                    try:
                        target_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    target_path = normalized_path
                    file_meta["stored_as"] = target_path.name
                elif target_path.suffix == ".npy":
                    # Never allow pickles from untrusted uploads.
                    payload = np.load(target_path, allow_pickle=False)
                else:
                    payload = None

                if payload is not None:
                    file_meta.update(validate_npy_payload(resolved_role, payload))
            except Exception as exc:
                target_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"{filename}: {exc}") from exc

        rel_path = to_settings_path(target_path)
        if resolved_role:
            updated_settings |= apply_upload_to_settings(settings, resolved_role, rel_path)

        saved.append(str(target_path))
        metadata.append({**file_meta, "path": rel_path})

    if updated_settings:
        save_settings(settings)

    return {"saved": saved, "metadata": metadata, "settings_updated": updated_settings}
