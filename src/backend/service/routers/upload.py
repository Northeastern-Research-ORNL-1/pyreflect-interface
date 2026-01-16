from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..settings_store import (
    UPLOAD_ROLE_SETTINGS_MAP,
    apply_upload_to_settings,
    load_settings,
    save_settings,
    to_settings_path,
    upload_role_storage_target,
    validate_npy_payload,
)

router = APIRouter()


@router.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    roles: list[str] | None = Form(default=None),
):
    saved: list[str] = []
    metadata: list[dict[str, Any]] = []
    settings = load_settings()
    updated_settings = False

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
        if target_path.suffix == ".npy" and resolved_role:
            try:
                payload = np.load(target_path, allow_pickle=True)
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
