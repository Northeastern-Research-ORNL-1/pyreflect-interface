from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile

from .. import config as cfg
from ..config import IS_PRODUCTION
from ..integrations.huggingface import HuggingFaceIntegration, upload_artifact_file
from ..services.guards import require_user_id
from ..services.csv_parser import parse_csv_file
from ..services.upload_canonicalization import (
    canonicalize_npy_payload,
    load_normalization_stats,
)
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
    hf: HuggingFaceIntegration | None = getattr(http_request.app.state, "hf", None)
    hf_available = bool(hf and hf.available)

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

        upload_id = uuid4().hex
        target_path = upload_role_storage_target(resolved_role, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as buffer:
            while True:
                chunk = await uploaded.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        file_meta: dict[str, Any] = {"filename": filename, "role": resolved_role or role}
        report_payload: dict[str, Any] | None = None
        raw_snapshot_path: Path | None = None

        def ensure_raw_snapshot(path: Path) -> Path:
            nonlocal raw_snapshot_path
            if raw_snapshot_path is None:
                snapshot_name = f".{path.name}.{upload_id}.raw"
                raw_snapshot_path = path.with_name(snapshot_name)
                shutil.copyfile(path, raw_snapshot_path)
            return raw_snapshot_path

        if resolved_role:
            try:
                if resolved_role == "normalization_stats":
                    if hf_available:
                        ensure_raw_snapshot(target_path)
                    norm_result = load_normalization_stats(target_path)
                    payload = norm_result.payload
                    normalized_path = target_path.with_suffix(".npy")
                    np.save(normalized_path, payload)
                    try:
                        if normalized_path != target_path:
                            target_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    target_path = normalized_path
                    file_meta["stored_as"] = target_path.name
                    file_meta.update(norm_result.metadata)
                    report_payload = norm_result.report
                elif target_path.suffix == ".npy":
                    payload = np.load(target_path, allow_pickle=False)
                    canonical_result = canonicalize_npy_payload(resolved_role, payload)
                    payload = canonical_result.payload
                    file_meta.update(canonical_result.metadata)
                    report_payload = canonical_result.report
                    if canonical_result.metadata.get("canonicalized"):
                        if hf_available:
                            ensure_raw_snapshot(target_path)
                        np.save(target_path, payload)
                elif target_path.suffix.lower() in (
                    ".csv",
                    ".txt",
                    ".dat",
                ) and resolved_role in {
                    "experimental_nr",
                    "nr_train",
                    "sld_train",
                }:
                    csv_result = parse_csv_file(target_path)
                    payload = csv_result.payload
                    file_meta.update(csv_result.metadata)
                    report_payload = csv_result.report

                    if hf_available:
                        ensure_raw_snapshot(target_path)

                    npy_path = target_path.with_suffix(".npy")
                    np.save(npy_path, payload)
                    try:
                        if npy_path != target_path:
                            target_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    target_path = npy_path
                    file_meta["stored_as"] = target_path.name

                    canonical_result = canonicalize_npy_payload(
                        resolved_role, payload
                    )
                    payload = canonical_result.payload
                    file_meta.update(canonical_result.metadata)
                    if canonical_result.report:
                        report_payload = {
                            **report_payload,
                            "canonicalization": canonical_result.report,
                        }
                    if canonical_result.metadata.get("canonicalized"):
                        np.save(target_path, payload)
                else:
                    payload = None

                if payload is not None:
                    file_meta.update(validate_npy_payload(resolved_role, payload))
            except Exception as exc:
                target_path.unlink(missing_ok=True)
                if raw_snapshot_path:
                    raw_snapshot_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"{filename}: {exc}") from exc

        rel_path = to_settings_path(target_path)
        if resolved_role:
            updated_settings |= apply_upload_to_settings(settings, resolved_role, rel_path)

        if resolved_role:
            report_dir = cfg.DATA_DIR / "upload_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_name = f"{upload_id}_{resolved_role}.json"
            report_path = report_dir / report_name

            if report_payload is None:
                report_payload = {
                    "role": resolved_role,
                    "canonicalized": False,
                    "actions": [],
                    "warnings": [],
                }
            report_payload = {
                **report_payload,
                "upload_id": upload_id,
                "filename": filename,
                "stored_path": rel_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
            file_meta["report_path_local"] = to_settings_path(report_path)

            if hf_available and hf is not None:
                actor = x_user_id or "anonymous"
                base = f"uploads/{actor}/{upload_id}/{resolved_role}"
                raw_local = raw_snapshot_path or target_path
                raw_hf_path = f"{base}/raw/{Path(filename).name}"
                canonical_hf_path = f"{base}/canonical/{target_path.name}"
                report_hf_path = f"{base}/report.json"
                ok_raw = upload_artifact_file(hf, file_path=raw_local, path_in_repo=raw_hf_path)
                ok_canonical = upload_artifact_file(
                    hf, file_path=target_path, path_in_repo=canonical_hf_path
                )
                ok_report = upload_artifact_file(
                    hf, file_path=report_path, path_in_repo=report_hf_path
                )
                file_meta["report_path_hf"] = report_hf_path if ok_report else None
                file_meta["raw_hf_path"] = raw_hf_path if ok_raw else None
                file_meta["canonical_hf_path"] = canonical_hf_path if ok_canonical else None

        if raw_snapshot_path:
            raw_snapshot_path.unlink(missing_ok=True)

        saved.append(str(target_path))
        metadata.append({**file_meta, "path": rel_path})

    if updated_settings:
        save_settings(settings)

    return {"saved": saved, "metadata": metadata, "settings_updated": updated_settings}
