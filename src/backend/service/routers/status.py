from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ..config import CURVES_DIR, DATA_DIR, EXPT_DIR, MODELS_DIR, SETTINGS_PATH
from ..services.pyreflect_runtime import PYREFLECT
from ..settings_store import load_settings, resolve_setting_path

router = APIRouter()


@router.get("/status")
async def get_status():
    def list_files(directory: Path, extensions: tuple[str, ...]) -> list[str]:
        if not directory.exists():
            return []
        return [
            f.name for f in directory.iterdir() if f.is_file() and f.suffix in extensions
        ]

    data_files = list_files(DATA_DIR, (".npy", ".pth", ".pt"))
    curve_files = list_files(CURVES_DIR, (".npy",))
    expt_files = list_files(EXPT_DIR, (".npy",))
    model_files = list_files(MODELS_DIR, (".pth", ".pt"))
    has_settings = SETTINGS_PATH.exists()
    settings = load_settings() if has_settings else {}
    settings_paths = {
        "nr_predict_sld": {
            "file": settings.get("nr_predict_sld", {}).get("file", {}),
            "models": {
                "model": settings.get("nr_predict_sld", {}).get("models", {}).get("model"),
                "normalization_stats": settings.get("nr_predict_sld", {})
                .get("models", {})
                .get("normalization_stats"),
            },
        },
        "sld_predict_chi": {"file": settings.get("sld_predict_chi", {}).get("file", {})},
    }
    settings_status = {
        "nr_predict_sld": {"file": {}, "models": {}},
        "sld_predict_chi": {"file": {}},
    }
    for group, values in settings_paths.get("nr_predict_sld", {}).get("file", {}).items():
        resolved = resolve_setting_path(values)
        settings_status["nr_predict_sld"]["file"][group] = bool(resolved and resolved.exists())
    for group, values in settings_paths.get("nr_predict_sld", {}).get("models", {}).items():
        resolved = resolve_setting_path(values)
        settings_status["nr_predict_sld"]["models"][group] = bool(resolved and resolved.exists())
    for group, values in settings_paths.get("sld_predict_chi", {}).get("file", {}).items():
        resolved = resolve_setting_path(values)
        settings_status["sld_predict_chi"]["file"][group] = bool(resolved and resolved.exists())

    return {
        "pyreflect_available": PYREFLECT.available,
        "has_settings": has_settings,
        "data_files": data_files,
        "curve_files": curve_files,
        "expt_files": expt_files,
        "model_files": model_files,
        "settings_paths": settings_paths,
        "settings_status": settings_status,
    }

