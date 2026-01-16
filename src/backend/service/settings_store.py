from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import (
    BACKEND_ROOT,
    CURVES_DIR,
    DATA_DIR,
    EXPT_DIR,
    MODELS_DIR,
    SETTINGS_PATH,
)

DEFAULT_SETTINGS_YAML = """\
# Configuration file for PyReflect GUI backend
root: .

### SLD-Chi settings ###
sld_predict_chi:
  file:
    model_experimental_sld_profile: data/mod_expt.npy
    model_sld_file: data/mod_sld_fp49.npy
    model_chi_params_file: data/mod_params_fp49.npy
  models:
    latent_dim: 2
    batch_size: 16
    ae_epochs: 20
    mlp_epochs: 20

### NR-SLD profile settings ###
nr_predict_sld:
  file:
    nr_train: data/curves/X_train_5_layers.npy
    sld_train: data/curves/y_train_5_layers.npy
    experimental_nr_file: data/combined_expt_denoised_nr.npy
  models:
    model: data/trained_nr_sld_model_no_dropout.pth
    num_film_layers: 5
    num_curves: 1000
    epochs: 3
    batch_size: 32
    layers: 12
    dropout: 0.0
    normalization_stats: data/normalization_stat.npy
"""


def ensure_backend_layout() -> None:
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPT_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(DEFAULT_SETTINGS_YAML, encoding="utf-8")


UPLOAD_ROLE_SETTINGS_MAP: dict[str, tuple[str, str, str]] = {
    "nr_train": ("nr_predict_sld", "file", "nr_train"),
    "sld_train": ("nr_predict_sld", "file", "sld_train"),
    "experimental_nr": ("nr_predict_sld", "file", "experimental_nr_file"),
    "normalization_stats": ("nr_predict_sld", "models", "normalization_stats"),
    "nr_sld_model": ("nr_predict_sld", "models", "model"),
    "sld_chi_experimental_profile": ("sld_predict_chi", "file", "model_experimental_sld_profile"),
    "sld_chi_model_sld_file": ("sld_predict_chi", "file", "model_sld_file"),
    "sld_chi_model_chi_params_file": ("sld_predict_chi", "file", "model_chi_params_file"),
}


def ensure_settings_structure(settings: dict | None) -> dict:
    settings = settings or {}
    settings.setdefault("sld_predict_chi", {})
    settings["sld_predict_chi"].setdefault("file", {})
    settings["sld_predict_chi"].setdefault("models", {})
    settings.setdefault("nr_predict_sld", {})
    settings["nr_predict_sld"].setdefault("file", {})
    settings["nr_predict_sld"].setdefault("models", {})
    return settings


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return ensure_settings_structure(
                yaml.safe_load(SETTINGS_PATH.read_text()) or {}
            )
        except Exception as exc:
            print(f"Warning: Failed to parse settings.yml ({exc}). Using defaults.")
    return ensure_settings_structure(yaml.safe_load(DEFAULT_SETTINGS_YAML) or {})


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(
        yaml.safe_dump(settings, sort_keys=False),
        encoding="utf-8",
    )


def to_settings_path(path: Path) -> str:
    return path.relative_to(BACKEND_ROOT).as_posix()


def resolve_setting_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    raw = Path(path_value)
    return raw if raw.is_absolute() else BACKEND_ROOT / raw


def apply_upload_to_settings(settings: dict, role: str, rel_path: str) -> bool:
    mapping = UPLOAD_ROLE_SETTINGS_MAP.get(role)
    if not mapping:
        return False
    section, group, key = mapping
    settings.setdefault(section, {}).setdefault(group, {})[key] = rel_path
    return True


def validate_npy_payload(role: str, payload: Any) -> dict:
    if role == "normalization_stats":
        if isinstance(payload, np.ndarray):
            payload = payload.item() if payload.dtype == object else payload
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            raise ValueError("normalization_stats must be a dict with 'x' and 'y' keys")
        return {"type": "dict"}

    if not isinstance(payload, np.ndarray):
        raise ValueError("Expected a numpy array")

    if role == "sld_chi_model_chi_params_file":
        if payload.ndim != 2:
            raise ValueError("Chi params file must be 2D (N, num_params)")
        return {"shape": payload.shape}

    if payload.ndim == 2 and role in {"sld_chi_experimental_profile", "sld_chi_model_sld_file"}:
        if payload.shape[0] != 2:
            raise ValueError("SLD curve must be (2, L) when 2D")
        return {"shape": payload.shape}

    if payload.ndim != 3 or payload.shape[1] != 2:
        raise ValueError("Curve data must have shape (N, 2, L)")

    return {"shape": payload.shape}


def upload_role_storage_target(role: str | None, filename: str) -> Path:
    """Decide where to store an uploaded file based on role and extension."""
    if filename.endswith((".yml", ".yaml")) and filename.startswith("settings"):
        return SETTINGS_PATH
    if role == "experimental_nr":
        return EXPT_DIR / filename
    if role in {
        "sld_chi_experimental_profile",
        "sld_chi_model_sld_file",
        "sld_chi_model_chi_params_file",
        "normalization_stats",
    }:
        return DATA_DIR / filename
    if role == "nr_sld_model" or filename.endswith((".pth", ".pt")):
        return MODELS_DIR / filename
    return CURVES_DIR / filename
