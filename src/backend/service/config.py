from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =====================
# Paths
# =====================

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"
CURVES_DIR = DATA_DIR / "curves"
MODELS_DIR = DATA_DIR / "models"
EXPT_DIR = CURVES_DIR / "expt"
SETTINGS_PATH = BACKEND_ROOT / "settings.yml"

# =====================
# Environment / Limits
# =====================

IS_PRODUCTION = os.getenv("PRODUCTION", "").lower() in ("true", "1", "yes")

_DEFAULT_LIMITS: dict[str, tuple[int | float, int | float]] = {
    "max_curves": (100000, 5000),
    "max_film_layers": (20, 10),
    "max_batch_size": (512, 64),
    "max_epochs": (1000, 50),
    "max_cnn_layers": (20, 12),
    "max_dropout": (0.9, 0.5),
    "max_latent_dim": (128, 32),
    "max_ae_epochs": (500, 100),
    "max_mlp_epochs": (500, 100),
}


def _get_limit(key: str, local_val: int | float, prod_val: int | float) -> int | float:
    if not IS_PRODUCTION:
        return local_val
    env_key = key.upper()
    env_val = os.getenv(env_key)
    if env_val is not None:
        return float(env_val) if isinstance(prod_val, float) else int(env_val)
    return prod_val


LIMITS: dict[str, int | float] = {
    key: _get_limit(key, local, prod) for key, (local, prod) in _DEFAULT_LIMITS.items()
}

# =====================
# Integrations
# =====================

MONGODB_URI = os.getenv("MONGODB_URI")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

# =====================
# CORS
# =====================

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# =====================
# Training Constants
# =====================

LEARNING_RATE = 2.15481e-05
WEIGHT_DECAY = 2.6324e-05
SPLIT_RATIO = 0.8

# =====================
# Local Model Storage
# =====================

# Max number of locally stored model files (MODELS_DIR/*.pth). When reached, new
# training runs will wait until a slot is freed (e.g. after HF upload cleanup or
# manual deletion).
MAX_LOCAL_MODELS = int(os.getenv("MAX_LOCAL_MODELS", "2"))

# How long to wait for a local model slot before failing (seconds). Set to 0 to
# wait indefinitely.
LOCAL_MODEL_WAIT_TIMEOUT_S = float(os.getenv("LOCAL_MODEL_WAIT_TIMEOUT_S", "900"))

# Poll interval while waiting for a slot (seconds).
LOCAL_MODEL_WAIT_POLL_S = float(os.getenv("LOCAL_MODEL_WAIT_POLL_S", "2.0"))

# =====================
# RQ (Job Queue)
# =====================

# Training job timeout for RQ (supports strings like "30m", "2h", or seconds).
RQ_JOB_TIMEOUT = os.getenv("RQ_JOB_TIMEOUT", "2h")


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# Whether the API process should start a local RQ worker subprocess.
# - Default: enabled for local dev, disabled for PRODUCTION (so you can use Modal workers).
START_LOCAL_RQ_WORKER = _get_bool_env("START_LOCAL_RQ_WORKER", default=not IS_PRODUCTION)
