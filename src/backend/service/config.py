from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# =====================
# Paths
# =====================

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Load env from the backend directory regardless of current working directory.
# (Starting `uvicorn` from a different CWD would otherwise skip `src/backend/.env`.)
load_dotenv(dotenv_path=BACKEND_ROOT / ".env")
# Optional fallback to CWD `.env` (won't override existing env vars).
load_dotenv()

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


def _parse_typed_env(env_key: str, default: int | float) -> int | float:
    env_val = os.getenv(env_key)
    if env_val is None:
        return default
    return float(env_val) if isinstance(default, float) else int(env_val)


# Always-available limit tables.
LOCAL_LIMITS: dict[str, int | float] = {
    k: local for k, (local, _prod) in _DEFAULT_LIMITS.items()
}
PRODUCTION_LIMITS: dict[str, int | float] = {
    k: _parse_typed_env(k.upper(), prod)
    for k, (_local, prod) in _DEFAULT_LIMITS.items()
}


# Effective limits for the current environment.
LIMITS: dict[str, int | float] = PRODUCTION_LIMITS if IS_PRODUCTION else LOCAL_LIMITS


def _parse_csv_env(name: str) -> list[str]:
    raw = os.getenv(name) or ""
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


# Comma-separated list of user IDs that should receive local (unlocked) limits
# even in production.
#
# NOTE: This is only as trustworthy as the `X-User-ID` header provided to the
# backend. In a production deployment, ensure that the backend is not directly
# reachable by untrusted clients (or add a proper auth layer), otherwise the
# header can be spoofed.
LIMITS_WHITELIST_USER_IDS: list[str] = _parse_csv_env("LIMITS_WHITELIST_USER_IDS")

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
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://pyreflect.shlawg.com",
).split(",")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS if o and o.strip()]

# Optional: regex-based CORS allowlist (useful for subdomains like https://*.shlawg.com).
# If set, FastAPI will allow origins matching the regex in addition to `CORS_ORIGINS`.
# Default: allow all *.shlawg.com subdomains for production deployment flexibility.
CORS_ALLOW_ORIGIN_REGEX = (
    os.getenv("CORS_ALLOW_ORIGIN_REGEX") or r"https://.*\.shlawg\.com"
).strip() or None


# =====================
# Training Constants
# =====================

LEARNING_RATE = 2.15481e-05
WEIGHT_DECAY = 2.6324e-05
SPLIT_RATIO = 0.8

# =====================
# Local Model Storage
# =====================

# Max number of locally stored model files per user (MODELS_DIR/*.pth).
# When reached, new training runs will evict the oldest models for that user.
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

# Stale job cleanup: if a job's meta.updated_at is older than this, consider it stale.
# This handles zombie jobs where Modal worker dies without cleanup.
# Default: 10 minutes (600 seconds). Workers update meta every ~1 second.
STALE_JOB_THRESHOLD_S = int(os.getenv("STALE_JOB_THRESHOLD_S", "600"))

# How often to run the background cleanup task (seconds).
STALE_JOB_CLEANUP_INTERVAL_S = int(os.getenv("STALE_JOB_CLEANUP_INTERVAL_S", "60"))


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_str_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _squash_whitespace(value: str | None) -> str | None:
    """
    Remove any whitespace characters from a string.

    This hardens `.env` copy/paste issues where values get wrapped across lines
    (e.g. Modal endpoint URLs).
    """
    if not value:
        return None
    cleaned = "".join(value.split())
    return cleaned or None


# Whether the API process should start a local RQ worker subprocess.
# - Default: enabled for local dev, disabled for PRODUCTION (so you can use Modal workers).
START_LOCAL_RQ_WORKER = _get_bool_env(
    "START_LOCAL_RQ_WORKER", default=not IS_PRODUCTION
)

# =====================
# Modal (Remote GPU Workers)
# =====================

# When using remote workers (START_LOCAL_RQ_WORKER=false), the backend can trigger
# a Modal worker immediately after enqueuing a job (on-demand, no schedule required).
MODAL_INSTANT_SPAWN = _get_bool_env("MODAL_INSTANT_SPAWN", default=True)
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "pyreflect-worker")
MODAL_FUNCTION_NAME = os.getenv("MODAL_FUNCTION_NAME", "poll_queue")
MODAL_SPAWN_LOCK_TTL_S = int(os.getenv("MODAL_SPAWN_LOCK_TTL_S", "900"))

# Optional: if you don't want the backend to depend on Modal auth, deploy the worker
# and set this to the `poll_queue` web endpoint URL (Modal provides it on deploy).
MODAL_POLL_URL = _squash_whitespace(_get_str_env("MODAL_POLL_URL"))
# Optional shared secret to protect the poll endpoint (sent as `?token=...`).
MODAL_TRIGGER_TOKEN = _get_str_env("MODAL_TRIGGER_TOKEN")

# Optional admin token for debug endpoints.
ADMIN_TOKEN = _get_str_env("ADMIN_TOKEN")

# =====================
# Checkpointing
# =====================

# How often to save checkpoints during training (epochs). Set to 0 to disable.
CHECKPOINT_EVERY_N_EPOCHS = int(os.getenv("CHECKPOINT_EVERY_N_EPOCHS", "5"))

# Whether to save checkpoint on best validation loss (in addition to periodic).
CHECKPOINT_BEST_ONLY = _get_bool_env("CHECKPOINT_BEST_ONLY", default=False)

# Separate HuggingFace repo for checkpoints (e.g., "org/checkpoints").
# If not set, checkpoints will be stored in a "checkpoints/" subfolder of HF_REPO_ID.
HF_CHECKPOINT_REPO_ID = _get_str_env("HF_CHECKPOINT_REPO_ID")
