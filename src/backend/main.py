"""
PyReflect Interface - FastAPI Backend
Provides REST API for generating NR/SLD curves using pyreflect
"""

import os
from dotenv import load_dotenv
import tempfile

# Load .env file if present
load_dotenv()
import time
import warnings
import io
import threading
import queue
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import List, TextIO, cast
from contextlib import asynccontextmanager

import json
from typing import Generator, Any

# =====================
# Production Limits
# =====================
IS_PRODUCTION = os.getenv("PRODUCTION", "").lower() in ("true", "1", "yes")

# Default limits: Local (unlimited) vs Production (strict)
_DEFAULT_LIMITS = {
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

# Build limits with optional env var overrides (only in production)
def _get_limit(key: str, local_val: int | float, prod_val: int | float) -> int | float:
    if not IS_PRODUCTION:
        return local_val  # Always use unlimited defaults in dev mode
    env_key = key.upper()  # e.g., max_curves -> MAX_CURVES
    env_val = os.getenv(env_key)
    if env_val is not None:
        return float(env_val) if isinstance(prod_val, float) else int(env_val)
    return prod_val

LIMITS = {
    key: _get_limit(key, local, prod)
    for key, (local, prod) in _DEFAULT_LIMITS.items()
}

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Backend paths
BACKEND_ROOT = Path(__file__).parent
DATA_DIR = BACKEND_ROOT / "data"
CURVES_DIR = DATA_DIR / "curves"
MODELS_DIR = DATA_DIR / "models"
EXPT_DIR = CURVES_DIR / "expt"
SETTINGS_PATH = BACKEND_ROOT / "settings.yml"

# MongoDB connection (optional - for result persistence)
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_AVAILABLE = False
mongo_client = None
mongo_db = None
generations_collection = None

if MONGODB_URI:
    try:
        from pymongo.mongo_client import MongoClient
        from pymongo.server_api import ServerApi
        mongo_client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
        # Ping to verify connection
        mongo_client.admin.command('ping')
        mongo_db = mongo_client["PyReflect"]
        generations_collection = mongo_db["generations"]
        MONGODB_AVAILABLE = True
        print("MongoDB connected: PyReflect")
    except Exception as e:
        print(f"Warning: MongoDB connection failed: {e}")

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

hf_api = None
if HF_TOKEN and HF_REPO_ID:
    try:
        from huggingface_hub import HfApi
        hf_api = HfApi(token=HF_TOKEN)
        print(f"Hugging Face API initialized (Repo: {HF_REPO_ID})")
    except Exception as e:
        print(f"Warning: Failed to initialize Hugging Face API: {e}")


def upload_to_hf(file_path: Path, model_id: str) -> bool:
    """Upload model file to Hugging Face Hub."""
    if not hf_api or not HF_REPO_ID:
        return False
    
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {model_id}.pth ({file_size_mb:.2f} MB) to HF Hub...")
        
        # Ensure repo exists (public dataset)
        try:
            hf_api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True, private=False)
        except Exception:
            # Ignore creation error (might already exist or permission issue, let upload handle it)
            pass

        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{model_id}.pth",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        return True
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        return False


# Import pyreflect components
PYREFLECT_AVAILABLE = False
COMPUTE_NR_AVAILABLE = False
compute_nr_from_sld = None
try:
    from pyreflect.input.reflectivity_data_generator import (
        ReflectivityDataGenerator,
        calculate_reflectivity,
    )
    from pyreflect.input.data_processor import DataProcessor
    from pyreflect.models.cnn import CNN
    from pyreflect.config.runtime import DEVICE
    import torch
    PYREFLECT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pyreflect not fully available: {e}")

if PYREFLECT_AVAILABLE:
    try:
        from pyreflect.pipelines import helper as pipelines_helper
        compute_nr_from_sld = pipelines_helper.compute_nr_from_sld
        COMPUTE_NR_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: compute_nr_from_sld not available: {e}")

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
    """Ensure data directories and settings.yml exist."""
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPT_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(DEFAULT_SETTINGS_YAML, encoding="utf-8")


# =====================
# Pydantic Models
# =====================

class FilmLayer(BaseModel):
    """Single film layer definition"""
    name: str = "layer"
    sld: float = Field(ge=0, le=10, description="Scattering Length Density")
    isld: float = Field(ge=0, le=1, default=0, description="Imaginary SLD")
    thickness: float = Field(ge=0, le=1000, default=100, description="Layer thickness in Angstroms")
    roughness: float = Field(ge=0, le=200, default=10, description="Interface roughness in Angstroms")


class GeneratorParams(BaseModel):
    """Parameters for curve generation"""
    numCurves: int = Field(ge=1, le=100000, default=1000)
    numFilmLayers: int = Field(ge=1, le=20, default=5)


class TrainingParams(BaseModel):
    """Parameters for model training"""
    batchSize: int = Field(ge=1, le=512, default=32)
    epochs: int = Field(ge=1, le=1000, default=10)
    layers: int = Field(ge=1, le=20, default=12)
    dropout: float = Field(ge=0, le=0.9, default=0.0)
    latentDim: int = Field(ge=2, le=128, default=16)
    aeEpochs: int = Field(ge=1, le=500, default=50)
    mlpEpochs: int = Field(ge=1, le=500, default=50)


def validate_limits(gen_params: GeneratorParams, train_params: TrainingParams) -> None:
    """Validate request against current limits (enforced in production)."""
    errors = []
    if gen_params.numCurves > LIMITS["max_curves"]:
        errors.append(f"numCurves ({gen_params.numCurves}) exceeds limit ({LIMITS['max_curves']})")
    if gen_params.numFilmLayers > LIMITS["max_film_layers"]:
        errors.append(f"numFilmLayers ({gen_params.numFilmLayers}) exceeds limit ({LIMITS['max_film_layers']})")
    if train_params.batchSize > LIMITS["max_batch_size"]:
        errors.append(f"batchSize ({train_params.batchSize}) exceeds limit ({LIMITS['max_batch_size']})")
    if train_params.epochs > LIMITS["max_epochs"]:
        errors.append(f"epochs ({train_params.epochs}) exceeds limit ({LIMITS['max_epochs']})")
    if train_params.layers > LIMITS["max_cnn_layers"]:
        errors.append(f"layers ({train_params.layers}) exceeds limit ({LIMITS['max_cnn_layers']})")
    if train_params.dropout > LIMITS["max_dropout"]:
        errors.append(f"dropout ({train_params.dropout}) exceeds limit ({LIMITS['max_dropout']})")
    if train_params.latentDim > LIMITS["max_latent_dim"]:
        errors.append(f"latentDim ({train_params.latentDim}) exceeds limit ({LIMITS['max_latent_dim']})")
    if train_params.aeEpochs > LIMITS["max_ae_epochs"]:
        errors.append(f"aeEpochs ({train_params.aeEpochs}) exceeds limit ({LIMITS['max_ae_epochs']})")
    if train_params.mlpEpochs > LIMITS["max_mlp_epochs"]:
        errors.append(f"mlpEpochs ({train_params.mlpEpochs}) exceeds limit ({LIMITS['max_mlp_epochs']})")
    
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))


class GenerateRequest(BaseModel):
    """Request body for generation endpoint"""
    layers: List[FilmLayer]
    generator: GeneratorParams
    training: TrainingParams
    name: str | None = None


class NRData(BaseModel):
    """Neutron Reflectivity data"""
    q: List[float]
    groundTruth: List[float]
    computed: List[float]  # NR computed from predicted SLD


class SLDData(BaseModel):
    """SLD Profile data"""
    z: List[float]
    groundTruth: List[float]
    predicted: List[float]


class TrainingData(BaseModel):
    """Training loss history"""
    epochs: List[int]
    trainingLoss: List[float]
    validationLoss: List[float]


class ChiDataPoint(BaseModel):
    """Chi parameter comparison point"""
    x: int
    predicted: float
    actual: float


class Metrics(BaseModel):
    """Model evaluation metrics"""
    mse: float
    r2: float
    mae: float


class GenerateResponse(BaseModel):
    """Response from generation endpoint"""
    nr: NRData
    sld: SLDData
    training: TrainingData
    chi: List[ChiDataPoint]
    metrics: Metrics
    model_id: str | None = None
    model_size_mb: float | None = None


# =====================
# Application Setup
# =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    import asyncio
    
    print("PyReflect Interface Backend starting...")
    print(f"   pyreflect available: {PYREFLECT_AVAILABLE}")
    print(f"   MongoDB available: {MONGODB_AVAILABLE}")
    ensure_backend_layout()
    
    # MongoDB cluster keepalive task
    keepalive_task = None
    if MONGODB_AVAILABLE and mongo_client is not None:
        async def mongo_keepalive():
            """Ping MongoDB every 5 minutes to keep cluster alive."""
            while True:
                try:
                    await asyncio.sleep(300)  # 5 minutes
                    mongo_client.admin.command("ping")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"MongoDB keepalive ping failed: {e}")
        
        keepalive_task = asyncio.create_task(mongo_keepalive())
    
    yield
    
    # Cleanup
    if keepalive_task:
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
    print("PyReflect Interface Backend shutting down...")


app = FastAPI(
    title="PyReflect Interface API",
    description="REST API for neutron reflectivity analysis using pyreflect",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Next.js frontend
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================
# Training Constants
# =====================
LEARNING_RATE = 2.15481e-05
WEIGHT_DECAY = 2.6324e-05
SPLIT_RATIO = 0.8


# =====================
# Helper Functions
# =====================

def compute_norm_stats(curves: np.ndarray) -> dict:
    """Return min/max stats for x and y dimensions."""
    x_points = curves[:, 0, :]
    y_points = curves[:, 1, :]
    return {
        "x": {"min": float(np.min(x_points)), "max": float(np.max(x_points))},
        "y": {"min": float(np.min(y_points)), "max": float(np.max(y_points))},
    }


# S3 Logic removed in favor of Hugging Face

def generate_with_pyreflect_streaming(
    layers: List[FilmLayer], gen_params: GeneratorParams, train_params: TrainingParams,
    user_id: str | None = None, name: str | None = None
) -> Generator[str, None, None]:
    """Generate data with streaming log output via SSE.
    
    Args:
        layers: Film layer definitions
        gen_params: Generator parameters
        train_params: Training parameters
        user_id: Optional GitHub user ID for MongoDB persistence
    """
    
    def emit(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
    
    total_start = time.perf_counter()
    
    def emit_warnings(context: str, warning_list: list[warnings.WarningMessage]) -> Generator[str, None, None]:
        if not warning_list:
            return
        max_warnings = 10
        for w in warning_list[:max_warnings]:
            yield emit("log", f"Warning ({context}): {w.message}")
        if len(warning_list) > max_warnings:
            yield emit("log", f"Warning ({context}): {len(warning_list) - max_warnings} more warnings...")

    # Heartbeat mechanism to prevent Cloudflare proxy timeout
    HEARTBEAT_INTERVAL = 15.0  # seconds
    last_heartbeat = [time.perf_counter()]  # mutable container for closure
    
    def maybe_heartbeat() -> str | None:
        """Return a keepalive comment if enough time has passed, else None."""
        now = time.perf_counter()
        if now - last_heartbeat[0] >= HEARTBEAT_INTERVAL:
            last_heartbeat[0] = now
            return ":keepalive\n\n"
        return None

    class QueueWriter(io.TextIOBase):
        def __init__(self, q: "queue.Queue[str]") -> None:
            super().__init__()
            self.q = q
            self._buffer = ""

        def write(self, s: str) -> int:
            if not s:
                return 0
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self.q.put(line)
            return len(s)

        def flush(self) -> None:
            if self._buffer.strip():
                self.q.put(self._buffer.strip())
            self._buffer = ""
    
    yield emit("log", f"Generating {gen_params.numCurves} synthetic curves with {gen_params.numFilmLayers} film layers...")
    
    # Check local storage limit (Max 2 models)
    # This prevents disk usage buildup if HF upload is not enabled or failing
    try:
        local_models = list(MODELS_DIR.glob("*.pth"))
        if len(local_models) >= 2:
            yield emit("log", f"Error: Local storage limit reached ({len(local_models)}/2 models).")
            yield emit("log", "Please delete old local models or configure Hugging Face to offload them.")
            yield emit("error", "Local model limit reached (max 2). Delete models to continue.")
            return
    except Exception as e:
        yield emit("log", f"Warning: Could not check local model count: {e}")
    
    data_generator = ReflectivityDataGenerator(
        num_layers=gen_params.numFilmLayers,
    )
    gen_start = time.perf_counter()
    log_queue: "queue.Queue[str]" = queue.Queue()
    gen_warnings: list[warnings.WarningMessage] = []
    gen_result: dict[str, Any] = {}
    gen_error: list[BaseException] = []

    def run_generate() -> None:
        writer = QueueWriter(log_queue)
        try:
            with warnings.catch_warnings(record=True) as warn_list:
                warnings.simplefilter("always")
                # Suppress refl1d deprecation warning about data argument
                warnings.filterwarnings("ignore", message=".*data argument is deprecated.*")
                with redirect_stdout(cast(TextIO, writer)), redirect_stderr(cast(TextIO, writer)):
                    result = data_generator.generate(gen_params.numCurves)
            gen_warnings.extend(warn_list)
            gen_result["data"] = result
        except Exception as exc:
            gen_error.append(exc)
        finally:
            writer.flush()

    gen_thread = threading.Thread(target=run_generate, daemon=True)
    gen_thread.start()

    while gen_thread.is_alive() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.2)
            if line.strip():
                yield emit("log", line.rstrip())
        except queue.Empty:
            pass

    gen_thread.join()
    if gen_error:
        raise gen_error[0]
    nr_curves, sld_curves = gen_result["data"]
    gen_time = time.perf_counter() - gen_start
    yield emit("log", f"   Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")
    yield emit("log", f"Generation took {gen_time:.2f}s")
    for warning_msg in emit_warnings("generation", gen_warnings):
        yield warning_msg
    
    yield emit("log", "Preprocessing data...")
    nr_log = np.array(nr_curves, copy=True)
    nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
    nr_stats = compute_norm_stats(nr_log)
    normalized_nr = DataProcessor.normalize_xy_curves(nr_curves, apply_log=True, min_max_stats=nr_stats)
    
    sld_stats = compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(sld_curves, apply_log=False, min_max_stats=sld_stats)
    
    reshaped_nr = normalized_nr[:, 1:2, :]
    
    yield emit("log", f"Training CNN model ({train_params.epochs} epochs, batch size {train_params.batchSize})...")
    model = CNN(layers=train_params.layers, dropout_prob=train_params.dropout).to(DEVICE)
    model.train()
    
    list_arrays = DataProcessor.split_arrays(reshaped_nr, normalized_sld, size_split=SPLIT_RATIO)
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=train_params.batchSize
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()
    
    epoch_list = []
    train_losses = []
    val_losses = []
    
    training_start = time.perf_counter()
    for epoch in range(train_params.epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                val_running_loss += loss_fn(outputs, y_batch).item()
        val_loss = val_running_loss / len(valid_loader)
        
        epoch_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Emit progress every epoch
        yield emit("progress", {
            "epoch": epoch + 1,
            "total": train_params.epochs,
            "trainLoss": train_loss,
            "valLoss": val_loss,
        })
        
        # Emit heartbeat if needed to prevent proxy timeout
        heartbeat = maybe_heartbeat()
        if heartbeat:
            yield heartbeat
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            yield emit("log", f"   Epoch {epoch + 1}/{train_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
            yield emit("log", f"   Epoch {epoch + 1}/{train_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    training_time = time.perf_counter() - training_start
    
    # Save the trained model
    import uuid
    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.pth"
    torch.save(model.state_dict(), model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    yield emit("log", f"Model saved locally: {model_id}.pth ({model_size_mb:.2f} MB)")
    
    
    # HF Upload
    if hf_api:
        yield emit("log", "Uploading to Hugging Face...")
        if upload_to_hf(model_path, model_id):
            yield emit("log", "Model uploaded to Hugging Face Hub")
            # Verify availability before deleting
            yield emit("log", "Verifying upload...")
            try:
                # Use HF API to check existence (robust for private repos)
                if hf_api.file_exists(repo_id=HF_REPO_ID, filename=f"{model_id}.pth", repo_type="dataset"):
                    model_path.unlink()
                    yield emit("log", "Verified on HF. Local model file deleted (cleanup)")
                else:
                    yield emit("log", "Warning: file_exists returned False after upload. keeping local file.")
            except Exception as e:
                yield emit("log", f"Warning: Failed to verify/delete: {e}")
        else:
            yield emit("log", "Warning: Model NOT uploaded to HF (Error occurred)")
    else:
        yield emit("log", "Hugging Face not configured")

    yield emit("log", "Training complete!")
    yield emit("log", f"Training took {training_time:.2f}s")
    yield emit("log", "Running inference on test sample...")
    
    # Get a test sample (use first from validation set)
    split_idx = int(len(nr_curves) * SPLIT_RATIO)
    test_idx = split_idx  # First test sample
    
    # Ground truth
    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]
    
    inference_start = time.perf_counter()
    # Predict SLD from NR using trained model
    model.eval()
    with torch.no_grad():
        test_nr_normalized = normalized_nr[test_idx:test_idx+1, 1:2, :]
        test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(DEVICE)
        pred_sld_normalized = model(test_input).cpu().numpy()  # shape [1, 2, L]
    
    # Denormalize predicted SLD using the correct method
    pred_sld_denorm = DataProcessor.denormalize_xy_curves(
        pred_sld_normalized, 
        stats=sld_stats, 
        apply_exp=False
    )
    pred_sld_y = pred_sld_denorm[0, 1, :]  # Get the y values (SLD) from first sample
    pred_sld_z = pred_sld_denorm[0, 0, :]
    
    # Standard z-axis (0-450 Å) for display only
    sld_z = np.linspace(0, 450, len(gt_sld[1]))
    
    # Compute NR from predicted SLD (round-trip validation)
    if COMPUTE_NR_AVAILABLE and compute_nr_from_sld is not None:
        yield emit("log", "Computing NR from predicted SLD...")
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            with warnings.catch_warnings(record=True) as nr_warnings:
                warnings.simplefilter("always")
                _, computed_r = compute_nr_from_sld(
                    pred_sld_profile,
                    Q=gt_nr[0],  # Use same Q points as ground truth
                    order="substrate_to_air"
                )
            for warning_msg in emit_warnings("computed NR", nr_warnings):
                yield warning_msg
            computed_nr = computed_r.tolist()
        except Exception as e:
            yield emit("log", f"Warning: Could not compute NR from predicted SLD: {e}")
            computed_nr = gt_nr[1].tolist()  # Fallback to ground truth
    else:
        yield emit("log", "Warning: compute_nr_from_sld not available; using ground truth NR.")
        computed_nr = gt_nr[1].tolist()

    # Chi comparison (predicted vs actual SLD values at sample points)
    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        {
            "x": int(i), 
            "predicted": float(pred_sld_y[idx]), 
            "actual": float(gt_sld[1][idx])
        }
        for i, idx in enumerate(sample_indices)
    ]

    final_mse = val_losses[-1] if val_losses else 0.0
    r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
    mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))
    inference_time = time.perf_counter() - inference_start
    total_time = time.perf_counter() - total_start
    
    yield emit("log", f"Timing: generation {gen_time:.2f}s, training {training_time:.2f}s, inference {inference_time:.2f}s, total {total_time:.2f}s")

    result = {
        "nr": {
            "q": gt_nr[0].tolist(), 
            "groundTruth": gt_nr[1].tolist(),
            "computed": computed_nr,
        },
        "sld": {
            "z": sld_z.tolist(), 
            "groundTruth": gt_sld[1].tolist(),
            "predicted": pred_sld_y.tolist(),
        },
        "training": {"epochs": epoch_list, "trainingLoss": train_losses, "validationLoss": val_losses},
        "chi": chi,
        "metrics": {"mse": float(final_mse), "r2": float(np.clip(r2, 0, 1)), "mae": mae},
        "name": name,
        "model_id": model_id,
    }
    yield emit("result", result)
    
    # Save to MongoDB if available and user is authenticated
    if MONGODB_AVAILABLE and generations_collection is not None and user_id:
        from datetime import datetime, timezone
        try:
            doc = {
                "user_id": user_id,
                "name": name,
                "created_at": datetime.now(timezone.utc),
                "params": {
                    "layers": [layer.model_dump() for layer in layers],
                    "generator": gen_params.model_dump(),
                    "training": train_params.model_dump(),
                },
                "result": result,
            }
            generations_collection.insert_one(doc)
            yield emit("log", "Results saved to database.")
        except Exception as e:
            yield emit("log", f"Warning: Could not save to database: {e}")


def generate_with_pyreflect(layers: List[FilmLayer], gen_params: GeneratorParams, train_params: TrainingParams) -> GenerateResponse:
    """Generate data using actual pyreflect package with real training (non-streaming)"""
    print(f"Generating {gen_params.numCurves} synthetic curves with {gen_params.numFilmLayers} film layers...")
    
    data_generator = ReflectivityDataGenerator(
        num_layers=gen_params.numFilmLayers,
    )
    nr_curves, sld_curves = data_generator.generate(gen_params.numCurves)
    print(f"   Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")
    
    print("Preprocessing data...")
    nr_log = np.array(nr_curves, copy=True)
    nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
    nr_stats = compute_norm_stats(nr_log)
    normalized_nr = DataProcessor.normalize_xy_curves(nr_curves, apply_log=True, min_max_stats=nr_stats)
    
    sld_stats = compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(sld_curves, apply_log=False, min_max_stats=sld_stats)
    
    reshaped_nr = normalized_nr[:, 1:2, :]
    
    print(f"Training CNN model ({train_params.epochs} epochs, batch size {train_params.batchSize})...")
    model = CNN(layers=train_params.layers, dropout_prob=train_params.dropout).to(DEVICE)
    model.train()
    
    list_arrays = DataProcessor.split_arrays(reshaped_nr, normalized_sld, size_split=SPLIT_RATIO)
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=train_params.batchSize
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()
    
    epoch_list = []
    train_losses = []
    val_losses = []
    
    for epoch in range(train_params.epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                val_running_loss += loss_fn(outputs, y_batch).item()
        val_loss = val_running_loss / len(valid_loader)
        
        epoch_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch + 1}/{train_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    # Save model
    import uuid
    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_id}.pth")
    
    print("Training complete!")
    print("Running inference on test sample...")
    
    # Get a test sample
    split_idx = int(len(nr_curves) * SPLIT_RATIO)
    test_idx = split_idx
    
    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]
    
    # Predict SLD from NR
    model.eval()
    with torch.no_grad():
        test_nr_normalized = normalized_nr[test_idx:test_idx+1, 1:2, :]
        test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(DEVICE)
        pred_sld_normalized = model(test_input).cpu().numpy()  # shape [1, 2, L]
    
    # Denormalize predicted SLD using the correct method
    pred_sld_denorm = DataProcessor.denormalize_xy_curves(
        pred_sld_normalized, 
        stats=sld_stats, 
        apply_exp=False
    )
    pred_sld_y = pred_sld_denorm[0, 1, :]  # Get the y values (SLD) from first sample
    pred_sld_z = pred_sld_denorm[0, 0, :]
    
    sld_z = np.linspace(0, 450, len(gt_sld[1]))
    
    # Compute NR from predicted SLD (round-trip validation)
    if COMPUTE_NR_AVAILABLE and compute_nr_from_sld is not None:
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            _, computed_r = compute_nr_from_sld(
                pred_sld_profile,
                Q=gt_nr[0],  # Use same Q points as ground truth
                order="substrate_to_air"
            )
            computed_nr = computed_r.tolist()
        except Exception as e:
            print(f"Warning: Could not compute NR from predicted SLD: {e}")
            computed_nr = gt_nr[1].tolist()  # Fallback to ground truth
    else:
        print("Warning: compute_nr_from_sld not available; using ground truth NR.")
        computed_nr = gt_nr[1].tolist()
    
    # Chi comparison
    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        ChiDataPoint(x=int(i), predicted=float(pred_sld_y[idx]), actual=float(gt_sld[1][idx]))
        for i, idx in enumerate(sample_indices)
    ]
    
    final_mse = val_losses[-1] if val_losses else 0.0
    r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
    mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))
    
    return GenerateResponse(
        nr=NRData(
            q=gt_nr[0].tolist(), 
            groundTruth=gt_nr[1].tolist(),
            computed=computed_nr,
        ),
        sld=SLDData(
            z=sld_z.tolist(), 
            groundTruth=gt_sld[1].tolist(),
            predicted=pred_sld_y.tolist(),
        ),
        training=TrainingData(epochs=epoch_list, trainingLoss=train_losses, validationLoss=val_losses),
        chi=chi,
        metrics=Metrics(mse=float(final_mse), r2=float(np.clip(r2, 0, 1)), mae=mae),
        model_id=model_id,
    )


# =====================
# API Endpoints
# =====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pyreflect_available": PYREFLECT_AVAILABLE,
    }


@app.get("/api/limits")
async def get_limits():
    """Get current parameter limits (stricter in production)."""
    return {
        "production": IS_PRODUCTION,
        "limits": LIMITS,
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate NR curves and SLD profiles based on film layer parameters.
    
    Returns neutron reflectivity data, SLD profiles, training loss curves,
    and chi parameter predictions.
    """
    validate_limits(request.generator, request.training)
    if not PYREFLECT_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="pyreflect not available. Please install pyreflect dependencies."
        )
    try:
        return generate_with_pyreflect(request.layers, request.generator, request.training)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    x_user_id: str | None = Header(default=None),
):
    """
    Stream generation progress via Server-Sent Events.
    Events: log (text message), progress (epoch info), result (final data)
    
    Pass X-User-ID header to persist results to MongoDB for the authenticated user.
    """
    validate_limits(request.generator, request.training)
    if not PYREFLECT_AVAILABLE:
        def error_stream():
            yield 'event: error\ndata: "pyreflect not available. Please install pyreflect dependencies."\n\n'
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    return StreamingResponse(
        generate_with_pyreflect_streaming(
            request.layers, request.generator, request.training,
            user_id=x_user_id, name=request.name
        ),
        media_type="text/event-stream",
    )


@app.get("/api/defaults")
async def get_defaults():
    """Get default parameter values"""
    return {
        "layers": [
            FilmLayer(name="substrate", sld=2.07, isld=0, thickness=0, roughness=1.8),
            FilmLayer(name="siox", sld=3.47, isld=0, thickness=12, roughness=2.0),
            FilmLayer(name="polymer_1", sld=3.8, isld=0, thickness=50, roughness=10),
            FilmLayer(name="polymer_2", sld=2.5, isld=0, thickness=150, roughness=30),
            FilmLayer(name="air", sld=0, isld=0, thickness=0, roughness=0),
        ],
        "generator": GeneratorParams(),
        "training": TrainingParams(),
    }





class SaveResultRequest(BaseModel):
    """Request body for saving generation results."""
    layers: List[FilmLayer]
    generator: GeneratorParams
    training: TrainingParams
    result: dict
    name: str | None = None


@app.post("/api/history")
async def save_history(
    request: SaveResultRequest,
    x_user_id: str | None = Header(default=None),
):
    """Save generation results to history manually (e.g. from import)."""
    if not MONGODB_AVAILABLE or generations_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    from datetime import datetime, timezone
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
        result = generations_collection.insert_one(doc)
        return {"success": True, "id": str(result.inserted_id)}
    except Exception as e:
        print(f"Error saving history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")


@app.get("/api/history")
async def get_history(x_user_id: str | None = Header(default=None)):
    """Get list of saved generations for the authenticated user."""
    if not MONGODB_AVAILABLE or generations_collection is None:
         raise HTTPException(status_code=503, detail="MongoDB not available")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    # Fetch recent saves, limit to 50 for now
    cursor = generations_collection.find(
        {"user_id": x_user_id},
        {
            "result.nr": 0, 
            "result.sld": 0, 
            "result.training": 0, 
            "result.chi": 0
        }
    ).sort("created_at", -1).limit(50)
    
    # Post-process to check for local model file existence
    local_model_ids = {p.stem for p in MODELS_DIR.glob("*.pth")}
    
    history = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        # Check if model exists locally
        model_id = doc.get("result", {}).get("model_id")
        doc["is_local"] = model_id in local_model_ids
        
        # Add HF URL if configured
        if model_id and HF_REPO_ID:
            doc["hf_url"] = f"https://huggingface.co/datasets/{HF_REPO_ID}/blob/main/{model_id}.pth"
            
        history.append(doc)
        
    return history


@app.get("/api/history/{save_id}")
async def get_save(save_id: str, x_user_id: str | None = Header(default=None)):
    """Get full data for a specific saved generation."""
    if not MONGODB_AVAILABLE or generations_collection is None:
         raise HTTPException(status_code=503, detail="MongoDB not available")
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
      
    from bson import ObjectId  
    try:
        oid = ObjectId(save_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid ID format")
        
    doc = generations_collection.find_one({"_id": oid, "user_id": x_user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Save not found")
        
    doc["_id"] = str(doc["_id"])
    return doc


@app.delete("/api/history/{save_id}")
async def delete_save(save_id: str, x_user_id: str | None = Header(default=None)):
    """Delete a saved generation."""
    if not MONGODB_AVAILABLE or generations_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID required")
        
    try:
        from bson import ObjectId
        if not ObjectId.is_valid(save_id):
            raise HTTPException(status_code=400, detail="Invalid ID format")

        # Find document first to get model_id for cleanup
        doc = generations_collection.find_one({"_id": ObjectId(save_id), "user_id": x_user_id})
        
        if doc and "result" in doc and "model_id" in doc["result"]:
            model_id = doc["result"]["model_id"]
            if model_id:
                try:
                    local_path = MODELS_DIR / f"{model_id}.pth"
                    if local_path.exists():
                        local_path.unlink()
                        print(f"Deleted orphan model file: {model_id}.pth")
                except Exception as e:
                    print(f"Warning: Failed to delete model file: {e}")

        result = generations_collection.delete_one({
            "_id": ObjectId(save_id),
            "user_id": x_user_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Save not found or access denied")
            
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting save: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete save")


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload dataset/model files to the backend data folder."""
    saved = []
    for uploaded in files:
        if not uploaded.filename:
            continue
        filename = Path(uploaded.filename).name
        if filename in {
            "normalization_stat.npy",
            "trained_nr_sld_model.pth",
            "trained_nr_sld_model_no_dropout.pth",
            "combined_expt_denoised_nr.npy",
            "combined_nr.npy",
            "combined_sld.npy",
        } or filename.endswith((".pth", ".pt")):
            target_dir = DATA_DIR
        elif filename.endswith((".yml", ".yaml")) and filename.startswith("settings"):
            target_dir = BACKEND_ROOT
        else:
            target_dir = CURVES_DIR

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        with target_path.open("wb") as buffer:
            while True:
                chunk = await uploaded.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
        saved.append(str(target_path))

    return {"saved": saved}


@app.get("/api/status")
async def get_status():
    """Return current backend status including available data files."""
    def list_files(directory: Path, extensions: tuple) -> List[str]:
        if not directory.exists():
            return []
        return [f.name for f in directory.iterdir() if f.is_file() and f.suffix in extensions]

    data_files = list_files(DATA_DIR, (".npy", ".pth", ".pt"))
    curve_files = list_files(CURVES_DIR, (".npy",))
    expt_files = list_files(EXPT_DIR, (".npy",))
    has_settings = SETTINGS_PATH.exists()

    return {
        "pyreflect_available": PYREFLECT_AVAILABLE,
        "has_settings": has_settings,
        "data_files": data_files,
        "curve_files": curve_files,
        "expt_files": expt_files,
    }


    return {
        "pyreflect_available": PYREFLECT_AVAILABLE,
        "has_settings": has_settings,
        "data_files": data_files,
        "curve_files": curve_files,
        "expt_files": expt_files,
    }


@app.get("/api/models/{model_id}")
async def download_model(model_id: str):
    """Download a saved model .pth file."""
    # Sanitize inputs
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")
        
    file_path = MODELS_DIR / f"{model_id}.pth"
    
    # 1. Try local file first (fastest)
    if file_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(
            path=file_path, 
            filename=f"pyreflect_model_{model_id[:8]}.pth",
            media_type="application/octet-stream"
        )
    
    # 2. Fallback to Hugging Face (Public Repo)
    if HF_REPO_ID:
        from fastapi.responses import RedirectResponse
        # Construct raw download URL for public repo (Dataset)
        hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{model_id}.pth"
        return RedirectResponse(url=hf_url)
        
    raise HTTPException(status_code=404, detail="Model not found locally or on Hugging Face")


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a local model .pth file."""
    if not model_id or "/" in model_id or "\\" in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID")
        
    file_path = MODELS_DIR / f"{model_id}.pth"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found locally")
        
    try:
        file_path.unlink()
        return {"status": "deleted", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}")


@app.get("/api/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get model size and status."""
    # 1. Check local
    local_path = MODELS_DIR / f"{model_id}.pth"
    if local_path.exists():
         size = local_path.stat().st_size / (1024*1024)
         return {"size_mb": size, "source": "local"}
    
    # 2. Check HF
    if HF_REPO_ID and hf_api:
        try:
             # Use get_paths_info for specific file metadata in dataset repo
             paths = hf_api.get_paths_info(repo_id=HF_REPO_ID, paths=[f"{model_id}.pth"], repo_type="dataset")
             if paths and len(paths) > 0:
                 size = paths[0].size / (1024*1024)
                 return {"size_mb": size, "source": "huggingface"}
        except Exception as e:
             print(f"HF Check failed: {e}")
             
    return {"size_mb": None, "source": "unknown"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
