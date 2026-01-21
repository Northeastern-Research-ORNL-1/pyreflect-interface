from __future__ import annotations

import io
import json
import queue
import threading
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Generator, TextIO, cast

import numpy as np

from ..config import (
    LEARNING_RATE,
    LOCAL_MODEL_WAIT_POLL_S,
    LOCAL_MODEL_WAIT_TIMEOUT_S,
    MAX_LOCAL_MODELS,
    MODELS_DIR,
    SPLIT_RATIO,
    WEIGHT_DECAY,
)
from ..integrations.huggingface import HuggingFaceIntegration, upload_model
from ..schemas import (
    ChiDataPoint,
    GenerateResponse,
    GeneratorParams,
    Metrics,
    NRData,
    SLDData,
    TrainingData,
    TrainingParams,
)
from .pyreflect_runtime import PYREFLECT, resolve_torch_device
from .local_model_limit import (
    save_torch_state_dict_with_local_limit,
    wait_for_local_model_slot,
)


def compute_norm_stats(curves: np.ndarray) -> dict:
    x_points = curves[:, 0, :]
    y_points = curves[:, 1, :]
    return {
        "x": {"min": float(np.min(x_points)), "max": float(np.max(x_points))},
        "y": {"min": float(np.min(y_points)), "max": float(np.max(y_points))},
    }


def generate_with_pyreflect_streaming(
    *,
    layers,
    gen_params: GeneratorParams,
    train_params: TrainingParams,
    user_id: str | None,
    name: str | None,
    mongo_generations,
    hf: HuggingFaceIntegration,
) -> Generator[str, None, None]:
    def emit(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    if not PYREFLECT.available or PYREFLECT.reflectivity_pipeline is None:
        yield emit(
            "error", "pyreflect not available. Please install pyreflect dependencies."
        )
        return

    ReflectivityDataGenerator = PYREFLECT.ReflectivityDataGenerator
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    device, device_reason = resolve_torch_device(
        torch, runtime_device=runtime_device, prefer_cuda=True
    )
    if device_reason:
        yield emit("log", f"Warning: {device_reason}")
    yield emit("log", f"Device selected: {device!s}")

    total_start = time.perf_counter()

    def emit_warnings(
        context: str, warning_list: list[warnings.WarningMessage]
    ) -> Generator[str, None, None]:
        if not warning_list:
            return
        max_warnings = 10
        for w in warning_list[:max_warnings]:
            yield emit("log", f"Warning ({context}): {w.message}")
        if len(warning_list) > max_warnings:
            yield emit(
                "log",
                f"Warning ({context}): {len(warning_list) - max_warnings} more warnings...",
            )

    HEARTBEAT_INTERVAL = 15.0
    last_heartbeat = [time.perf_counter()]

    def maybe_heartbeat() -> str | None:
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

    yield emit(
        "log",
        f"Generating {gen_params.numCurves} synthetic curves with {gen_params.numFilmLayers} film layers...",
    )

    try:
        for msg in wait_for_local_model_slot(
            models_dir=MODELS_DIR,
            max_models=MAX_LOCAL_MODELS,
            timeout_s=LOCAL_MODEL_WAIT_TIMEOUT_S,
            poll_s=LOCAL_MODEL_WAIT_POLL_S,
        ):
            yield emit("log", msg)
            heartbeat = maybe_heartbeat()
            if heartbeat:
                yield heartbeat
    except TimeoutError as exc:
        yield emit("log", f"Error: {exc}")
        yield emit(
            "log",
            "Delete old local models or configure Hugging Face to offload them.",
        )
        yield emit("error", str(exc))
        return
    except Exception as exc:
        yield emit("log", f"Warning: Could not check/wait for local model slots: {exc}")

    layer_desc = None
    layer_bound = None
    if gen_params.layerBound:
        layer_desc = [
            layer.model_dump() if hasattr(layer, "model_dump") else layer
            for layer in layers
        ]
        layer_bound = [
            b.model_dump() if hasattr(b, "model_dump") else b
            for b in gen_params.layerBound
        ]

    data_generator = ReflectivityDataGenerator(
        num_layers=gen_params.numFilmLayers,
        layer_desc=layer_desc,
        layer_bound=layer_bound,
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
                warnings.filterwarnings(
                    "ignore", message=".*data argument is deprecated.*"
                )
                with (
                    redirect_stdout(cast(TextIO, writer)),
                    redirect_stderr(cast(TextIO, writer)),
                ):
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
    yield emit(
        "log",
        f"   Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}",
    )
    yield emit("log", f"Generation took {gen_time:.2f}s")
    for warning_msg in emit_warnings("generation", gen_warnings):
        yield warning_msg

    yield emit("log", "Preprocessing data...")
    nr_log = np.array(nr_curves, copy=True)
    nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
    nr_stats = compute_norm_stats(nr_log)
    normalized_nr = DataProcessor.normalize_xy_curves(
        nr_curves, apply_log=True, min_max_stats=nr_stats
    )

    sld_stats = compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(
        sld_curves, apply_log=False, min_max_stats=sld_stats
    )

    reshaped_nr = normalized_nr[:, 1:2, :]

    yield emit(
        "log",
        f"Training CNN model ({train_params.epochs} epochs, batch size {train_params.batchSize})...",
    )
    model = CNN(layers=train_params.layers, dropout_prob=train_params.dropout).to(
        device
    )
    model.train()

    list_arrays = DataProcessor.split_arrays(
        reshaped_nr, normalized_sld, size_split=SPLIT_RATIO
    )
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=train_params.batchSize
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.MSELoss()

    epoch_list = []
    train_losses = []
    val_losses = []

    training_start = time.perf_counter()
    for epoch in range(train_params.epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_running_loss += loss_fn(outputs, y_batch).item()
        val_loss = val_running_loss / len(valid_loader)

        epoch_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        yield emit(
            "progress",
            {
                "epoch": epoch + 1,
                "total": train_params.epochs,
                "trainLoss": train_loss,
                "valLoss": val_loss,
            },
        )

        heartbeat = maybe_heartbeat()
        if heartbeat:
            yield heartbeat

        yield emit(
            "log",
            f"   Epoch {epoch + 1}/{train_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}",
        )

    training_time = time.perf_counter() - training_start

    import uuid

    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.pth"
    yield emit("log", "Preparing model for save (moving tensors to CPU)...")
    try:
        raw_state_dict = model.state_dict()
        cpu_state_dict = {}
        for key, value in raw_state_dict.items():
            try:
                cpu_state_dict[key] = value.detach().cpu()  # type: ignore[union-attr]
            except Exception:
                cpu_state_dict[key] = value
    except Exception as exc:
        yield emit("log", f"Warning: Failed to prepare CPU state_dict: {exc}")
        cpu_state_dict = model.state_dict()
    try:
        for msg in save_torch_state_dict_with_local_limit(
            torch=torch,
            state_dict=cpu_state_dict,
            model_path=model_path,
            models_dir=MODELS_DIR,
            max_models=MAX_LOCAL_MODELS,
            timeout_s=LOCAL_MODEL_WAIT_TIMEOUT_S,
            poll_s=LOCAL_MODEL_WAIT_POLL_S,
            user_id=None,
        ):
            yield emit("log", msg)
            heartbeat = maybe_heartbeat()
            if heartbeat:
                yield heartbeat
    except TimeoutError as exc:
        yield emit("log", f"Error: {exc}")
        yield emit("error", str(exc))
        return
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    yield emit("log", f"Model saved locally: {model_id}.pth ({model_size_mb:.2f} MB)")

    if hf.available and hf.api and hf.repo_id:
        yield emit("log", "Uploading to Hugging Face...")
        if upload_model(hf, model_path, model_id):
            yield emit("log", "Model uploaded to Hugging Face Hub")
            yield emit("log", "Verifying upload...")
            try:
                if hf.api.file_exists(
                    repo_id=hf.repo_id,
                    filename=f"{model_id}.pth",
                    repo_type="dataset",
                ):
                    model_path.unlink()
                    yield emit(
                        "log", "Verified on HF. Local model file deleted (cleanup)"
                    )
                else:
                    yield emit(
                        "log",
                        "Warning: file_exists returned False after upload. keeping local file.",
                    )
            except Exception as exc:
                yield emit("log", f"Warning: Failed to verify/delete: {exc}")
        else:
            yield emit("log", "Warning: Model NOT uploaded to HF (Error occurred)")
    else:
        yield emit("log", "Hugging Face not configured")

    yield emit("log", "Training complete!")
    yield emit("log", f"Training took {training_time:.2f}s")
    yield emit("log", "Running inference on test sample...")

    split_idx = int(len(nr_curves) * SPLIT_RATIO)
    test_idx = split_idx

    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]

    inference_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        test_nr_normalized = normalized_nr[test_idx : test_idx + 1, 1:2, :]
        test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(device)
        pred_sld_normalized = model(test_input).cpu().numpy()

    pred_sld_denorm = DataProcessor.denormalize_xy_curves(
        pred_sld_normalized,
        stats=sld_stats,
        apply_exp=False,
    )
    pred_sld_y = pred_sld_denorm[0, 1, :]
    pred_sld_z = pred_sld_denorm[0, 0, :]

    sld_z = np.linspace(0, 450, len(gt_sld[1]))

    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        yield emit("log", "Computing NR from predicted SLD...")
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            with warnings.catch_warnings(record=True) as nr_warnings:
                warnings.simplefilter("always")
                _, computed_r = compute_nr_from_sld(
                    pred_sld_profile,
                    Q=gt_nr[0],
                    order="substrate_to_air",
                )
            for warning_msg in emit_warnings("computed NR", nr_warnings):
                yield warning_msg
            computed_nr = computed_r.tolist()
        except Exception as exc:
            yield emit(
                "log", f"Warning: Could not compute NR from predicted SLD: {exc}"
            )
            computed_nr = gt_nr[1].tolist()
    else:
        yield emit(
            "log", "Warning: compute_nr_from_sld not available; using ground truth NR."
        )
        computed_nr = gt_nr[1].tolist()

    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        {
            "x": int(i),
            "predicted": float(pred_sld_y[idx]),
            "actual": float(gt_sld[1][idx]),
        }
        for i, idx in enumerate(sample_indices)
    ]

    final_mse = val_losses[-1] if val_losses else 0.0
    r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
    mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))
    inference_time = time.perf_counter() - inference_start
    total_time = time.perf_counter() - total_start

    yield emit(
        "log",
        f"Timing: generation {gen_time:.2f}s, training {training_time:.2f}s, inference {inference_time:.2f}s, total {total_time:.2f}s",
    )

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
        "training": {
            "epochs": epoch_list,
            "trainingLoss": train_losses,
            "validationLoss": val_losses,
        },
        "chi": chi,
        "metrics": {
            "mse": float(final_mse),
            "r2": float(np.clip(r2, 0, 1)),
            "mae": mae,
        },
        "name": name,
        "model_id": model_id,
    }
    yield emit("result", result)

    if mongo_generations is not None and user_id:
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
            mongo_generations.insert_one(doc)
            yield emit("log", "Results saved to database.")
        except Exception as exc:
            yield emit("log", f"Warning: Could not save to database: {exc}")


def generate_with_pyreflect(
    layers,
    gen_params: GeneratorParams,
    train_params: TrainingParams,
) -> GenerateResponse:
    if not PYREFLECT.available:
        raise RuntimeError("pyreflect not available")

    ReflectivityDataGenerator = PYREFLECT.ReflectivityDataGenerator
    DataProcessor = PYREFLECT.DataProcessor
    CNN = PYREFLECT.CNN
    runtime_device = PYREFLECT.DEVICE
    torch = PYREFLECT.torch
    compute_nr_from_sld = PYREFLECT.compute_nr_from_sld

    device, device_reason = resolve_torch_device(
        torch, runtime_device=runtime_device, prefer_cuda=True
    )
    if device_reason:
        print(f"Warning: {device_reason}")
    print(f"Device selected: {device!s}")

    print(
        f"Generating {gen_params.numCurves} synthetic curves with {gen_params.numFilmLayers} film layers..."
    )

    layer_desc = None
    layer_bound = None
    if gen_params.layerBound:
        layer_desc = [
            layer.model_dump() if hasattr(layer, "model_dump") else layer
            for layer in layers
        ]
        layer_bound = [
            b.model_dump() if hasattr(b, "model_dump") else b
            for b in gen_params.layerBound
        ]

    data_generator = ReflectivityDataGenerator(
        num_layers=gen_params.numFilmLayers,
        layer_desc=layer_desc,
        layer_bound=layer_bound,
    )
    nr_curves, sld_curves = data_generator.generate(gen_params.numCurves)
    print(f"   Generated NR shape: {nr_curves.shape}, SLD shape: {sld_curves.shape}")

    print("Preprocessing data...")
    nr_log = np.array(nr_curves, copy=True)
    nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
    nr_stats = compute_norm_stats(nr_log)
    normalized_nr = DataProcessor.normalize_xy_curves(
        nr_curves, apply_log=True, min_max_stats=nr_stats
    )

    sld_stats = compute_norm_stats(sld_curves)
    normalized_sld = DataProcessor.normalize_xy_curves(
        sld_curves, apply_log=False, min_max_stats=sld_stats
    )

    reshaped_nr = normalized_nr[:, 1:2, :]

    print(
        f"Training CNN model ({train_params.epochs} epochs, batch size {train_params.batchSize})..."
    )
    model = CNN(layers=train_params.layers, dropout_prob=train_params.dropout).to(
        device
    )
    model.train()

    list_arrays = DataProcessor.split_arrays(
        reshaped_nr, normalized_sld, size_split=SPLIT_RATIO
    )
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(
        *tensor_arrays, batch_size=train_params.batchSize
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.MSELoss()

    epoch_list: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(train_params.epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_running_loss += loss_fn(outputs, y_batch).item()
        val_loss = val_running_loss / len(valid_loader)

        epoch_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"   Epoch {epoch + 1}/{train_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}"
            )

    import uuid

    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.pth"
    print("Preparing model for save (moving tensors to CPU)...")
    try:
        raw_state_dict = model.state_dict()
        cpu_state_dict = {}
        for key, value in raw_state_dict.items():
            try:
                cpu_state_dict[key] = value.detach().cpu()  # type: ignore[union-attr]
            except Exception:
                cpu_state_dict[key] = value
    except Exception as exc:
        print(f"Warning: Failed to prepare CPU state_dict: {exc}")
        cpu_state_dict = model.state_dict()
    for msg in save_torch_state_dict_with_local_limit(
        torch=torch,
        state_dict=cpu_state_dict,
        model_path=model_path,
        models_dir=MODELS_DIR,
        max_models=MAX_LOCAL_MODELS,
        timeout_s=LOCAL_MODEL_WAIT_TIMEOUT_S,
        poll_s=LOCAL_MODEL_WAIT_POLL_S,
        user_id=None,
    ):
        print(msg)
    print(f"Model saved: {model_id}.pth")

    print("Training complete!")
    print("Running inference on test sample...")

    split_idx = int(len(nr_curves) * SPLIT_RATIO)
    test_idx = split_idx

    gt_nr = nr_curves[test_idx]
    gt_sld = sld_curves[test_idx]

    model.eval()
    with torch.no_grad():
        test_nr_normalized = normalized_nr[test_idx : test_idx + 1, 1:2, :]
        test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(device)
        pred_sld_normalized = model(test_input).cpu().numpy()

    pred_sld_denorm = DataProcessor.denormalize_xy_curves(
        pred_sld_normalized,
        stats=sld_stats,
        apply_exp=False,
    )
    pred_sld_y = pred_sld_denorm[0, 1, :]
    pred_sld_z = pred_sld_denorm[0, 0, :]

    sld_z = np.linspace(0, 450, len(gt_sld[1]))

    if PYREFLECT.compute_nr_available and compute_nr_from_sld is not None:
        try:
            pred_sld_profile = (pred_sld_z, pred_sld_y)
            _, computed_r = compute_nr_from_sld(
                pred_sld_profile,
                Q=gt_nr[0],
                order="substrate_to_air",
            )
            computed_nr = computed_r.tolist()
        except Exception as exc:
            print(f"Warning: Could not compute NR from predicted SLD: {exc}")
            computed_nr = gt_nr[1].tolist()
    else:
        print("Warning: compute_nr_from_sld not available; using ground truth NR.")
        computed_nr = gt_nr[1].tolist()

    sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
    chi = [
        ChiDataPoint(
            x=int(i), predicted=float(pred_sld_y[idx]), actual=float(gt_sld[1][idx])
        )
        for i, idx in enumerate(sample_indices)
    ]

    final_mse = val_losses[-1] if val_losses else 0.0
    r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
    mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))

    return GenerateResponse(
        nr=NRData(
            q=gt_nr[0].tolist(), groundTruth=gt_nr[1].tolist(), computed=computed_nr
        ),
        sld=SLDData(
            z=sld_z.tolist(),
            groundTruth=gt_sld[1].tolist(),
            predicted=pred_sld_y.tolist(),
        ),
        training=TrainingData(
            epochs=epoch_list, trainingLoss=train_losses, validationLoss=val_losses
        ),
        chi=chi,
        metrics=Metrics(mse=float(final_mse), r2=float(np.clip(r2, 0, 1)), mae=mae),
        model_id=model_id,
    )
