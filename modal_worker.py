"""
PyReflect Modal GPU Worker

Deploy with: modal deploy modal_worker.py
The worker automatically scales up when jobs arrive and scales down when idle.
"""

import modal

# Create Modal app
app = modal.App("pyreflect-worker")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "redis",
        "rq",
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "pymongo",
        "huggingface_hub",
    )
    .pip_install("git+https://github.com/williamQyq/pyreflect.git")
)


@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU (cheapest)
    timeout=3600,  # 1 hour max per job
    secrets=[modal.Secret.from_name("pyreflect-redis")],  # Redis credentials
)
def process_training_job():
    """
    Connect to Redis queue and process one training job.
    Modal will auto-scale this function based on queue depth.
    """
    import os
    import time
    import uuid
    from datetime import datetime, timezone
    from typing import Any

    import numpy as np
    import torch
    from redis import Redis
    from rq import Queue
    from rq.job import Job

    # Import pyreflect components
    from pyreflect.input.reflectivity_data_generator import ReflectivityDataGenerator
    from pyreflect.input.data_processor import DataProcessor
    from pyreflect.models.cnn import CNN
    from pyreflect.config.runtime import DEVICE

    try:
        from pyreflect.pipelines.helper import compute_nr_from_sld
        COMPUTE_NR_AVAILABLE = True
    except ImportError:
        COMPUTE_NR_AVAILABLE = False

    # Training constants
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    SPLIT_RATIO = 0.8

    # Connect to Redis
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        print("❌ REDIS_URL not set in secrets")
        return

    redis_conn = Redis.from_url(redis_url)
    queue = Queue("training", connection=redis_conn)

    print(f"✅ Connected to Redis, queue has {len(queue)} jobs")

    # Dequeue one job
    job = queue.dequeue()
    if not job:
        print("No jobs in queue")
        return

    print(f"📋 Processing job: {job.id}")

    # Get job parameters
    job_params = job.args[0] if job.args else {}
    kwargs = job.kwargs or {}
    user_id = kwargs.get("user_id")
    name = kwargs.get("name")
    mongo_uri = kwargs.get("mongo_uri")

    logs = []

    def log(message: str):
        print(message)
        logs.append(message)
        job.meta["logs"] = logs
        job.meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        job.save_meta()

    def update_progress(epoch: int, total: int, train_loss: float, val_loss: float):
        job.meta["progress"] = {
            "epoch": epoch,
            "total": total,
            "trainLoss": train_loss,
            "valLoss": val_loss,
        }
        job.save_meta()

    def _compute_norm_stats(curves: np.ndarray) -> dict:
        x_points = curves[:, 0, :]
        y_points = curves[:, 1, :]
        return {
            "x": {"min": float(np.min(x_points)), "max": float(np.max(x_points))},
            "y": {"min": float(np.min(y_points)), "max": float(np.max(y_points))},
        }

    try:
        # Initialize job meta
        job.meta["status"] = "initializing"
        job.meta["logs"] = logs
        if user_id:
            job.meta["user_id"] = user_id
        if name:
            job.meta["name"] = name
        job.meta["started_at"] = datetime.now(timezone.utc).isoformat()
        job.save_meta()

        # Extract parameters
        gen_params = job_params.get("generator", {})
        train_params = job_params.get("training", {})

        num_curves = gen_params.get("numCurves", 1000)
        num_film_layers = gen_params.get("numFilmLayers", 3)
        epochs = train_params.get("epochs", 50)
        batch_size = train_params.get("batchSize", 32)
        layers = train_params.get("layers", [512, 256, 128])
        dropout = train_params.get("dropout", 0.2)

        total_start = time.perf_counter()

        # Data Generation
        log(f"🔄 Generating {num_curves} curves with {num_film_layers} layers...")
        job.meta["status"] = "generating"
        job.save_meta()

        gen_start = time.perf_counter()
        data_generator = ReflectivityDataGenerator(num_layers=num_film_layers)
        nr_curves, sld_curves = data_generator.generate(num_curves)
        gen_time = time.perf_counter() - gen_start
        log(f"   Generated in {gen_time:.2f}s")

        # Preprocessing
        log("📊 Preprocessing...")
        job.meta["status"] = "preprocessing"
        job.save_meta()

        nr_log = np.array(nr_curves, copy=True)
        nr_log[:, 1, :] = np.log10(np.clip(nr_log[:, 1, :], 1e-8, None))
        nr_stats = _compute_norm_stats(nr_log)
        normalized_nr = DataProcessor.normalize_xy_curves(nr_curves, apply_log=True, min_max_stats=nr_stats)

        sld_stats = _compute_norm_stats(sld_curves)
        normalized_sld = DataProcessor.normalize_xy_curves(sld_curves, apply_log=False, min_max_stats=sld_stats)

        reshaped_nr = normalized_nr[:, 1:2, :]

        # Training
        log(f"🏋️ Training ({epochs} epochs)...")
        job.meta["status"] = "training"
        job.save_meta()

        model = CNN(layers=layers, dropout_prob=dropout).to(DEVICE)
        model.train()

        list_arrays = DataProcessor.split_arrays(reshaped_nr, normalized_sld, size_split=SPLIT_RATIO)
        tensor_arrays = DataProcessor.convert_tensors(list_arrays)
        _, _, _, train_loader, valid_loader, _ = DataProcessor.get_dataloaders(*tensor_arrays, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = torch.nn.MSELoss()

        epoch_list = []
        train_losses = []
        val_losses = []

        training_start = time.perf_counter()
        for epoch in range(epochs):
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

            update_progress(epoch + 1, epochs, train_loss, val_loss)
            log(f"   Epoch {epoch + 1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            # Check for stop request
            job.refresh()
            if job.meta.get("stop_requested"):
                log("⚠️ Stop requested")
                break

        training_time = time.perf_counter() - training_start
        log(f"   Training completed in {training_time:.2f}s")

        # Inference
        log("🔍 Running inference...")
        job.meta["status"] = "inference"
        job.save_meta()

        split_idx = int(len(nr_curves) * SPLIT_RATIO)
        gt_nr = nr_curves[split_idx]
        gt_sld = sld_curves[split_idx]

        model.eval()
        with torch.no_grad():
            test_nr_normalized = normalized_nr[split_idx:split_idx + 1, 1:2, :]
            test_input = torch.tensor(test_nr_normalized, dtype=torch.float32).to(DEVICE)
            pred_sld_normalized = model(test_input).cpu().numpy()

        pred_sld_denorm = DataProcessor.denormalize_xy_curves(pred_sld_normalized, stats=sld_stats, apply_exp=False)
        pred_sld_y = pred_sld_denorm[0, 1, :]
        pred_sld_z = pred_sld_denorm[0, 0, :]

        sld_z = np.linspace(0, 450, len(gt_sld[1]))

        # Compute NR from predicted SLD
        computed_nr = gt_nr[1].tolist()
        if COMPUTE_NR_AVAILABLE:
            try:
                _, computed_r = compute_nr_from_sld((pred_sld_z, pred_sld_y), Q=gt_nr[0], order="substrate_to_air")
                computed_nr = computed_r.tolist()
            except Exception as exc:
                log(f"   Warning: Could not compute NR: {exc}")

        # Metrics
        sample_indices = np.linspace(0, len(pred_sld_y) - 1, 50, dtype=int)
        chi = [
            {"x": int(i), "predicted": float(pred_sld_y[idx]), "actual": float(gt_sld[1][idx])}
            for i, idx in enumerate(sample_indices)
        ]

        final_mse = val_losses[-1] if val_losses else 0.0
        r2 = 1 - (final_mse / np.var(normalized_sld[:, 1, :]))
        mae = float(np.mean(np.abs(pred_sld_y - gt_sld[1])))
        total_time = time.perf_counter() - total_start

        model_id = str(uuid.uuid4())

        log(f"✅ Complete! Total: {total_time:.2f}s, MSE: {final_mse:.6f}")

        # Build result
        result = {
            "nr": {"q": gt_nr[0].tolist(), "groundTruth": gt_nr[1].tolist(), "computed": computed_nr},
            "sld": {"z": sld_z.tolist(), "groundTruth": gt_sld[1].tolist(), "predicted": pred_sld_y.tolist()},
            "training": {"epochs": epoch_list, "trainingLoss": train_losses, "validationLoss": val_losses},
            "chi": chi,
            "metrics": {"mse": float(final_mse), "r2": float(np.clip(r2, 0, 1)), "mae": mae},
            "name": name,
            "model_id": model_id,
            "timing": {"generation": gen_time, "training": training_time, "total": total_time},
        }

        # Save to MongoDB
        if mongo_uri and user_id:
            job.meta["status"] = "saving_to_history"
            job.save_meta()
            log("💾 Saving to database...")
            try:
                from pymongo import MongoClient
                client = MongoClient(mongo_uri)
                db = client.get_default_database()
                db.generations.insert_one({
                    "user_id": user_id,
                    "name": name,
                    "created_at": datetime.now(timezone.utc),
                    "params": job_params,
                    "result": result,
                })
                log("   ✅ Saved!")
            except Exception as exc:
                log(f"   ⚠️ DB error: {exc}")

        # Finalize
        job.meta["status"] = "completed"
        job.meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        job.save_meta()

        # Set job result
        job.set_status("finished")
        return result

    except Exception as exc:
        log(f"❌ Error: {exc}")
        job.meta["status"] = "failed"
        job.meta["error"] = str(exc)
        job.save_meta()
        job.set_status("failed")
        raise


@app.function(image=image, schedule=modal.Cron("* * * * *"))  # Every minute
def poll_queue():
    """
    Cron job that checks the queue and spawns workers for pending jobs.
    This ensures jobs are processed even when no worker is running.
    """
    import os
    from redis import Redis
    from rq import Queue

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return

    redis_conn = Redis.from_url(redis_url)
    queue = Queue("training", connection=redis_conn)

    pending = len(queue)
    if pending > 0:
        print(f"📋 {pending} jobs pending, spawning worker...")
        process_training_job.spawn()


# For local testing
if __name__ == "__main__":
    print("Deploy with: modal deploy modal_worker.py")
    print("Or run locally: modal run modal_worker.py::process_training_job")
