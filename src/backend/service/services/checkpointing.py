"""
Checkpointing service for training jobs.

Saves and loads training checkpoints to allow resuming after crashes.
Checkpoints are stored on a separate HuggingFace Hub repo (HF_CHECKPOINT_REPO_ID)
to keep them isolated from final models.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huggingface_hub import HfApi


@dataclass
class Checkpoint:
    """Training checkpoint data."""

    job_id: str
    epoch: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    train_losses: list[float]
    val_losses: list[float]
    epoch_list: list[int]
    best_val_loss: float
    saved_at: str
    # Optional: normalization stats needed to resume
    nr_stats: dict | None = None
    sld_stats: dict | None = None


def _get_checkpoint_repo() -> str | None:
    """Get the checkpoint repo ID from config."""
    from ..config import HF_CHECKPOINT_REPO_ID, HF_REPO_ID

    # Use dedicated checkpoint repo if set, otherwise fall back to main repo
    return HF_CHECKPOINT_REPO_ID or HF_REPO_ID


def _get_checkpoint_path(job_id: str) -> str:
    """Get the path for a checkpoint file within the repo."""
    from ..config import HF_CHECKPOINT_REPO_ID

    # If using dedicated checkpoint repo, store at root level
    # If using main repo, store in checkpoints/ subfolder
    if HF_CHECKPOINT_REPO_ID:
        return f"{job_id}.pth"
    return f"checkpoints/{job_id}.pth"


def _ensure_checkpoint_repo(api: "HfApi", repo_id: str, log_fn: Any = print) -> bool:
    """Ensure the checkpoint repo exists, create if needed."""
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
        return True
    except Exception as exc:
        log_fn(f"Warning: Could not create checkpoint repo {repo_id}: {exc}")
        return False


def save_checkpoint(
    api: "HfApi",
    torch: Any,
    checkpoint: Checkpoint,
    log_fn: Any = print,
) -> bool:
    """
    Save a checkpoint to HuggingFace Hub.

    Always overwrites the same file to avoid clutter.
    Returns True on success, False on failure.
    """
    repo_id = _get_checkpoint_repo()
    if not repo_id or not api:
        log_fn("Warning: HuggingFace not configured for checkpointing")
        return False

    try:
        # Ensure repo exists
        _ensure_checkpoint_repo(api, repo_id, log_fn)

        checkpoint_data = {
            "job_id": checkpoint.job_id,
            "epoch": checkpoint.epoch,
            "model_state_dict": checkpoint.model_state_dict,
            "optimizer_state_dict": checkpoint.optimizer_state_dict,
            "train_losses": checkpoint.train_losses,
            "val_losses": checkpoint.val_losses,
            "epoch_list": checkpoint.epoch_list,
            "best_val_loss": checkpoint.best_val_loss,
            "saved_at": checkpoint.saved_at,
            "nr_stats": checkpoint.nr_stats,
            "sld_stats": checkpoint.sld_stats,
        }

        buffer = io.BytesIO()
        torch.save(checkpoint_data, buffer)
        buffer.seek(0)

        path = _get_checkpoint_path(checkpoint.job_id)
        api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        log_fn(f"Checkpoint saved: epoch {checkpoint.epoch} -> {repo_id}/{path}")
        return True

    except Exception as exc:
        log_fn(f"Warning: Failed to save checkpoint: {exc}")
        return False


def _move_optimizer_state_to_device(
    optimizer_state_dict: dict[str, Any], device: Any, torch: Any
) -> dict[str, Any]:
    """
    Move optimizer state tensors to the specified device.

    The optimizer state dict has structure:
    {
        'state': {param_id: {'step': tensor, 'exp_avg': tensor, ...}},
        'param_groups': [...]
    }

    torch.load with map_location doesn't always move nested tensors properly,
    so we need to do it manually.
    """
    if "state" not in optimizer_state_dict:
        return optimizer_state_dict

    for param_id, param_state in optimizer_state_dict["state"].items():
        for key, value in param_state.items():
            if torch.is_tensor(value):
                param_state[key] = value.to(device)

    return optimizer_state_dict


def load_checkpoint(
    api: "HfApi",
    torch: Any,
    job_id: str,
    device: Any,
    log_fn: Any = print,
) -> Checkpoint | None:
    """
    Load a checkpoint from HuggingFace Hub.

    Returns None if no checkpoint exists or loading fails.
    """
    repo_id = _get_checkpoint_repo()
    if not repo_id or not api:
        return None

    path = _get_checkpoint_path(job_id)

    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type="dataset",
            token=api.token,
        )

        checkpoint_data = torch.load(
            local_path, map_location=device, weights_only=False
        )

        # Move optimizer state tensors to device (torch.load doesn't always do this)
        optimizer_state = checkpoint_data["optimizer_state_dict"]
        optimizer_state = _move_optimizer_state_to_device(
            optimizer_state, device, torch
        )

        checkpoint = Checkpoint(
            job_id=checkpoint_data["job_id"],
            epoch=checkpoint_data["epoch"],
            model_state_dict=checkpoint_data["model_state_dict"],
            optimizer_state_dict=optimizer_state,
            train_losses=checkpoint_data["train_losses"],
            val_losses=checkpoint_data["val_losses"],
            epoch_list=checkpoint_data["epoch_list"],
            best_val_loss=checkpoint_data["best_val_loss"],
            saved_at=checkpoint_data["saved_at"],
            nr_stats=checkpoint_data.get("nr_stats"),
            sld_stats=checkpoint_data.get("sld_stats"),
        )

        log_fn(f"Checkpoint loaded: epoch {checkpoint.epoch} from {repo_id}/{path}")
        return checkpoint

    except Exception as exc:
        # File doesn't exist or can't be loaded - that's fine, start fresh
        log_fn(f"No checkpoint found for job {job_id}: {exc}")
        return None


def delete_checkpoint(
    api: "HfApi",
    job_id: str,
    log_fn: Any = print,
) -> bool:
    """
    Delete a checkpoint from HuggingFace Hub after successful completion.

    Returns True on success, False on failure (non-fatal).
    """
    repo_id = _get_checkpoint_repo()
    if not repo_id or not api:
        return False

    path = _get_checkpoint_path(job_id)

    try:
        api.delete_file(
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        log_fn(f"Checkpoint deleted: {repo_id}/{path}")
        return True

    except Exception as exc:
        # File might not exist, that's fine
        log_fn(f"Could not delete checkpoint (may not exist): {exc}")
        return False


def list_checkpoints(api: "HfApi") -> list[str]:
    """
    List all checkpoint job IDs on HuggingFace Hub.

    Returns list of job_ids that have checkpoints.
    """
    repo_id = _get_checkpoint_repo()
    if not repo_id or not api:
        return []

    try:
        from huggingface_hub import list_repo_files
        from ..config import HF_CHECKPOINT_REPO_ID

        files = list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            token=api.token,
        )

        checkpoints = []
        for f in files:
            # If using dedicated repo, files are at root level
            if HF_CHECKPOINT_REPO_ID:
                if f.endswith(".pth"):
                    job_id = f[: -len(".pth")]
                    checkpoints.append(job_id)
            else:
                # If using main repo, files are in checkpoints/ subfolder
                if f.startswith("checkpoints/") and f.endswith(".pth"):
                    job_id = f[len("checkpoints/") : -len(".pth")]
                    checkpoints.append(job_id)

        return checkpoints

    except Exception:
        return []
