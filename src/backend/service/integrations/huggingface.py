from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HuggingFaceIntegration:
    available: bool
    repo_id: str | None
    api: Any | None


def init_huggingface(token: str | None, repo_id: str | None) -> HuggingFaceIntegration:
    if not token or not repo_id:
        return HuggingFaceIntegration(available=False, repo_id=repo_id, api=None)
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print(f"Hugging Face API initialized (Repo: {repo_id})")
        return HuggingFaceIntegration(available=True, repo_id=repo_id, api=api)
    except Exception as exc:
        print(f"Warning: Failed to initialize Hugging Face API: {exc}")
        return HuggingFaceIntegration(available=False, repo_id=repo_id, api=None)


def upload_model(hf: HuggingFaceIntegration, file_path: Any, model_id: str) -> bool:
    if not hf.available or not hf.api or not hf.repo_id:
        return False
    try:
        file_size_mb = None
        try:
            if isinstance(file_path, (str, Path)):
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            else:
                # Try to infer size for file-like objects.
                pos = file_path.tell()
                file_path.seek(0, 2)
                file_size_mb = file_path.tell() / (1024 * 1024)
                file_path.seek(pos)
        except Exception:
            file_size_mb = None

        if file_size_mb is not None:
            print(f"Uploading {model_id}.pth ({file_size_mb:.2f} MB) to HF Hub...")
        else:
            print(f"Uploading {model_id}.pth to HF Hub...")

        try:
            hf.api.create_repo(
                repo_id=hf.repo_id, repo_type="dataset", exist_ok=True, private=False
            )
        except Exception:
            pass

        try:
            # Ensure file-like objects start from the beginning.
            if not isinstance(file_path, (str, Path)):
                file_path.seek(0)
        except Exception:
            pass

        # Use folder structure: models/{model_id}/{model_id}.pth
        hf.api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"models/{model_id}/{model_id}.pth",
            repo_id=hf.repo_id,
            repo_type="dataset",
        )
        return True
    except Exception as exc:
        print(f"Error uploading to Hugging Face: {exc}")
        return False


def upload_training_data(
    hf: HuggingFaceIntegration,
    nr_curves: Any,
    sld_curves: Any,
    model_id: str,
) -> bool:
    """
    Upload training data (.npy files) to HF in the model's folder.
    
    Files are saved as:
    - models/{model_id}/nr_train.npy
    - models/{model_id}/sld_train.npy
    """
    if not hf.available or not hf.api or not hf.repo_id:
        return False
    
    try:
        import io
        import numpy as np
        
        # Ensure repo exists
        try:
            hf.api.create_repo(
                repo_id=hf.repo_id, repo_type="dataset", exist_ok=True, private=False
            )
        except Exception:
            pass
        
        # Upload NR curves
        nr_buffer = io.BytesIO()
        np.save(nr_buffer, nr_curves)
        nr_size_mb = nr_buffer.tell() / (1024 * 1024)
        nr_buffer.seek(0)
        
        print(f"Uploading nr_train.npy ({nr_size_mb:.2f} MB) to HF Hub...")
        hf.api.upload_file(
            path_or_fileobj=nr_buffer,
            path_in_repo=f"models/{model_id}/nr_train.npy",
            repo_id=hf.repo_id,
            repo_type="dataset",
        )
        
        # Upload SLD curves
        sld_buffer = io.BytesIO()
        np.save(sld_buffer, sld_curves)
        sld_size_mb = sld_buffer.tell() / (1024 * 1024)
        sld_buffer.seek(0)
        
        print(f"Uploading sld_train.npy ({sld_size_mb:.2f} MB) to HF Hub...")
        hf.api.upload_file(
            path_or_fileobj=sld_buffer,
            path_in_repo=f"models/{model_id}/sld_train.npy",
            repo_id=hf.repo_id,
            repo_type="dataset",
        )
        
        print(f"Training data uploaded to models/{model_id}/")
        return True
    except Exception as exc:
        print(f"Error uploading training data to Hugging Face: {exc}")
        return False


def download_training_data(
    hf: HuggingFaceIntegration, model_id: str
) -> tuple[Any, Any] | None:
    """
    Download existing training data (.npy files) from HuggingFace.
    
    Used by retry jobs to reuse previously generated training data instead
    of regenerating from scratch.
    
    Args:
        hf: HuggingFace integration instance
        model_id: The model ID whose training data to download
        
    Returns:
        Tuple of (nr_curves, sld_curves) as numpy arrays, or None if not found
    """
    if not hf.available or not hf.api or not hf.repo_id:
        return None
    
    try:
        import numpy as np
        from huggingface_hub import hf_hub_download
        
        nr_path = f"models/{model_id}/nr_train.npy"
        sld_path = f"models/{model_id}/sld_train.npy"
        
        # Check if files exist first
        try:
            if hasattr(hf.api, "file_exists"):
                nr_exists = hf.api.file_exists(
                    repo_id=hf.repo_id, filename=nr_path, repo_type="dataset"
                )
                sld_exists = hf.api.file_exists(
                    repo_id=hf.repo_id, filename=sld_path, repo_type="dataset"
                )
                if not nr_exists or not sld_exists:
                    print(f"Training data not found on HF for {model_id}")
                    return None
        except Exception:
            # If file_exists check fails, try downloading anyway
            pass
        
        print(f"Downloading nr_train.npy from HF for {model_id}...")
        nr_local = hf_hub_download(
            repo_id=hf.repo_id,
            filename=nr_path,
            repo_type="dataset",
        )
        
        print(f"Downloading sld_train.npy from HF for {model_id}...")
        sld_local = hf_hub_download(
            repo_id=hf.repo_id,
            filename=sld_path,
            repo_type="dataset",
        )
        
        nr_curves = np.load(nr_local)
        sld_curves = np.load(sld_local)
        
        print(f"Loaded training data: NR shape={nr_curves.shape}, SLD shape={sld_curves.shape}")
        return nr_curves, sld_curves
        
    except Exception as exc:
        print(f"Failed to download training data from HF: {exc}")
        return None


def delete_model_file(hf: HuggingFaceIntegration, model_id: str) -> bool:
    if not hf.available or not hf.api or not hf.repo_id:
        return False
    try:
        hf.api.delete_file(
            repo_id=hf.repo_id,
            path_in_repo=f"models/{model_id}/{model_id}.pth",
            repo_type="dataset",
        )
        return True
    except Exception as exc:
        print(f"Warning: Failed to delete HF model file: {exc}")
        return False


def get_remote_model_info(hf: HuggingFaceIntegration, model_id: str) -> dict[str, Any]:
    """
    Get information about a model file on Hugging Face.
    
    Returns dict with:
    - size_mb: file size in MB (None if not found or error)
    - source: "huggingface" if HF is configured, "unknown" otherwise
    - error: error message if something went wrong (optional)
    """
    if not hf.available or not hf.api or not hf.repo_id:
        return {"size_mb": None, "source": "unknown", "error": "HF not available"}
    
    file_path = f"models/{model_id}/{model_id}.pth"
    
    # Try multiple methods to check if file exists and get its size
    # Method 1: Use file_exists (simpler, but doesn't give size)
    try:
        if hasattr(hf.api, "file_exists"):
            exists = hf.api.file_exists(
                repo_id=hf.repo_id,
                filename=file_path,
                repo_type="dataset",
            )
            if not exists:
                return {"size_mb": None, "source": "huggingface", "error": "not_found"}
    except Exception as exc:
        # If file_exists fails, try get_paths_info as fallback
        print(f"HF file_exists check failed for {model_id}, trying get_paths_info: {exc}")
    
    # Method 2: Use get_paths_info (gives size information)
    try:
        paths = hf.api.get_paths_info(
            repo_id=hf.repo_id, paths=[file_path], repo_type="dataset"
        )
        if paths and len(paths) > 0:
            size_mb = paths[0].size / (1024 * 1024)
            return {"size_mb": size_mb, "source": "huggingface"}
        # Model not found on HF
        return {"size_mb": None, "source": "huggingface", "error": "not_found"}
    except Exception as exc:
        error_msg = str(exc)
        error_type = type(exc).__name__
        print(f"HF get_paths_info failed for {model_id} (repo={hf.repo_id}): {error_type}: {error_msg}", flush=True)
        
        # Provide more specific error information
        if "401" in error_msg or "Unauthorized" in error_msg:
            error_detail = "authentication_failed"
        elif "403" in error_msg or "Forbidden" in error_msg:
            error_detail = "access_denied"
        elif "404" in error_msg or "Not Found" in error_msg:
            error_detail = "not_found"
        elif "timeout" in error_msg.lower() or "Timeout" in error_msg:
            error_detail = "timeout"
        else:
            error_detail = error_msg[:200]  # Truncate long error messages
        
        return {"size_mb": None, "source": "huggingface", "error": error_detail}
