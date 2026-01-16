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


def upload_model(hf: HuggingFaceIntegration, file_path: Path, model_id: str) -> bool:
    if not hf.available or not hf.api or not hf.repo_id:
        return False
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {model_id}.pth ({file_size_mb:.2f} MB) to HF Hub...")

        try:
            hf.api.create_repo(
                repo_id=hf.repo_id, repo_type="dataset", exist_ok=True, private=False
            )
        except Exception:
            pass

        hf.api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{model_id}.pth",
            repo_id=hf.repo_id,
            repo_type="dataset",
        )
        return True
    except Exception as exc:
        print(f"Error uploading to Hugging Face: {exc}")
        return False


def delete_model_file(hf: HuggingFaceIntegration, model_id: str) -> bool:
    if not hf.available or not hf.api or not hf.repo_id:
        return False
    try:
        hf.api.delete_file(
            repo_id=hf.repo_id,
            path_in_repo=f"{model_id}.pth",
            repo_type="dataset",
        )
        return True
    except Exception as exc:
        print(f"Warning: Failed to delete HF model file: {exc}")
        return False


def get_remote_model_info(hf: HuggingFaceIntegration, model_id: str) -> dict[str, Any]:
    if not hf.available or not hf.api or not hf.repo_id:
        return {"size_mb": None, "source": "unknown"}
    try:
        paths = hf.api.get_paths_info(
            repo_id=hf.repo_id, paths=[f"{model_id}.pth"], repo_type="dataset"
        )
        if paths:
            size_mb = paths[0].size / (1024 * 1024)
            return {"size_mb": size_mb, "source": "huggingface"}
    except Exception as exc:
        print(f"HF Check failed: {exc}")
    return {"size_mb": None, "source": "unknown"}

