from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator, Iterable


def _list_local_models(models_dir: Path) -> list[Path]:
    try:
        return list(models_dir.glob("*.pth"))
    except FileNotFoundError:
        return []


@contextmanager
def _models_dir_lock(models_dir: Path) -> Iterable[None]:
    """
    Best-effort cross-process lock for operations that need to be atomic with
    respect to MODELS_DIR contents (e.g. count-check + save).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    lock_path = models_dir / ".models.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            import fcntl  # type: ignore

            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            pass
        yield
    finally:
        try:
            import fcntl  # type: ignore

            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        os.close(fd)


@contextmanager
def models_dir_lock(models_dir: Path) -> Iterable[None]:
    """Public wrapper around the best-effort MODELS_DIR lock."""
    with _models_dir_lock(models_dir):
        yield


def _model_id_is_safe(model_id: str) -> bool:
    return bool(model_id) and "/" not in model_id and "\\" not in model_id


def _meta_path_for_model_id(models_dir: Path, model_id: str) -> Path:
    return models_dir / f"{model_id}.meta.json"


def write_model_meta(
    *,
    models_dir: Path,
    model_id: str,
    user_id: str | None,
    created_at: datetime | None = None,
    model_size_mb: float | None = None,
    source: str | None = None,
) -> None:
    if not _model_id_is_safe(model_id):
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path_for_model_id(models_dir, model_id)
    meta = {
        "model_id": model_id,
        "user_id": user_id,
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "model_size_mb": model_size_mb,
        "source": source,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def delete_local_model(*, models_dir: Path, model_id: str) -> bool:
    if not _model_id_is_safe(model_id):
        return False
    deleted_any = False
    model_path = models_dir / f"{model_id}.pth"
    meta_path = _meta_path_for_model_id(models_dir, model_id)
    try:
        if model_path.exists():
            model_path.unlink()
            deleted_any = True
    except Exception:
        pass
    try:
        if meta_path.exists():
            meta_path.unlink()
            deleted_any = True
    except Exception:
        pass
    return deleted_any


def evict_old_models_for_user(
    *,
    models_dir: Path,
    user_id: str | None,
    max_models: int,
) -> list[str]:
    """
    Ensure the number of model files for a user stays <= max_models by deleting
    the oldest models.

    Returns a list of evicted model_ids (oldest first).
    """
    if max_models <= 0:
        return []

    def parse_created_at(meta: dict, model_path: Path) -> float:
        created_at = meta.get("created_at")
        if isinstance(created_at, str):
            try:
                return datetime.fromisoformat(created_at).timestamp()
            except Exception:
                pass
        try:
            return model_path.stat().st_mtime
        except Exception:
            return 0.0

    models_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[float, str]] = []
    for meta_path in models_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta_user = meta.get("user_id")
        if user_id is None:
            if meta_user is not None:
                continue
        else:
            if meta_user != user_id:
                continue

        model_id = meta.get("model_id")
        if not isinstance(model_id, str) or not _model_id_is_safe(model_id):
            continue

        model_path = models_dir / f"{model_id}.pth"
        if not model_path.exists():
            # Stale meta; best-effort cleanup
            try:
                meta_path.unlink()
            except Exception:
                pass
            continue

        entries.append((parse_created_at(meta, model_path), model_id))

    entries.sort(key=lambda x: x[0])  # oldest first
    evicted: list[str] = []
    while len(entries) > max_models:
        _, model_id = entries.pop(0)
        if delete_local_model(models_dir=models_dir, model_id=model_id):
            evicted.append(model_id)
    return evicted


def wait_for_local_model_slot(
    *,
    models_dir: Path,
    max_models: int,
    timeout_s: float,
    poll_s: float,
    log_every_s: float = 10.0,
) -> Generator[str, None, None]:
    """
    Yield occasional log messages while waiting for MODELS_DIR to have capacity.

    If timeout_s <= 0, waits indefinitely.
    Raises TimeoutError if the timeout is exceeded.
    """
    if max_models <= 0:
        return

    start = time.monotonic()
    last_log = 0.0

    while True:
        with _models_dir_lock(models_dir):
            local_models = _list_local_models(models_dir)
            count = len(local_models)
            if count < max_models:
                return

        now = time.monotonic()
        if timeout_s > 0 and now - start >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting for local model slot ({count}/{max_models})."
            )

        if last_log == 0.0 or now - last_log >= log_every_s:
            last_log = now
            yield (
                f"Local model storage is full ({count}/{max_models}). "
                f"Waiting for a slot to free up..."
            )

        time.sleep(max(poll_s, 0.1))


def save_torch_state_dict_with_local_limit(
    *,
    torch,
    state_dict,
    model_path: Path,
    models_dir: Path,
    max_models: int,
    timeout_s: float,
    poll_s: float,
    user_id: str | None = None,
    log_every_s: float = 10.0,
) -> Generator[str, None, None]:
    """
    Atomically (best-effort) enforce capacity and save a model file.

    Capacity is enforced per-user when `user_id` is provided and model meta
    files are present.
    """
    start = time.monotonic()
    last_log = 0.0

    model_id = model_path.stem
    if not _model_id_is_safe(model_id):
        raise ValueError("Invalid model_path (unsafe model_id)")

    while True:
        evicted_ids: list[str] = []
        with models_dir_lock(models_dir):
            if max_models > 0:
                if user_id is not None:
                    evicted_ids = evict_old_models_for_user(
                        models_dir=models_dir, user_id=user_id, max_models=max_models - 1
                    )
                else:
                    # Fallback: global eviction based purely on count.
                    local_models = _list_local_models(models_dir)
                    local_models.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
                    while len(local_models) >= max_models:
                        victim = local_models.pop(0)
                        if victim.suffix == ".pth":
                            delete_local_model(models_dir=models_dir, model_id=victim.stem)
                            evicted_ids.append(victim.stem)

            # Save the model.
            torch.save(state_dict, model_path)
            try:
                size_mb = model_path.stat().st_size / (1024 * 1024)
            except Exception:
                size_mb = None
            write_model_meta(
                models_dir=models_dir,
                model_id=model_id,
                user_id=user_id,
                model_size_mb=size_mb,
                source="local",
            )

        for victim_id in evicted_ids:
            yield f"Evicted old model to free space: {victim_id}.pth"
        return

        now = time.monotonic()
        if timeout_s > 0 and now - start >= timeout_s:
            raise TimeoutError("Timed out waiting to save model locally.")

        if last_log == 0.0 or now - last_log >= log_every_s:
            last_log = now
            yield "Waiting to save model..."
        time.sleep(max(poll_s, 0.1))


def drain_wait_logs(wait_iter: Iterable[str], log: Callable[[str], None]) -> None:
    """Convenience for non-streaming contexts (e.g. RQ jobs)."""
    for message in wait_iter:
        log(message)
