from __future__ import annotations

import os
import time
from contextlib import contextmanager
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
    log_every_s: float = 10.0,
) -> Generator[str, None, None]:
    """
    Atomically (best-effort) wait for capacity and save a model file.

    This avoids races where multiple processes pass a pre-check concurrently and
    then exceed the max when saving.
    """
    start = time.monotonic()
    last_log = 0.0

    while True:
        with _models_dir_lock(models_dir):
            local_models = _list_local_models(models_dir)
            count = len(local_models)
            if max_models <= 0 or count < max_models:
                torch.save(state_dict, model_path)
                return

        now = time.monotonic()
        if timeout_s > 0 and now - start >= timeout_s:
            raise TimeoutError(
                f"Timed out waiting to save model locally ({count}/{max_models})."
            )

        if last_log == 0.0 or now - last_log >= log_every_s:
            last_log = now
            yield (
                f"Local model storage is full ({count}/{max_models}). "
                "Waiting to save model..."
            )

        time.sleep(max(poll_s, 0.1))


def drain_wait_logs(wait_iter: Iterable[str], log: Callable[[str], None]) -> None:
    """Convenience for non-streaming contexts (e.g. RQ jobs)."""
    for message in wait_iter:
        log(message)

