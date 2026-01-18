from __future__ import annotations


def create_app():
    # Lazy import so `import service.jobs` (RQ workers) does not pull in the
    # full FastAPI stack and its dependencies unless needed.
    from .app import create_app as _create_app

    return _create_app()


__all__ = ["create_app"]
