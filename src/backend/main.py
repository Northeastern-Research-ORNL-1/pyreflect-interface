"""
PyReflect Interface - FastAPI Backend

This repository now uses the modular backend under `src/backend/service/`.
This file is intentionally a thin uvicorn entrypoint wrapper so the common
command still works:

  uv run uvicorn main:app --reload --port 8000
"""

from service import create_app

app = create_app()

