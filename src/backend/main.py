"""
PyReflect Interface - FastAPI Backend

Entry point for uvicorn: `uvicorn main:app`.
Application code lives under `service/`.
"""

from service import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
