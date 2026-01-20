from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas import GenerateRequest, GenerateResponse, validate_limits
from ..services.limits_access import get_effective_limits
from ..services.pyreflect_runtime import PYREFLECT
from ..services.real_data import generate_with_real_data_streaming
from ..services.synthetic import generate_with_pyreflect, generate_with_pyreflect_streaming

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    x_user_id: str | None = Header(default=None),
):
    if request.dataSource == "real":
        raise HTTPException(
            status_code=400, detail="Real data mode is only supported on /api/generate/stream"
        )
    effective_limits, _, _ = get_effective_limits(user_id=x_user_id)
    validate_limits(request.generator, request.training, limits=effective_limits)
    if not PYREFLECT.available:
        raise HTTPException(
            status_code=503,
            detail="pyreflect not available. Please install pyreflect dependencies.",
        )
    try:
        return generate_with_pyreflect(request.layers, request.generator, request.training)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    http_request: Request,
    x_user_id: str | None = Header(default=None),
):
    mongo = getattr(http_request.app.state, "mongo", None)
    hf = getattr(http_request.app.state, "hf", None)
    mongo_generations = getattr(mongo, "generations", None) if mongo else None

    if request.dataSource == "real":
        return StreamingResponse(
            generate_with_real_data_streaming(
                request, user_id=x_user_id, mongo_generations=mongo_generations, hf=hf
            ),
            media_type="text/event-stream",
        )

    effective_limits, _, _ = get_effective_limits(user_id=x_user_id)
    validate_limits(request.generator, request.training, limits=effective_limits)
    if not PYREFLECT.available:
        def error_stream():
            yield 'event: error\ndata: "pyreflect not available. Please install pyreflect dependencies."\n\n'

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    return StreamingResponse(
        generate_with_pyreflect_streaming(
            layers=request.layers,
            gen_params=request.generator,
            train_params=request.training,
            user_id=x_user_id,
            name=request.name,
            mongo_generations=mongo_generations,
            hf=hf,
        ),
        media_type="text/event-stream",
    )
