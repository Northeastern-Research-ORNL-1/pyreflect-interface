from __future__ import annotations

from fastapi import APIRouter, Header

from ..config import IS_PRODUCTION
from ..schemas import FilmLayer, GeneratorParams, TrainingParams
from ..services.limits_access import get_effective_limits
from ..services.pyreflect_runtime import PYREFLECT

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "pyreflect_available": PYREFLECT.available}


@router.get("/limits")
async def get_limits(
    x_user_id: str | None = Header(default=None),
):
    limits, access_granted, limit_source = get_effective_limits(user_id=x_user_id)
    return {
        "production": IS_PRODUCTION,
        "limits": limits,
        "access_granted": access_granted,
        "limit_source": limit_source,
    }


@router.get("/defaults")
async def get_defaults():
    return {
        "layers": [
            FilmLayer(name="substrate", sld=2.07, isld=0, thickness=0, roughness=1.8),
            FilmLayer(name="siox", sld=3.47, isld=0, thickness=12, roughness=2.0),
            FilmLayer(name="polymer_1", sld=3.8, isld=0, thickness=50, roughness=10),
            FilmLayer(name="polymer_2", sld=2.5, isld=0, thickness=150, roughness=30),
            FilmLayer(name="air", sld=0, isld=0, thickness=0, roughness=0),
        ],
        "generator": GeneratorParams(),
        "training": TrainingParams(),
    }
