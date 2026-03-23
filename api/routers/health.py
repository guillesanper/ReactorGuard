"""Health and readiness probe endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Kubernetes liveness probe."""
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ready", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Kubernetes readiness probe — checks model and feature store."""
    # TODO: verify model loaded and feature store reachable
    return HealthResponse(status="ready", version="0.1.0")
