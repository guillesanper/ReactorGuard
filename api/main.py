"""ReactorGuard FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app

from api.routers import explain, health, predict


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    # Startup: load model, warm up feature store connection
    yield
    # Shutdown: flush telemetry, close connections


app = FastAPI(
    title="ReactorGuard API",
    description="Anomaly detection API for nuclear reactor sensor streams.",
    version="0.1.0",
    lifespan=lifespan,
)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Register routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(explain.router, prefix="/explain", tags=["explain"])
