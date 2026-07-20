"""Anomaly prediction endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from data.schemas.sensor_reading import SensorReading

router = APIRouter()


class PredictionResponse(BaseModel):
    """Model prediction with uncertainty bounds."""

    timestamp: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Anomaly probability")
    is_anomaly: bool
    prediction_interval_low: float
    prediction_interval_high: float
    confidence_level: float
    fault_type: str | None = None


@router.post("/", response_model=PredictionResponse)
async def predict(reading: SensorReading) -> PredictionResponse:
    """Run anomaly detection on a single sensor reading."""
    # TODO: extract features, run PINN inference, apply MAPIE intervals
    raise HTTPException(status_code=501, detail="Model not yet loaded")
