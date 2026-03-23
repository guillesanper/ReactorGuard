"""Anomaly prediction endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class SensorReading(BaseModel):
    """Single timestep of reactor sensor data."""

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    core_power: float = Field(..., description="Reactor core thermal power [MWth]")
    coolant_temp_in: float = Field(..., description="Coolant inlet temperature [K]")
    coolant_temp_out: float = Field(..., description="Coolant outlet temperature [K]")
    primary_pressure: float = Field(..., description="Primary circuit pressure [MPa]")
    coolant_flow_rate: float = Field(..., description="Coolant mass flow rate [kg/s]")
    fuel_temp: float = Field(..., description="Fuel centreline temperature [K]")
    neutron_flux_ex_core: float = Field(..., description="Ex-core neutron flux [n/cm²·s]")
    steam_generator_level: float = Field(..., description="Steam generator water level [m]")


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
