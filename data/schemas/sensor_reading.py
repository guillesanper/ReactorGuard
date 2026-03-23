"""Pydantic schemas for reactor sensor data (shared by API, ingestion, and generators)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class SensorReading(BaseModel):
    """Single timestep of raw reactor sensor data as produced by SCADA/Kafka."""

    timestamp: datetime
    reactor_id: str = Field(..., description="Unique reactor identifier")

    # Core parameters
    core_power: float = Field(..., ge=0.0, le=4000.0, description="Thermal power [MWth]")
    neutron_flux_ex_core: float = Field(..., ge=0.0, description="Ex-core flux [n/cm²·s]")

    # Thermal-hydraulics
    coolant_temp_in: float = Field(..., ge=273.15, le=700.0, description="Inlet temperature [K]")
    coolant_temp_out: float = Field(..., ge=273.15, le=700.0, description="Outlet temperature [K]")
    primary_pressure: float = Field(..., ge=0.0, le=20.0, description="Primary pressure [MPa]")
    coolant_flow_rate: float = Field(..., ge=0.0, description="Mass flow rate [kg/s]")

    # Fuel
    fuel_temp: float = Field(..., ge=273.15, le=3000.0, description="Fuel centreline temp [K]")

    # Secondary circuit
    steam_generator_level: float = Field(..., ge=0.0, le=10.0, description="SG level [m]")

    # Optional fault label (None during online inference, set during simulation)
    fault_type: str | None = None
    is_anomaly: bool = False

    @model_validator(mode="after")
    def check_temperature_gradient(self) -> SensorReading:
        if self.coolant_temp_out < self.coolant_temp_in:
            raise ValueError(
                f"Outlet temp ({self.coolant_temp_out} K) must be >= inlet temp "
                f"({self.coolant_temp_in} K) for normal flow direction."
            )
        return self


class SensorBatch(BaseModel):
    """Batch of sensor readings (e.g. from Kafka consumer poll)."""

    readings: list[SensorReading]
    source_topic: str
    partition: int
    offset_start: int
