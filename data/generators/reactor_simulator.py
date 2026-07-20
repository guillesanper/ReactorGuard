"""Synthetic reactor data generator.

Produces time-series sensor data with configurable fault injection.
Used for training, integration tests, and local development without a
real reactor connection.

ESTADO: ROTO. Este modulo construye el schema plano (reactor_id, core_power,
is_anomaly, fault_type) anterior al commit f130d27 y lanza ValidationError
contra el schema canonico de data/schemas/sensor_reading.py en cada llamada.
Queda excluido de mypy en pyproject.toml ([[tool.mypy.overrides]]) mientras
tanto. Se reescribe en Fase 3 (T5.3); su suite de tests
(tests/safety/test_safety_constraints.py) esta en skip por el mismo motivo.

Usage:
    python -m data.generators.reactor_simulator --config params.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]

from data.schemas.sensor_reading import SensorReading

logger = logging.getLogger(__name__)

# Nominal operating point (matches params.yaml defaults)
NOMINAL = {
    "core_power": 3000.0,
    "coolant_temp_in": 564.0,
    "coolant_temp_out": 594.0,
    "primary_pressure": 15.5,
    "coolant_flow_rate": 18000.0,
    "fuel_temp": 900.0,
    "neutron_flux_ex_core": 3.2e13,
    "steam_generator_level": 4.5,
}

# Noise standard deviations (1σ operational variability)
SIGMA = {
    "core_power": 15.0,
    "coolant_temp_in": 0.5,
    "coolant_temp_out": 0.8,
    "primary_pressure": 0.05,
    "coolant_flow_rate": 100.0,
    "fuel_temp": 5.0,
    "neutron_flux_ex_core": 1e11,
    "steam_generator_level": 0.05,
}


class ReactorSimulator:
    """Generates synthetic reactor sensor streams with fault injection.

    Args:
        n_samples: Total number of timesteps to generate.
        fault_injection_rate: Fraction of samples with faults.
        openmc_seed: Random seed for reproducibility.
        dt_seconds: Sampling interval.
    """

    def __init__(
        self,
        n_samples: int = 50000,
        fault_injection_rate: float = 0.05,
        openmc_seed: int = 42,
        dt_seconds: float = 1.0,
        reactor_id: str = "R-001",
    ) -> None:
        self.n_samples = n_samples
        self.fault_injection_rate = fault_injection_rate
        self.dt_seconds = dt_seconds
        self.reactor_id = reactor_id
        np.random.seed(openmc_seed)
        random.seed(openmc_seed)

    def generate(self) -> list[SensorReading]:
        """Generate the full dataset."""
        readings: list[SensorReading] = []
        t0 = datetime(2024, 1, 1, tzinfo=UTC)

        for i in range(self.n_samples):
            ts = t0 + timedelta(seconds=i * self.dt_seconds)
            inject_fault = random.random() < self.fault_injection_rate  # noqa: S311

            if inject_fault:
                reading = self._inject_fault(ts)
            else:
                reading = self._nominal(ts)

            readings.append(reading)

        logger.info("Generated %d samples (%d faults).", self.n_samples,
                    sum(r.is_anomaly for r in readings))
        return readings

    def _nominal(self, ts: datetime) -> SensorReading:
        values = {k: float(np.random.normal(NOMINAL[k], SIGMA[k])) for k in NOMINAL}
        # Ensure outlet > inlet after noise
        values["coolant_temp_out"] = max(
            values["coolant_temp_out"], values["coolant_temp_in"] + 0.1
        )
        return SensorReading.model_validate(
            {"timestamp": ts, "reactor_id": self.reactor_id, **values}
        )

    def _inject_fault(self, ts: datetime) -> SensorReading:
        fault_type = random.choice([  # noqa: S311
            "loss_of_coolant",
            "reactivity_insertion",
            "steam_generator_tube_rupture",
            "loss_of_feedwater",
        ])
        values = {k: float(np.random.normal(NOMINAL[k], SIGMA[k])) for k in NOMINAL}

        if fault_type == "loss_of_coolant":
            values["primary_pressure"] *= 0.7
            values["coolant_flow_rate"] *= 0.4
        elif fault_type == "reactivity_insertion":
            values["core_power"] *= 1.15
            values["neutron_flux_ex_core"] *= 1.2
            values["fuel_temp"] *= 1.1
        elif fault_type == "steam_generator_tube_rupture":
            values["steam_generator_level"] *= 0.3
            values["primary_pressure"] *= 0.92
        elif fault_type == "loss_of_feedwater":
            values["steam_generator_level"] *= 0.1
            values["coolant_temp_out"] *= 1.05

        values["coolant_temp_out"] = max(
            values["coolant_temp_out"], values["coolant_temp_in"] + 0.1
        )
        return SensorReading.model_validate(
            {"timestamp": ts, "reactor_id": self.reactor_id,
             "fault_type": fault_type, "is_anomaly": True, **values}
        )


def main(config_path: str = "params.yaml") -> None:
    with open(config_path) as f:
        params = yaml.safe_load(f)
    s = params["simulation"]

    sim = ReactorSimulator(
        n_samples=s["n_samples"],
        fault_injection_rate=s["fault_injection_rate"],
        openmc_seed=s["openmc_seed"],
        dt_seconds=s["dt_seconds"],
    )
    readings = sim.generate()

    out = Path("data/raw/simulation.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in readings:
            f.write(r.model_dump_json() + "\n")
    logger.info("Wrote %d records to %s", len(readings), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args.config)
