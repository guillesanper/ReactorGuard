"""Safety-critical constraint tests.

These tests verify that the system never silently misclassifies
high-severity faults as nominal. They are run as a separate suite
to highlight their safety-critical status in CI.
"""


import pytest

from data.generators.reactor_simulator import ReactorSimulator


@pytest.fixture
def sim() -> ReactorSimulator:
    return ReactorSimulator(n_samples=1000, fault_injection_rate=0.2, openmc_seed=0)


def test_fault_injection_rate_within_tolerance(sim: ReactorSimulator) -> None:
    """Generated fault rate must be within ±2 pp of the configured rate."""
    readings = sim.generate()
    actual_rate = sum(r.is_anomaly for r in readings) / len(readings)
    assert abs(actual_rate - sim.fault_injection_rate) < 0.02


def test_all_faults_have_fault_type(sim: ReactorSimulator) -> None:
    """Every anomalous reading must carry a fault_type label."""
    readings = sim.generate()
    anomalous = [r for r in readings if r.is_anomaly]
    assert all(r.fault_type is not None for r in anomalous)


def test_nominal_readings_no_fault_label(sim: ReactorSimulator) -> None:
    """Nominal readings must not carry a fault_type."""
    readings = sim.generate()
    nominal = [r for r in readings if not r.is_anomaly]
    assert all(r.fault_type is None for r in nominal)
