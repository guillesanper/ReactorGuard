"""Unit tests for data.simulation.scenario_planner.

Cubren el reparto por maximo resto, la configuracion de topes, la capacidad por
tipo, el presupuesto de D11 (cuotas por severidad y runs), la busqueda de la tasa
maxima alcanzable y el plan de runs: composicion, invariantes de cada plan,
determinismo y el `ValueError` con cifras cuando la cuota no cabe.

El planificador no integra el modelo: trabaja con `SimulationParams`, el catalogo
y los niveles, asi que estos tests son baratos y usan la configuracion versionada.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.simulation.plant_params import load_plant_params
from data.simulation.scenario import (
    FAULT_KINDS,
    FAULT_SEVERITY,
    FaultAxis,
    FaultLevel,
    InstrumentFaultKind,
    ScenarioKind,
)
from data.simulation.scenario_planner import (
    PLANNER_STREAM_KEY,
    PlannerSettings,
    ScenarioConfig,
    _free_sensor,
    _place_spike,
    _plan_instrument_episodes,
    apportion,
    compute_budget,
    max_feasible_rate,
    parse_planner_settings,
    plan_dataset,
    plan_runs,
    planner_rng,
    type_capacity,
)
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.simulation_params import SimulationParams, load_simulation_params
from tests.support.sim_frames import CONFIGS, build_catalog, write_sim_params

CAPACITY_B = {"critical": 0.55, "high": 0.38, "medium": 0.06, "low": 0.01}
"""Weights close to what the instrument types can label (feasible up to ~15.7 %)."""

RATE = 0.12


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned sensor catalog (27 sensors, 6 types)."""
    return build_catalog()


@pytest.fixture(scope="module")
def config() -> ScenarioConfig:
    """Return the versioned fault levels and planner caps."""
    return ScenarioConfig.from_plant(load_plant_params(CONFIGS / "plant_parameters.yaml"))


@pytest.fixture()
def params(tmp_path: Path) -> SimulationParams:
    """Return the dataset-sized configuration: 34 runs of 1500 steps, 40 % rate."""
    return load_simulation_params(
        write_sim_params(tmp_path, n_samples=50000, run_length_steps=1500, openmc_seed=42)
    )


class TestApportion:
    """Largest-remainder (Hamilton) apportionment of an integer."""

    def test_sums_to_total_by_largest_remainder(self) -> None:
        """The leftover units go to the largest fractions, ties to the lowest index."""
        assert apportion(10, [1, 1, 1]) == [4, 3, 3]
        assert apportion(10, [1, 2, 3]) == [2, 3, 5]
        assert apportion(7, [0.5, 0.5]) == [4, 3]

    def test_minimum_is_guaranteed(self) -> None:
        """Every part gets the minimum before the proportional share."""
        assert apportion(7, [1, 1], minimum=2) == [4, 3]
        assert apportion(6, [0, 5], minimum=2) == [2, 4]

    @pytest.mark.parametrize(
        ("total", "weights", "minimum", "match"),
        [
            (5, [], 0, "at least one"),
            (5, [1.0, -1.0], 0, "non-negative"),
            (5, [0.0, 0.0], 0, "positive sum"),
            (5, [1.0, float("nan")], 0, "non-negative"),
            (3, [1.0, 1.0], 2, "cannot give"),
        ],
    )
    def test_invalid(self, total: int, weights: list[float], minimum: int, match: str) -> None:
        """Bad weights or an unreachable minimum are errors."""
        with pytest.raises(ValueError, match=match):
            apportion(total, weights, minimum)

    @pytest.mark.parametrize("total", [0, 1, 17, 1000])
    def test_always_sums_to_total(self, total: int) -> None:
        """The sum is exact for any total."""
        assert sum(apportion(total, [0.3, 0.9, 0.05, 0.4])) == total


class TestSettings:
    """`scenario_planner:` resolves to typed caps or fails naming the key."""

    def test_versioned_block(self, config: ScenarioConfig) -> None:
        """The versioned file gives the documented caps."""
        planner = config.planner
        assert planner.n_min_runs == 3
        assert planner.min_normal_runs == 2
        assert 0.0 < planner.max_sensor_fraction <= 1.0
        assert planner.coverage_gate == 0.95

    @pytest.mark.parametrize(
        ("key", "value", "error", "match"),
        [
            (None, None, KeyError, "n_min_runs"),
            ("n_min_runs", 0, ValueError, "whole number"),
            ("n_min_runs", 2.5, ValueError, "whole number"),
            ("min_normal_runs", -1, ValueError, "whole number"),
            ("max_sensor_fraction", 1.5, ValueError, r"\(0, 1\]"),
            ("max_window_fraction", 0.0, ValueError, r"\(0, 1\]"),
            ("max_spike_rate", "x", ValueError, "number"),
            ("coverage_gate", float("inf"), ValueError, "finite"),
        ],
    )
    def test_invalid(self, key: str | None, value: Any, error: type[Exception], match: str) -> None:
        """Each malformed value (or a missing key) is named."""
        raw = dict(load_plant_params(CONFIGS / "plant_parameters.yaml").scenario_planner)
        if key is None:
            del raw["n_min_runs"]
        else:
            raw[key] = value
        with pytest.raises(error, match=match):
            parse_planner_settings(raw)


class TestPlannerRng:
    """The planner has its own stream, independent of every run's."""

    def test_deterministic_and_independent_of_the_runs(self, params: SimulationParams) -> None:
        """Same params, same stream; and it differs from the stream of any run."""
        a = planner_rng(params).random(4)
        b = planner_rng(params).random(4)
        assert np.array_equal(a, b)
        assert PLANNER_STREAM_KEY > params.n_runs
        seeds = np.random.SeedSequence(params.openmc_seed).spawn(params.n_runs)
        for child in seeds:
            assert not np.array_equal(np.random.default_rng(child).random(4), a)

    def test_depends_on_the_master_seed(self, params: SimulationParams) -> None:
        """Another master seed, another plan stream."""
        other = dataclasses.replace(params, openmc_seed=43)
        assert not np.array_equal(planner_rng(params).random(4), planner_rng(other).random(4))


class TestCapacity:
    """What one run of each type can label under the density caps."""

    def test_process_is_all_sensors_times_the_window(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """A process fault labels every sensor of its window, up to max_window_fraction."""
        cap = type_capacity(ScenarioKind.CORE_TEMP_EXCURSION, params, catalog, config)
        n_sensors = len(catalog.sensors)
        window = int(config.planner.max_window_fraction * params.run_length_steps)
        assert cap.max_readings == n_sensors * window
        assert cap.min_readings == n_sensors * 151
        assert cap.max_episodes == 1
        assert cap.fraction(n_sensors * params.run_length_steps) == pytest.approx(
            config.planner.max_window_fraction, abs=1e-3
        )

    def test_spike_capacity_is_sensors_times_rate(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """Spikes: at most max_sensor_fraction of the sensors x max_spike_rate x steps."""
        cap = type_capacity(ScenarioKind.NOISE_SPIKE, params, catalog, config)
        sensors = int(config.planner.max_sensor_fraction * len(catalog.sensors))
        per_sensor = int(config.planner.max_spike_rate * params.run_length_steps)
        assert cap.max_readings == sensors * per_sensor
        assert cap.mean_window == 1.0

    def test_instrument_types_label_far_less_than_process(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """The structural reason the 40 % is out of reach: one sensor per episode."""
        process = type_capacity(ScenarioKind.VOID_FORMATION, params, catalog, config)
        for kind in (
            ScenarioKind.DRIFT,
            ScenarioKind.STUCK,
            ScenarioKind.NOISE_SPIKE,
            ScenarioKind.BIAS,
            ScenarioKind.INTERMITTENT,
        ):
            assert type_capacity(kind, params, catalog, config).max_readings < process.max_readings

    def test_bias_mean_window_uses_the_onset_range(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """A bias runs from its onset to the end: mean (1 - onset midpoint) x steps."""
        cap = type_capacity(ScenarioKind.BIAS, params, catalog, config)
        low, high = config.instrument.for_kind(InstrumentFaultKind.BIAS).onset_fraction
        assert cap.mean_window == pytest.approx(1500 * (1 - 0.5 * (low + high)))

    def test_normal_is_not_a_fault_type(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """Normal operation has no capacity."""
        with pytest.raises(KeyError, match="normal"):
            type_capacity(ScenarioKind.NORMAL, params, catalog, config)

    def test_window_cap_below_the_default_window(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """A cap that cannot hold the default fault window is a configuration error."""
        tight = dataclasses.replace(
            config, planner=dataclasses.replace(config.planner, max_window_fraction=0.05)
        )
        with pytest.raises(ValueError, match="max_window_fraction"):
            type_capacity(ScenarioKind.VOID_FORMATION, params, catalog, tight)


class TestBudget:
    """The D11 budget: quotas by severity and the runs they cost."""

    def test_quotas_are_weight_times_target(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """quota = weight x rate x readings, and the weights sum to the target."""
        budget = compute_budget(params, catalog, config)
        assert budget.total_readings == 34 * 1500 * 27
        assert budget.target_readings == pytest.approx(0.40 * budget.total_readings)
        by = {s.severity.value: s for s in budget.severities}
        assert by["CRITICAL"].quota_readings == pytest.approx(0.30 * budget.target_readings)
        assert sum(s.quota_readings for s in budget.severities) == pytest.approx(
            budget.target_readings
        )
        assert {k.value for s in budget.severities for k in s.kinds} == {
            k.value for k in FAULT_KINDS
        }

    def test_d11_defaults_are_unreachable(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """40 % with the D11 weights needs many times the runs there are."""
        budget = compute_budget(params, catalog, config)
        assert not budget.feasible
        assert budget.runs_needed > 5 * budget.runs_available
        assert budget.runs_available == params.n_runs - config.planner.min_normal_runs

    def test_runs_floor_at_n_min_runs(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """A tiny quota still costs n_min_runs per type."""
        budget = compute_budget(params, catalog, config, fault_injection_rate=0.001)
        assert all(s.runs_per_kind == config.planner.n_min_runs for s in budget.severities)
        assert all(s.capped_by_minimum for s in budget.severities)
        assert budget.runs_needed == 8 * config.planner.n_min_runs

    @pytest.mark.parametrize("rate", [0.0, 1.0, -0.1])
    def test_rate_must_be_inside_zero_one(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig, rate: float
    ) -> None:
        """The rate is a fraction of the readings."""
        with pytest.raises(ValueError, match="fault_injection_rate"):
            compute_budget(params, catalog, config, fault_injection_rate=rate)

    def test_weights_must_be_the_four_severities_summing_to_one(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """A weighting that does not sum to 1 is rejected."""
        with pytest.raises(ValueError, match="summing to 1"):
            compute_budget(
                params,
                catalog,
                config,
                severity_weights={"critical": 0.5, "high": 0.5, "medium": 0.5, "low": 0.5},
            )


class TestMaxFeasibleRate:
    """The largest rate the planner can satisfy with given weights."""

    def test_is_the_boundary_of_feasibility(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """Feasible at the returned rate, infeasible a hair above it."""
        rate = max_feasible_rate(params, catalog, config, CAPACITY_B)
        assert 0.10 < rate < 0.25
        inside = compute_budget(
            params, catalog, config, fault_injection_rate=rate, severity_weights=CAPACITY_B
        )
        outside = compute_budget(
            params, catalog, config, fault_injection_rate=rate * 1.02, severity_weights=CAPACITY_B
        )
        assert inside.feasible
        assert not outside.feasible

    def test_d11_weights_reach_only_a_few_percent(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """With 0.25 / 0.15 on MEDIUM / LOW the dataset can label ~1.4 %, not 40 %."""
        rate = max_feasible_rate(params, catalog, config)
        assert 0.005 < rate < 0.03

    def test_zero_when_the_minimum_runs_do_not_fit(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """Fewer runs than 8 types x n_min_runs: no rate is feasible."""
        small = dataclasses.replace(params, n_samples=10 * 1500)
        assert max_feasible_rate(small, catalog, config) == 0.0

    def test_a_bigger_dataset_does_not_rescue_the_budget(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """Quotas and capacities scale with the runs: 100x more runs only drop the overhead."""
        base = max_feasible_rate(params, catalog, config, CAPACITY_B)
        big = dataclasses.replace(params, n_samples=params.n_samples * 100)
        scaled = max_feasible_rate(big, catalog, config, CAPACITY_B)
        assert base < scaled < 0.25


class TestPlanRuns:
    """The planned dataset: composition, per-run invariants and determinism."""

    @pytest.fixture()
    def planned(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> Any:
        """Return a feasible plan (12 % with weights close to the capacities)."""
        return plan_dataset(
            params,
            catalog,
            planner_rng(params),
            config,
            fault_injection_rate=RATE,
            severity_weights=CAPACITY_B,
        )

    def test_default_configuration_fails_with_the_numbers(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """40 % with the D11 weights is refused, and the message carries the evidence."""
        with pytest.raises(ValueError, match="unreachable") as excinfo:
            plan_runs(params, catalog, planner_rng(params), config)
        text = str(excinfo.value)
        for needle in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "1377000", "largest rate", "D11"):
            assert needle in text

    def test_critical_floor_is_checked(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """weight x rate below critical_min_fraction is a configuration error."""
        with pytest.raises(ValueError, match="critical_min_fraction"):
            plan_runs(
                params,
                catalog,
                planner_rng(params),
                config,
                fault_injection_rate=0.05,
                severity_weights=CAPACITY_B,
            )

    def test_composition(
        self, planned: Any, config: ScenarioConfig, params: SimulationParams
    ) -> None:
        """34 runs: every type at least n_min_runs times, normal runs reserved."""
        counts = Counter(p.scenario for p in planned.plans)
        assert len(planned.plans) == params.n_runs
        assert all(counts[k] >= config.planner.n_min_runs for k in FAULT_KINDS)
        assert counts[ScenarioKind.NORMAL] >= config.planner.min_normal_runs
        assert sum(counts.values()) == params.n_runs
        assert planned.by_scenario == {
            k.value: v for k, v in sorted(counts.items(), key=lambda kv: kv[0].value)
        }

    def test_runs_are_contiguous_and_independently_seeded(
        self, planned: Any, params: SimulationParams
    ) -> None:
        """Run i starts at start_time + i x duration and owns spawn_key (i,)."""
        for index, plan in enumerate(planned.plans):
            assert plan.run_index == index
            assert plan.spawn_key == (index,)
            assert plan.seed_entropy == params.openmc_seed
            assert plan.start_time == params.start_time + timedelta(
                seconds=index * params.run_duration_s
            )
        assert len({p.run_id for p in planned.plans}) == len(planned.plans)

    def test_labelled_rate_and_severity_split(self, planned: Any, params: SimulationParams) -> None:
        """The labelled share is the target within 2 pp; severities follow the weights."""
        assert abs(planned.labelled_fraction - RATE) < 0.02
        total = planned.budget.total_readings
        shares = {k: v / total for k, v in planned.readings_by_severity().items()}
        for severity, key in (("CRITICAL", "critical"), ("HIGH", "high")):
            assert shares[severity] == pytest.approx(CAPACITY_B[key] * RATE, rel=0.25)
        assert shares["CRITICAL"] >= params.critical_min_fraction

    def test_every_type_has_all_three_levels(self, planned: Any) -> None:
        """Levels rotate: no type is planned with a missing level."""
        for kind in FAULT_KINDS:
            assert all(planned.kinds[kind].levels[lv.value] >= 1 for lv in FaultLevel)

    def test_process_windows_respect_the_cap_and_the_warmup(
        self, planned: Any, params: SimulationParams, config: ScenarioConfig
    ) -> None:
        """A process fault starts after the warm-up and spans at most the window cap."""
        cap = config.planner.max_window_fraction * params.run_length_steps
        for plan in planned.plans:
            spec = plan.process_fault
            if spec is None:
                continue
            assert spec.injected_step >= params.warmup_steps
            assert spec.last_active_step is not None
            assert spec.last_active_step - spec.injected_step + 1 <= cap

    def test_instrument_episodes_respect_the_sensor_caps(
        self,
        planned: Any,
        params: SimulationParams,
        catalog: SensorCatalog,
        config: ScenarioConfig,
    ) -> None:
        """Distinct sensors per run (except spikes), at most the sensor cap, after the warm-up."""
        cap = max(1, int(config.planner.max_sensor_fraction * len(catalog.sensors)))
        per_sensor = int(config.planner.max_spike_rate * params.run_length_steps)
        for plan in planned.plans:
            if plan.scenario.axis is not FaultAxis.INSTRUMENT:
                continue
            sensors = Counter(e.sensor_id for e in plan.instrument_faults)
            assert 1 <= len(sensors) <= cap
            assert set(sensors) <= set(catalog.tags)
            if plan.scenario is ScenarioKind.NOISE_SPIKE:
                assert max(sensors.values()) <= per_sensor
            else:
                assert max(sensors.values()) == 1
            assert all(e.start_step >= params.warmup_steps for e in plan.instrument_faults)
            assert all(e.kind.scenario is plan.scenario for e in plan.instrument_faults)

    def test_bias_starts_between_25_and_50_percent(
        self, planned: Any, params: SimulationParams
    ) -> None:
        """A bias begins inside the configured fraction of the run and runs to the end."""
        for plan in planned.plans:
            if plan.scenario is not ScenarioKind.BIAS:
                continue
            for episode in plan.instrument_faults:
                assert 0.25 * params.run_length_steps <= episode.start_step
                assert episode.start_step <= 0.50 * params.run_length_steps
                assert episode.last_step == params.run_length_steps - 1

    def test_labels_come_from_the_static_severity_table(self, planned: Any) -> None:
        """Every planned run of a type has the TDD severity of that type."""
        for plan in planned.plans:
            if plan.scenario is ScenarioKind.NORMAL:
                assert plan.scenario.severity is None
            else:
                assert plan.scenario.severity is FAULT_SEVERITY[plan.scenario]

    def test_deterministic_and_seed_sensitive(
        self,
        params: SimulationParams,
        catalog: SensorCatalog,
        config: ScenarioConfig,
        planned: Any,
    ) -> None:
        """Same stream, same plan; another master seed, another plan."""
        again = plan_runs(
            params,
            catalog,
            planner_rng(params),
            config,
            fault_injection_rate=RATE,
            severity_weights=CAPACITY_B,
        )
        assert again == planned.plans
        other = dataclasses.replace(params, openmc_seed=7)
        different = plan_runs(
            other,
            catalog,
            planner_rng(other),
            config,
            fault_injection_rate=RATE,
            severity_weights=CAPACITY_B,
        )
        assert different != planned.plans

    def test_planned_readings_property(self, planned: Any) -> None:
        """The labelled readings are the sum over the plans."""
        assert planned.planned_readings == sum(
            p.expected_labelled_readings(planned.n_sensors) for p in planned.plans
        )
        assert sum(planned.readings_by_severity().values()) == planned.planned_readings
        assert sum(kp.planned_readings for kp in planned.kinds.values()) == planned.planned_readings


class TestPlacementLimits:
    """The planner stops, with the cause, when the caps leave no room."""

    def test_no_room_for_the_grid(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """One sensor per run cannot cover six groups x three levels in three runs."""
        one_sensor = dataclasses.replace(
            config,
            planner=dataclasses.replace(config.planner, max_sensor_fraction=1 / 27),
        )
        with pytest.raises(ValueError, match="no room"):
            _plan_instrument_episodes(
                ScenarioKind.STUCK,
                3,
                100.0,
                params,
                catalog,
                one_sensor,
                np.random.default_rng(0),
            )

    def test_cannot_place_that_many_spikes(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """Spikes are at least two samples apart: a sensor holds at most half the window."""
        dense = dataclasses.replace(
            config, planner=dataclasses.replace(config.planner, max_spike_rate=1.0)
        )
        with pytest.raises(ValueError, match="cannot place another spike"):
            _plan_instrument_episodes(
                ScenarioKind.NOISE_SPIKE,
                1,
                9000.0,
                params,
                catalog,
                dense,
                np.random.default_rng(0),
            )

    def test_episode_longer_than_the_room_after_the_warmup(
        self, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
    ) -> None:
        """A drift of 900 steps does not fit after a 600-step warm-up of a 1000-step run."""
        short = dataclasses.replace(params, run_length_steps=1000, warmup_fraction=0.6)
        with pytest.raises(ValueError, match="warm-up"):
            _plan_instrument_episodes(
                ScenarioKind.DRIFT, 3, 1000.0, short, catalog, config, np.random.default_rng(0)
            )

    def test_free_sensor_rules(self) -> None:
        """Distinct sensors for most kinds; spikes may repeat up to the per-sensor cap."""
        members = ("A", "B", "C")
        assert _free_sensor({}, members, False, 30, 2, 0) == "A"
        assert _free_sensor({"A": 1}, members, False, 30, 2, 0) == "B"
        assert _free_sensor({"A": 1, "B": 1}, members, False, 30, 2, 0) is None
        assert _free_sensor({"A": 30, "B": 1}, members, True, 30, 2, 0) == "B"
        assert _free_sensor({"A": 30, "B": 30}, members, True, 30, 2, 0) is None

    def test_place_spike_keeps_two_samples_apart(self) -> None:
        """A free step is at least 2 away from every taken one; a full range gives None."""
        rng = np.random.default_rng(0)
        taken = {10, 12, 14}
        for _ in range(20):
            step = _place_spike(taken, 8, 18, rng)
            assert step is not None
            assert all(abs(step - other) >= 2 for other in taken)
        assert _place_spike({1, 3, 5}, 1, 6, rng) is None

    def test_planner_settings_dataclass_is_frozen(self, config: ScenarioConfig) -> None:
        """The settings cannot be mutated by accident."""
        assert isinstance(config.planner, PlannerSettings)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.planner.n_min_runs = 9  # type: ignore[misc]
