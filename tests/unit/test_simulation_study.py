"""Unit tests for data.generators.simulation_study.

Las mediciones de F3-M3 salen de este modulo, asi que sus tests comprueban que
mide lo que dice medir: el estudio de cadencia reproduce el mismo escenario fisico
a cada dt, la calibracion de ruido detecta que un sigma excesivo viola reglas, y el
determinismo se verifica tambien contra un proceso nuevo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.generators import simulation_study
from data.generators.reactor_simulator import ReactorSimulator
from data.generators.simulation_study import (
    CANDIDATE_WEIGHTS,
    DEFAULT_CADENCES_S,
    EXPECTED_VALIDATOR_FAULT,
    CadenceResult,
    FaultRuleCell,
    FaultRunResult,
    InstrumentRuleCell,
    RuleMarginStats,
    _jsonable,
    build_report,
    cadence_run,
    cadence_study,
    capacity_report,
    determinism_check,
    fault_matrix,
    fault_run,
    instrument_detection,
    instrument_matrix,
    main,
    measure_throughput,
    noise_calibration,
    parquet_digest,
    plan_coverage_report,
    simulate_runs,
    single_channel_roles,
    subprocess_digest,
    summarize_fault_matrix,
    summarize_instrument_matrix,
    summarize_rules,
    with_noise_sigma,
)
from data.schemas.sensor_spans import SensorSpan
from data.simulation.physics_constraints import (
    RULE_NAMES,
    ConstraintLimits,
    evaluate_rules_detailed,
    rule_margins,
)
from data.simulation.scenario import (
    INSTRUMENT_KINDS,
    FaultLevel,
    InstrumentFaultKind,
    ScenarioKind,
)
from tests.support.sim_frames import build_simulator, wide_frame, write_sim_params


@dataclass
class _FakeResult:
    faults: list[str] = field(default_factory=list)


class _FakeValidator:
    """Validator double: alerts on every 10th reading with one fault."""

    def __init__(self, spans: dict[str, SensorSpan]) -> None:
        self.seen = 0

    def validate(self, reading: Any) -> _FakeResult:
        self.seen += 1
        return _FakeResult(["kalman_anomaly"] if self.seen % 10 == 0 else [])

    def get_metrics(self) -> dict[str, Any]:
        return {"faults_by_type": {"kalman_anomaly": self.seen // 10}}


@pytest.fixture(scope="module")
def simulator(tmp_path_factory: pytest.TempPathFactory) -> ReactorSimulator:
    """Return a simulator over 3 runs of 30 steps."""
    return build_simulator(tmp_path_factory.mktemp("study"))


@pytest.fixture(scope="module")
def limits(simulator: ReactorSimulator) -> ConstraintLimits:
    """Return the constraint limits of the reference plant."""
    return ConstraintLimits.from_plant(simulator.model.plant)


class TestNoiseSigmaOverride:
    """with_noise_sigma replaces sigmas without touching the catalog."""

    def test_replaces_only_the_named_tags(self, simulator: ReactorSimulator) -> None:
        """Named sensors change, the rest and the roles are kept."""
        base = simulator.catalog
        changed = with_noise_sigma(base, {"TC-HL-1": 0.5})
        assert changed.by_tag("TC-HL-1").noise_sigma == 0.5
        assert changed.by_tag("TC-HL-2").noise_sigma == base.by_tag("TC-HL-2").noise_sigma
        assert changed.roles == base.roles
        assert base.by_tag("TC-HL-1").noise_sigma != 0.5

    def test_unknown_tag(self, simulator: ReactorSimulator) -> None:
        """A tag outside the catalog is an error."""
        with pytest.raises(KeyError, match="NOPE"):
            with_noise_sigma(simulator.catalog, {"NOPE": 1.0})

    @pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_sigma(self, simulator: ReactorSimulator, sigma: float) -> None:
        """Sigma must be finite and positive."""
        with pytest.raises(ValueError, match="sigma"):
            with_noise_sigma(simulator.catalog, {"TC-HL-1": sigma})


class TestRuleMarginStats:
    """The derived distances in sigma units are guarded against empty or flat samples."""

    def test_sigma_units(self) -> None:
        """mean/std and min/std."""
        stats = RuleMarginStats(10, 10, 0, 1.0, 4.0, 2.0)
        assert stats.sigma_to_threshold == 2.0
        assert stats.min_margin_sigma == 0.5

    def test_none_without_spread(self) -> None:
        """No std (or no evaluable rows) gives None instead of dividing by zero."""
        assert RuleMarginStats(5, 0, 0, None, None, None).sigma_to_threshold is None
        assert RuleMarginStats(5, 5, 0, 1.0, 2.0, 0.0).min_margin_sigma is None


class TestNoiseCalibration:
    """Rules measured on noisy normal runs, with their distance to the threshold."""

    def test_catalog_noise_violates_nothing(
        self, simulator: ReactorSimulator, limits: ConstraintLimits
    ) -> None:
        """The calibrated catalog gives zero violations and positive margins."""
        sims = simulate_runs(simulator, n_runs=2, n_steps=20)
        result = noise_calibration(sims, simulator.catalog, simulator.model.table, limits)
        assert result.n_runs == 2
        assert result.n_rows == 40
        assert result.n_rows_violating == 0
        assert result.total_violations == 0
        assert list(result.rules) == list(RULE_NAMES)
        for rule, stats in result.rules.items():
            assert stats.min_margin is not None and stats.min_margin > 0.0, rule
        assert "TC-HL-1" in result.sigma_by_tag
        assert "TC-CORE-01" not in result.sigma_by_tag

    def test_excessive_noise_is_detected(
        self, simulator: ReactorSimulator, limits: ConstraintLimits
    ) -> None:
        """A large loop sigma violates the energy rule and the margin turns negative."""
        sims = simulate_runs(simulator, n_runs=2, n_steps=30)
        loop = list(simulator.catalog.roles["T_out"]) + list(simulator.catalog.roles["T_in"])
        noisy = with_noise_sigma(simulator.catalog, {tag: 3.0 for tag in loop})
        result = noise_calibration(sims, noisy, simulator.model.table, limits, label="loud")
        assert result.label == "loud"
        assert result.rules["energy_balance"].n_violations > 0
        assert result.rules["energy_balance"].min_margin is not None
        assert result.rules["energy_balance"].min_margin < 0.0

    def test_margin_sign_matches_violations(
        self, simulator: ReactorSimulator, limits: ConstraintLimits
    ) -> None:
        """summarize_rules counts violations and margin<0 identically."""
        sim = simulate_runs(simulator, n_runs=1, n_steps=30)[0]
        loop = list(simulator.catalog.roles["T_out"]) + list(simulator.catalog.roles["T_in"])
        noisy = with_noise_sigma(simulator.catalog, {tag: 3.0 for tag in loop})
        from data.simulation.dataset_builder import measure, wide_frame_from_matrix

        measured = measure(sim, noisy, np.random.default_rng(1))
        wide = wide_frame_from_matrix(measured, noisy, sim.timestamps)
        evaluation = evaluate_rules_detailed(
            wide, noisy.roles, simulator.model.table, limits
        )
        margins = rule_margins(wide, noisy.roles, simulator.model.table, limits)
        stats = summarize_rules(evaluation, margins)
        for rule in RULE_NAMES:
            negative = int((margins[rule][evaluation.evaluable[rule]] < 0).sum())
            assert stats[rule].n_violations == negative

    def test_needs_runs(self, simulator: ReactorSimulator, limits: ConstraintLimits) -> None:
        """An empty run list is an error."""
        with pytest.raises(ValueError, match="at least one run"):
            noise_calibration([], simulator.catalog, simulator.model.table, limits)


class TestCadenceStudy:
    """The same physical scenario is sampled at several cadences."""

    def test_result_fields_at_180_s(self, simulator: ReactorSimulator) -> None:
        """At 180 s the dT/dt rule is unobservable and nothing is violated."""
        result = cadence_run(simulator, 180.0, 3600.0, validator_factory=_FakeValidator)
        assert isinstance(result, CadenceResult)
        assert result.n_steps == 21
        assert result.n_readings == 21 * 27
        assert result.true_rows_violating == 0
        assert all(v == 0 for v in result.true_violations_by_rule.values())
        assert result.min_observable_jump_k == pytest.approx(1800.0)
        assert result.max_state_dt_dt_c_s < 10.0
        assert result.max_true_fd_rate_c_s < 10.0
        assert 0.0 <= result.mass_evaluable_true <= 1.0

    def test_validator_counts_come_from_the_validator(self, simulator: ReactorSimulator) -> None:
        """The fake validator alerts on every tenth reading; the study counts them."""
        result = cadence_run(simulator, 180.0, 3600.0, validator_factory=_FakeValidator)
        assert result.validator_alerted_readings == (21 * 27) // 10
        assert result.validator_faults == result.validator_alerted_readings
        assert result.alerts_per_1000 == pytest.approx(1000.0 * result.validator_faults / (21 * 27))
        assert result.validator_faults_by_type == {"kalman_anomaly": (21 * 27) // 10}

    def test_hold_and_ramp_split(self, simulator: ReactorSimulator) -> None:
        """The scenario contains ramps and alerts are reported for steady and ramping samples."""
        result = cadence_run(simulator, 180.0, 4 * 3600.0, validator_factory=_FakeValidator)
        assert result.ramp_fraction > 0.0
        assert result.alerts_per_1000_hold is not None
        assert result.alerts_per_1000_ramp is not None

    def test_alerts_are_attributed_to_sensors(self, simulator: ReactorSimulator) -> None:
        """The per-sensor tally adds up to the total number of alerts."""
        result = cadence_run(simulator, 180.0, 3600.0, validator_factory=_FakeValidator)
        assert sum(result.validator_faults_by_sensor.values()) == result.validator_faults
        assert set(result.validator_faults_by_sensor) <= set(simulator.catalog.tags)

    def test_study_scenario_has_ramps_unlike_the_dataset_warmup(
        self, simulator: ReactorSimulator
    ) -> None:
        """A short warm-up in seconds, not the dataset's fraction, lets ramps happen."""
        with_ramps = cadence_run(
            simulator, 180.0, 4 * 3600.0, validator_factory=_FakeValidator, warmup_s=600.0
        )
        all_warmup = cadence_run(
            simulator, 180.0, 4 * 3600.0, validator_factory=_FakeValidator, warmup_s=10 * 3600.0
        )
        assert with_ramps.ramp_fraction > 0.0
        assert all_warmup.ramp_fraction == 0.0

    def test_finer_cadence_observes_more(self, simulator: ReactorSimulator) -> None:
        """At 1 s the dT/dt rule can fire on a 10 K jump; at 180 s it needs 1800 K."""
        fine = cadence_run(simulator, 1.0, 120.0, validator_factory=_FakeValidator)
        assert fine.n_steps == 121
        assert fine.min_observable_jump_k == pytest.approx(10.0)

    def test_noise_of_the_rate_grows_as_dt_shrinks(self, simulator: ReactorSimulator) -> None:
        """The finite-difference noise of T_avg scales as 1/dt (the evaluable-mass trap)."""
        coarse = cadence_run(simulator, 180.0, 3600.0, validator_factory=_FakeValidator)
        fine = cadence_run(simulator, 10.0, 600.0, validator_factory=_FakeValidator)
        assert fine.dt_avg_noise_k_s > 5.0 * coarse.dt_avg_noise_k_s

    def test_with_the_real_validator(self, simulator: ReactorSimulator) -> None:
        """The default factory builds the Phase 2 validator lazily and runs it."""
        result = cadence_run(simulator, 180.0, 3600.0)
        assert result.validator_faults >= 0
        assert result.validator_alerted_readings <= result.n_readings

    def test_study_covers_each_cadence(self, simulator: ReactorSimulator) -> None:
        """One result per requested cadence, in order."""
        results = cadence_study(
            simulator, (180.0, 60.0), duration_s=1800.0, validator_factory=_FakeValidator
        )
        assert [r.dt_s for r in results] == [180.0, 60.0]
        assert DEFAULT_CADENCES_S == (1.0, 10.0, 60.0, 180.0)

    def test_invalid_cadence(self, simulator: ReactorSimulator) -> None:
        """dt <= 0 and a duration shorter than one interval are errors."""
        with pytest.raises(ValueError, match="dt_s"):
            cadence_run(simulator, 0.0, 100.0, validator_factory=_FakeValidator)
        with pytest.raises(ValueError, match="samples"):
            cadence_run(simulator, 180.0, 10.0, validator_factory=_FakeValidator)


class TestDeterminismAndThroughput:
    """Reproducibility in-process and across processes, and end-to-end speed."""

    def test_in_process(self, simulator: ReactorSimulator, tmp_path: Path) -> None:
        """Two builds give equal frames and equal parquet bytes."""
        result = determinism_check(simulator, simulator.plans[0], tmp_path)
        assert result.frames_equal
        assert result.parquet_equal
        assert result.cross_process_equal is None
        assert result.n_rows == 30 * 27
        assert len(result.sha256) == 64

    def test_across_processes(self, tmp_path: Path) -> None:
        """A fresh interpreter produces the same bytes."""
        params_path = write_sim_params(tmp_path)
        from data.simulation.simulation_params import load_simulation_params

        sim = ReactorSimulator.from_params(load_simulation_params(params_path))
        result = determinism_check(sim, sim.plans[1], tmp_path, params_path)
        assert result.cross_process_equal is True

    def test_across_processes_with_a_shortened_run(self, tmp_path: Path) -> None:
        """The child honours a shorter run (--steps)."""
        import dataclasses

        from data.simulation.simulation_params import load_simulation_params

        params_path = write_sim_params(tmp_path)
        sim = ReactorSimulator.from_params(load_simulation_params(params_path))
        plan = dataclasses.replace(sim.plans[0], n_steps=12)
        result = determinism_check(sim, plan, tmp_path, params_path)
        assert result.cross_process_equal is True
        assert result.n_rows == 12 * 27

    def test_failed_child_is_reported(self, tmp_path: Path) -> None:
        """A child that cannot load its params raises RuntimeError with its stderr."""
        with pytest.raises(RuntimeError, match="digest subprocess failed"):
            subprocess_digest(tmp_path / "missing.yaml", 0, None)

    def test_digest_depends_on_the_content(
        self, simulator: ReactorSimulator, tmp_path: Path
    ) -> None:
        """Different frames hash differently; the same frame hashes the same."""
        a = simulator.generate_run(simulator.plans[0])
        b = simulator.generate_run(simulator.plans[1])
        assert parquet_digest(a, tmp_path, "a.parquet") == parquet_digest(a, tmp_path, "a2.parquet")
        assert parquet_digest(a, tmp_path, "a.parquet") != parquet_digest(b, tmp_path, "b.parquet")

    def test_throughput(self, simulator: ReactorSimulator, tmp_path: Path) -> None:
        """The throughput report counts steps and readings and splits the time."""
        result = measure_throughput(simulator, tmp_path, n_runs=2, n_steps=10)
        assert result.n_runs == 2
        assert result.n_steps == 20
        assert result.n_readings == 20 * 27
        assert result.steps_per_s > 0.0
        assert result.readings_per_s == pytest.approx(result.steps_per_s * 27, rel=0.01)
        assert min(result.seconds_integrate, result.seconds_assemble, result.seconds_write) > 0.0


class TestReport:
    """The report is JSON and the CLI writes it."""

    def test_build_report_is_json(self, simulator: ReactorSimulator, tmp_path: Path) -> None:
        """All four studies are present and serialisable."""
        params_path = write_sim_params(tmp_path)
        report = build_report(
            simulator,
            params_path,
            cadences_s=(180.0, 60.0),
            duration_s=1800.0,
            n_runs=2,
            n_steps=15,
            validator_factory=_FakeValidator,
        )
        assert set(report) == {"params", "cadence", "noise", "determinism", "throughput"}
        assert len(report["cadence"]) == 2
        assert report["noise"]["n_rows"] == 30
        assert report["determinism"]["cross_process_equal"] is True
        json.dumps(report)

    def test_jsonable_converts_numpy_and_containers(self) -> None:
        """numpy scalars become Python numbers; tuples become lists."""
        out = _jsonable({"a": np.float64(1.5), "b": (np.int64(2), 3)})
        assert out == {"a": 1.5, "b": [2, 3]}
        assert type(out["a"]) is float

    def test_main_report(self, tmp_path: Path) -> None:
        """`report` writes a JSON file."""
        params_path = write_sim_params(tmp_path)
        output = tmp_path / "report.json"
        code = main(
            ["report", "--params", str(params_path), "--output", str(output),
             "--duration-s", "1800", "--n-runs", "1", "--steps", "12"]
        )
        assert code == 0
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["noise"]["n_runs"] == 1
        assert [c["dt_s"] for c in data["cadence"]] == list(DEFAULT_CADENCES_S)

    def test_main_digest(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """`digest` prints the SHA-256 of a run."""
        params_path = write_sim_params(tmp_path)
        code = main(["digest", "--params", str(params_path), "--steps", "10"])
        assert code == 0
        printed = capsys.readouterr().out.strip().splitlines()[-1]
        assert len(printed) == 64
        int(printed, 16)


_HOLD = 6
_RECOVERY = 2


@pytest.fixture(scope="module")
def void_high(simulator: ReactorSimulator) -> FaultRunResult:
    """Return a high-level void run (expensive: integrated once per module)."""
    return fault_run(
        simulator,
        ScenarioKind.VOID_FORMATION,
        FaultLevel.HIGH,
        hold_steps=_HOLD,
        recovery_steps=_RECOVERY,
    )


class TestFaultMatrix:
    """The fault matrix measures which rules a process fault violates, and when."""

    HOLD = _HOLD
    RECOVERY = _RECOVERY

    def test_window_and_identity(self, void_high: FaultRunResult) -> None:
        """The run starts at the end of the warm-up and the window is closed."""
        assert void_high.scenario == "void_formation"
        assert void_high.level == "high"
        assert void_high.injected_step == 6
        assert void_high.last_active_step == 6 + 10 + self.HOLD + self.RECOVERY
        cell = void_high.true_rules["boiling"]
        assert cell.n_window_steps == void_high.last_active_step - void_high.injected_step + 1

    def test_every_rule_has_a_cell(self, void_high: FaultRunResult) -> None:
        """One cell per rule, for true values and for measurements."""
        assert tuple(void_high.true_rules) == RULE_NAMES
        assert tuple(void_high.measured_rules) == RULE_NAMES
        assert tuple(void_high.outside_true) == RULE_NAMES

    def test_boiling_is_violated_with_a_delay(self, void_high: FaultRunResult) -> None:
        """A high void level boils: rule 6 fires, after the injection and not before."""
        cell = void_high.true_rules["boiling"]
        assert cell.n_violations > 0
        assert cell.rate is not None and cell.rate > 0.0
        assert cell.first_delay_steps is not None
        assert 1 <= cell.first_delay_steps <= 16

    def test_true_state_derivative_exceeds_the_limit_but_the_sampled_one_does_not(
        self, void_high: FaultRunResult
    ) -> None:
        """The instantaneous dT/dt passes 10 C/s (D9) yet rule 3 cannot see it at 180 s."""
        assert void_high.state_rate_violation_steps > 0
        assert void_high.max_fd_rate_c_s < 10.0
        assert void_high.true_rules["max_dt_dt"].n_violations == 0

    def test_energy_is_conserved(self, void_high: FaultRunResult) -> None:
        """Conservation holds on the full state although the rules are violated."""
        assert void_high.conservation_compliant
        assert void_high.max_energy_residual < 1e-6

    def test_a_mild_fault_violates_nothing(self, simulator: ReactorSimulator) -> None:
        """A low flow reduction leaves every rule quiet, on truth and measurement."""
        result = fault_run(
            simulator,
            ScenarioKind.COOLANT_FLOW_REDUCTION,
            FaultLevel.LOW,
            hold_steps=self.HOLD,
            recovery_steps=self.RECOVERY,
        )
        assert all(c.n_violations == 0 for c in result.true_rules.values())
        assert all(c.first_delay_steps is None for c in result.true_rules.values())
        assert result.response.margin_min_k > 10.0

    def test_matrix_groups_runs_by_cell(self, simulator: ReactorSimulator) -> None:
        """Repetitions use different run indices and aggregate into one cell."""
        results = fault_matrix(
            simulator,
            scenarios=(ScenarioKind.CORE_TEMP_EXCURSION,),
            levels=(FaultLevel.LOW,),
            n_repeats=2,
            hold_steps=self.HOLD,
            recovery_steps=self.RECOVERY,
        )
        assert [r.run_index for r in results] == [0, 1]
        summary = summarize_fault_matrix(results)
        assert list(summary) == ["core_temp_excursion/low"]
        cell = summary["core_temp_excursion/low"]
        assert cell["n_runs"] == 2
        assert cell["conservation_compliant"] is True
        assert set(cell["true"]) == set(RULE_NAMES)
        json.dumps(_jsonable(summary))

    def test_rejects_zero_repeats(self, simulator: ReactorSimulator) -> None:
        """At least one run per cell."""
        with pytest.raises(ValueError, match="n_repeats"):
            fault_matrix(simulator, n_repeats=0)

    def test_cell_rate_is_none_without_evaluable_rows(self) -> None:
        """A rule that never evaluated has no rate."""
        cell = FaultRuleCell(
            n_window_steps=4, n_evaluable=0, n_violations=0, first_delay_steps=None
        )
        assert cell.rate is None

    def test_main_faults_writes_the_matrix(self, tmp_path: Path) -> None:
        """`faults` writes summary and runs for all nine cells."""
        params_path = write_sim_params(tmp_path)
        output = tmp_path / "matrix.json"
        code = main(
            ["faults", "--params", str(params_path), "--output", str(output),
             "--repeats", "1", "--hold-steps", "3", "--recovery-steps", "1"]
        )
        assert code == 0
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data["summary"]) == 9
        assert len(data["runs"]) == 9
        assert data["params"]["repeats"] == 1


class TestRuleMarginsAgainstTheModel:
    """The true state sits well inside every rule (margins are positive)."""

    def test_equilibrium_margins(
        self, simulator: ReactorSimulator, limits: ConstraintLimits
    ) -> None:
        """Noise-free equilibrium: every evaluable margin is positive."""
        from tests.support.sim_frames import ramp_loads, run_load_profile

        model = simulator.model
        trace = run_load_profile(model, ramp_loads(1.0, 1.0, 5, 3))
        wide = wide_frame(trace, model, simulator.catalog)
        margins = rule_margins(wide, simulator.catalog.roles, model.table, limits)
        assert (margins.min(skipna=True) > 0.0).all()


# ----------------------------------------------------------------------
# F3-M5: fallos de instrumento, capacidad y plan
# ----------------------------------------------------------------------


@dataclass
class _FaultStub:
    """Validator fault double carrying the FaultType value the study reads."""

    fault_type: Any


@dataclass
class _TypeStub:
    value: str


class _FlagSensorValidator:
    """Validator double: flags EVERY reading of one sensor with a given fault type."""

    def __init__(self, tag: str, fault: str) -> None:
        self.tag = tag
        self.fault = fault

    def validate(self, reading: Any) -> _FakeResult:
        if reading.sensor.id == self.tag:
            return _FakeResult([_FaultStub(_TypeStub(self.fault))])  # type: ignore[list-item]
        return _FakeResult([])


@pytest.fixture(scope="module")
def big_simulator(tmp_path_factory: pytest.TempPathFactory) -> ReactorSimulator:
    """Return a simulator over the dataset-sized plan (34 runs of 1500 steps); never integrated."""
    return build_simulator(
        tmp_path_factory.mktemp("big"),
        n_samples=50000,
        run_length_steps=1500,
        openmc_seed=42,
    )


class TestSingleChannelRoles:
    """Isolating a sensor removes the median vote for the roles it plays."""

    def test_only_the_roles_of_the_sensor_change(self, simulator: ReactorSimulator) -> None:
        """TC-HL-1 becomes the only T_out channel; every other role is unchanged."""
        roles = simulator.catalog.roles
        isolated = single_channel_roles(roles, "TC-HL-1")
        assert isolated["T_out"] == ("TC-HL-1",)
        assert isolated["T_in"] == roles["T_in"]
        assert isolated["flow"] == roles["flow"]

    def test_a_flow_channel_is_isolated_in_every_role_it_plays(
        self, simulator: ReactorSimulator
    ) -> None:
        """FT-HL-1 is in `flow` and in `flow_hot_leg`: both become single-channel."""
        isolated = single_channel_roles(simulator.catalog.roles, "FT-HL-1")
        assert isolated["flow"] == ("FT-HL-1",)
        assert isolated["flow_hot_leg"] == ("FT-HL-1",)
        assert isolated["flow_cold_leg"] == simulator.catalog.roles["flow_cold_leg"]

    def test_a_sensor_outside_the_roles_changes_nothing(self, simulator: ReactorSimulator) -> None:
        """TC-CORE-01 is in no role."""
        roles = simulator.catalog.roles
        assert single_channel_roles(roles, "TC-CORE-01") == {k: tuple(v) for k, v in roles.items()}


@pytest.fixture(scope="module")
def cells(simulator: ReactorSimulator) -> list[InstrumentRuleCell]:
    """Return a small matrix: low and high levels on three sensors, both directions."""
    return instrument_matrix(
        simulator,
        levels=(FaultLevel.LOW, FaultLevel.HIGH),
        sensors=("TC-HL-1", "TC-CORE-01", "FT-HL-1"),
        directions=(1, -1),
        control_sensors=(),
    )


class TestInstrumentMatrix:
    """The rule matrix of the five instrument faults, sensor by sensor."""

    def test_one_cell_per_kind_level_sensor_direction(
        self, cells: list[InstrumentRuleCell]
    ) -> None:
        """5 kinds x 2 levels x 3 sensors x 2 directions, every rule in every cell."""
        assert len(cells) == 5 * 2 * 3 * 2
        assert all(set(c.voted) == set(RULE_NAMES) for c in cells)
        assert all(set(c.per_channel) == set(RULE_NAMES) for c in cells)
        assert {c.start_step for c in cells} == {12}

    def test_a_sensor_outside_the_roles_violates_nothing(
        self, cells: list[InstrumentRuleCell]
    ) -> None:
        """No rule reads TC-CORE-01: it cannot alter any of them, voted or isolated."""
        control = [c for c in cells if c.sensor_id == "TC-CORE-01"]
        assert control
        assert all(c.in_roles == () for c in control)
        for cell in control:
            assert all(r.n_violations == 0 for r in cell.voted.values())
            assert all(r.n_violations == 0 for r in cell.per_channel.values())

    def test_a_high_bias_on_a_t_out_channel_breaks_the_energy_balance(
        self, cells: list[InstrumentRuleCell]
    ) -> None:
        """~11 K on one hot-leg channel is a >5 % error of the core delta T, voted or not."""
        bias = [
            c
            for c in cells
            if c.kind == "bias" and c.level == "high" and c.sensor_id == "TC-HL-1"
        ]
        assert bias
        for cell in bias:
            assert cell.voted["energy_balance"].n_violations > 0
            assert cell.per_channel["energy_balance"].n_violations > 0
            assert cell.in_roles == ("T_out",)

    def test_isolating_the_channel_exposes_what_the_median_hides(
        self, cells: list[InstrumentRuleCell]
    ) -> None:
        """A low bias on one of two T_out channels: the median halves it, the channel does not."""
        low = [
            c for c in cells if c.kind == "bias" and c.level == "low" and c.sensor_id == "TC-HL-1"
        ]
        voted = sum(c.voted["energy_balance"].n_violations for c in low)
        single = sum(c.per_channel["energy_balance"].n_violations for c in low)
        assert single > voted

    def test_outside_the_window_the_rules_stay_quiet(
        self, cells: list[InstrumentRuleCell]
    ) -> None:
        """Before the episode nothing is violated (the plant is the normal one)."""
        for cell in cells:
            if cell.kind == "bias":
                continue
            assert sum(cell.outside_voted.values()) == 0

    def test_summary_separates_role_and_non_role_sensors(
        self, cells: list[InstrumentRuleCell]
    ) -> None:
        """Per (kind, level): role sensors by sensor, non-role as a block."""
        summary = summarize_instrument_matrix(cells)
        assert set(summary) == {
            f"{k}/{lv}"
            for k in ("drift", "stuck", "noise_spike", "bias", "intermittent")
            for lv in ("low", "high")
        }
        entry = summary["bias/high"]
        assert entry["n_cells"] == 6
        assert entry["role_sensors"]["n_cells"] == 4
        assert entry["non_role_sensors"]["n_cells"] == 2
        assert set(entry["role_sensors"]["by_sensor"]) == {"TC-HL-1", "FT-HL-1"}
        assert entry["role_sensors"]["by_sensor"]["TC-HL-1"]["roles"] == ["T_out"]
        voted = entry["role_sensors"]["voted"]
        assert voted["energy_balance"]["n_violations"] > 0
        assert entry["non_role_sensors"]["voted"]["energy_balance"]["n_violations"] == 0

    def test_default_sensors_are_the_role_sensors_plus_the_control(
        self, simulator: ReactorSimulator
    ) -> None:
        """Without `sensors`, the 14 role sensors and the control are measured."""
        one = instrument_matrix(
            simulator,
            kinds=(InstrumentFaultKind.NOISE_SPIKE,),
            levels=(FaultLevel.HIGH,),
            directions=(1,),
        )
        tags = {c.sensor_id for c in one}
        assert "TC-CORE-01" in tags
        assert {"TC-HL-1", "TC-CL-2", "FT-CL-1", "PT-PRZ-02", "NI-FLUX-03", "PWR-TH-01"} <= tags
        assert len(tags) == 15

    def test_requires_a_direction(self, simulator: ReactorSimulator) -> None:
        """An empty direction list is an error."""
        with pytest.raises(ValueError, match="directions"):
            instrument_matrix(simulator, directions=())

    def test_episode_window_is_clipped_to_the_run(self, simulator: ReactorSimulator) -> None:
        """A drift of 900 steps in a 30-step run covers what fits."""
        cells = instrument_matrix(
            simulator,
            kinds=(InstrumentFaultKind.DRIFT,),
            levels=(FaultLevel.LOW,),
            sensors=("TC-HL-1",),
            directions=(1,),
        )
        assert (cells[0].start_step, cells[0].last_step) == (12, 29)


class TestInstrumentDetection:
    """What the Phase 2 validator sees of each injected instrument fault."""

    def test_counts_with_a_validator_that_flags_one_sensor(
        self, simulator: ReactorSimulator
    ) -> None:
        """Only the episode on the flagged sensor is detected; false positives are counted."""
        cells = instrument_detection(
            simulator,
            n_repeats=1,
            kinds=(InstrumentFaultKind.STUCK,),
            levels=(FaultLevel.HIGH,),
            validator_factory=lambda spans: _FlagSensorValidator("TC-CORE-01", "stuck"),
        )
        assert len(cells) == 1
        cell = cells[0]
        assert (cell.kind, cell.level) == ("stuck", "high")
        # One episode per sensor type (6); window 18 samples each (12..29).
        assert cell.n_episodes == 6
        assert cell.n_episodes_excl_flow == 5
        assert cell.labelled_readings == 6 * 18
        assert cell.episodes_detected == 1
        assert cell.episodes_detected_excl_flow == 1
        assert cell.readings_flagged == 18
        assert cell.readings_flagged_expected == 18
        assert cell.faults_by_type == {"stuck": 18}
        assert cell.first_delay_steps == [0]
        assert cell.episodes_by_group["thermocouple"] == (1, 1)
        assert cell.episodes_by_group["flow"] == (0, 1)
        # TC-CORE-01 is flagged on its 12 unlabelled samples: all false positives, no flow.
        n_non_flow = len(simulator.catalog.sensors) - 5
        unlabelled = 30 * n_non_flow - 5 * 18
        assert cell.false_positive_per_1000 == pytest.approx(1000 * 12 / unlabelled)
        assert cell.false_positive_per_1000_flow == 0.0

    def test_flow_false_positives_are_reported_apart(self, simulator: ReactorSimulator) -> None:
        """A validator that flags a flow channel only inflates the flow false-positive rate."""
        cells = instrument_detection(
            simulator,
            n_repeats=1,
            kinds=(InstrumentFaultKind.BIAS,),
            levels=(FaultLevel.LOW,),
            validator_factory=lambda spans: _FlagSensorValidator("FT-HL-2", "kalman_anomaly"),
        )
        cell = cells[0]
        assert cell.false_positive_per_1000 == 0.0
        assert cell.false_positive_per_1000_flow is not None
        assert cell.false_positive_per_1000_flow > 0.0
        assert cell.episodes_detected == 0

    def test_spikes_are_separate_episodes(self, simulator: ReactorSimulator) -> None:
        """`spikes_per_sensor` spikes per sensor, one episode and one reading each."""
        cells = instrument_detection(
            simulator,
            n_repeats=1,
            kinds=(InstrumentFaultKind.NOISE_SPIKE,),
            levels=(FaultLevel.HIGH,),
            spikes_per_sensor=3,
            validator_factory=lambda spans: _FlagSensorValidator("NOPE", "noise_spike"),
        )
        cell = cells[0]
        assert cell.n_episodes == 6 * 3
        assert cell.labelled_readings == 6 * 3
        assert cell.episodes_detected == 0

    def test_the_real_validator_sees_a_long_freeze(self, simulator: ReactorSimulator) -> None:
        """With the Phase 2 validator unchanged, an 18-sample freeze is a `stuck` fault."""
        cells = instrument_detection(
            simulator,
            n_repeats=1,
            kinds=(InstrumentFaultKind.STUCK,),
            levels=(FaultLevel.HIGH,),
        )
        cell = cells[0]
        assert cell.episodes_detected_excl_flow == cell.n_episodes_excl_flow == 5
        assert cell.faults_by_type.get("stuck", 0) > 0
        assert cell.readings_flagged_expected == cell.faults_by_type["stuck"]

    def test_requires_a_repeat(self, simulator: ReactorSimulator) -> None:
        """n_repeats >= 1."""
        with pytest.raises(ValueError, match="n_repeats"):
            instrument_detection(simulator, n_repeats=0)

    def test_expected_fault_types_cover_the_five_kinds(self) -> None:
        """Every manipulation names the validator fault it should produce."""
        assert set(EXPECTED_VALIDATOR_FAULT) == {k.value for k in INSTRUMENT_KINDS}


class TestCapacityAndPlan:
    """Budget, caps, reachable rate and the coverage of the planned dataset."""

    def test_capacity_report_of_the_default_configuration(
        self, big_simulator: ReactorSimulator
    ) -> None:
        """The D11 40 % is not feasible; the report says by how much, with the caps."""
        report = capacity_report(big_simulator)
        assert report["budget"]["feasible"] is False
        assert report["budget"]["runs_needed"] > report["budget"]["runs_available"]
        assert report["params"]["n_runs"] == 34
        severities = {s["severity"]: s for s in report["budget"]["severities"]}
        assert severities["LOW"]["capacity_per_run"] == 270
        assert severities["CRITICAL"]["kinds"] == [
            "stuck",
            "core_temp_excursion",
            "void_formation",
        ]
        rates = report["max_feasible_rate"]
        assert 0.005 < rates["d11"]["rate"] < 0.03
        assert rates["capacity_b"]["rate"] > rates["capacity_a"]["rate"] > rates["d11"]["rate"]
        assert rates["capacity_b"]["critical_share"] >= 0.05
        assert report["capacity_per_run"]["void_formation"]["max_episodes"] == 1

    def test_plan_report_of_a_feasible_configuration(
        self, big_simulator: ReactorSimulator
    ) -> None:
        """Eight types and the whole grid, with the partitions measured."""
        report = plan_coverage_report(
            big_simulator,
            fault_injection_rate=0.12,
            severity_weights=CANDIDATE_WEIGHTS["capacity_b"],
        )
        assert abs(report["rate"] - 0.12) < 0.02
        assert report["scenarios"]["normal"] >= 2
        coverage = report["coverage"]
        assert coverage["types_ok"] and coverage["grid_ok"] and coverage["passed"]
        assert coverage["overall"]["n_types"] == 8
        assert coverage["overall"]["grid_size"] == 8 * 6 * 3
        assert set(coverage["partitions"]) == {"train", "val", "test"}
        assert coverage["partitions"]["train"]["n_types"] == 8
        shares = report["by_severity"]
        assert shares["CRITICAL"]["fraction"] >= 0.05

    def test_plan_report_raises_when_unreachable(self, big_simulator: ReactorSimulator) -> None:
        """The default rate and weights are refused, with the numbers."""
        with pytest.raises(ValueError, match="unreachable"):
            plan_coverage_report(big_simulator)

    def test_main_plan_writes_capacity_and_plan(self, tmp_path: Path) -> None:
        """`plan` writes the capacity table and the coverage of a feasible plan."""
        params_path = write_sim_params(
            tmp_path, n_samples=50000, run_length_steps=1500, openmc_seed=42
        )
        output = tmp_path / "plan.json"
        code = main(
            [
                "plan",
                "--params",
                str(params_path),
                "--output",
                str(output),
                "--rate",
                "0.12",
                "--weights",
                "capacity_b",
            ]
        )
        assert code == 0
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["capacity"]["budget"]["feasible"] is False
        assert data["plan"]["coverage"]["passed"] is True

    def test_main_plan_records_the_refusal(self, tmp_path: Path) -> None:
        """Without overrides the plan entry carries the planner message."""
        params_path = write_sim_params(
            tmp_path, n_samples=50000, run_length_steps=1500, openmc_seed=42
        )
        output = tmp_path / "plan.json"
        assert main(["plan", "--params", str(params_path), "--output", str(output)]) == 0
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "unreachable" in data["plan"]["error"]

    def test_main_instrument_writes_matrix_and_detection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`instrument` writes the matrix summary, the detection cells and the raw cells."""
        original = simulation_study.instrument_matrix

        def two_sensors(simulator: ReactorSimulator, **kwargs: Any) -> list[InstrumentRuleCell]:
            """Restrict the matrix to two sensors and one direction (the full one is slow)."""
            return original(
                simulator,
                sensors=("TC-HL-1", "TC-CORE-01"),
                directions=(1,),
                control_sensors=(),
                **kwargs,
            )

        monkeypatch.setattr(simulation_study, "instrument_matrix", two_sensors)
        params_path = write_sim_params(tmp_path)
        output = tmp_path / "instrument.json"
        code = main(
            ["instrument", "--params", str(params_path), "--output", str(output), "--repeats", "1"]
        )
        assert code == 0
        data = json.loads(output.read_text(encoding="utf-8"))
        assert len(data["matrix"]) == 15
        assert len(data["cells"]) == 5 * 3 * 2
        assert len(data["detection"]) == 15
        assert data["params"]["repeats"] == 1
        assert {d["kind"] for d in data["detection"]} == {k.value for k in INSTRUMENT_KINDS}
