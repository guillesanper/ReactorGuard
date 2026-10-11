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

from data.generators.reactor_simulator import ReactorSimulator
from data.generators.simulation_study import (
    DEFAULT_CADENCES_S,
    CadenceResult,
    RuleMarginStats,
    _jsonable,
    build_report,
    cadence_run,
    cadence_study,
    determinism_check,
    main,
    measure_throughput,
    noise_calibration,
    parquet_digest,
    simulate_runs,
    subprocess_digest,
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
