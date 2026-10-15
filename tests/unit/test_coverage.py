"""Unit tests for data.simulation.coverage.

Cubren la rejilla (tipo x grupo de sensor x nivel), las condiciones de modo
cubierto (k_min lecturas, n_min episodios), el corte temporal por particion con
la misma aritmetica que el stage `split`, y que medir sobre el PLAN de un run
equivale a medir sobre su FRAME.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from data.generators.reactor_simulator import ReactorSimulator
from data.simulation.coverage import (
    DEFAULT_RATIOS,
    FAULT_KINDS,
    LEVELS,
    PARTITIONS,
    CoverageCollector,
    FaultMode,
    ModeStats,
    coverage_report,
    grid,
    partition_bounds,
    partition_of,
)
from data.simulation.instrument_faults import build_episode
from data.simulation.scenario import (
    FaultLevel,
    InstrumentEpisode,
    InstrumentFaultKind,
    ProcessFaultSpec,
    RunPlan,
    ScenarioKind,
    make_run_plan,
)
from tests.support.sim_frames import build_simulator

SENSORS = (
    ("TC-A", "thermocouple"),
    ("TC-B", "thermocouple"),
    ("FT-A", "flow"),
    ("PT-A", "pressure"),
)
GROUPS = ("flow", "pressure", "thermocouple")


@dataclass(frozen=True)
class Ep:
    """One labelled window of a synthetic frame."""

    kind: str
    level: str
    start: int
    length: int
    tag: str | None = None
    axis: str = "instrument"
    corrupted_every: int = 0


def synth_frame(run_id: str, scenario: str, n_steps: int, episodes: list[Ep]) -> pd.DataFrame:
    """Return the long frame of a run with the columns the collector reads."""
    rows = []
    for step in range(n_steps):
        for tag, group in SENSORS:
            rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "timestep": step,
                    "sensor_id": tag,
                    "sensor_type": group,
                    "process_fault": "none",
                    "instrument_fault": "none",
                    "fault_level": None,
                    "episode_id": None,
                    "corrupted": False,
                }
            )
    frame = pd.DataFrame(rows)
    for number, ep in enumerate(episodes):
        in_window = frame["timestep"].between(ep.start, ep.start + ep.length - 1)
        mask = in_window if ep.tag is None else in_window & (frame["sensor_id"] == ep.tag)
        column = "process_fault" if ep.axis == "process" else "instrument_fault"
        frame.loc[mask, column] = ep.kind
        frame.loc[mask, "fault_level"] = ep.level
        frame.loc[mask, "episode_id"] = f"{run_id}:{ep.kind}:{number}"
        if ep.corrupted_every:
            frame.loc[mask & (frame["timestep"] % ep.corrupted_every == 0), "corrupted"] = True
    return frame


class TestPartitions:
    """The cut is the one of the `split` stage: integer division, remainder to test."""

    def test_bounds(self) -> None:
        """70/15/15 over 100 steps; the leftover of the integer cut goes to test."""
        assert partition_bounds(100) == (70, 85)
        assert partition_bounds(7) == (4, 5)
        assert partition_bounds(1500) == (1050, 1275)

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(0, "train"), (69, "train"), (70, "val"), (84, "val"), (85, "test"), (99, "test")],
    )
    def test_partition_of(self, index: int, expected: str) -> None:
        """Steps map to train, val and test in temporal order."""
        assert partition_of(index, 100) == expected

    def test_invalid_index_and_ratios(self) -> None:
        """Out-of-range index or ratios that do not sum to 1 are errors."""
        with pytest.raises(ValueError, match="outside"):
            partition_of(100, 100)
        with pytest.raises(ValueError, match="ratios"):
            partition_bounds(100, (0.5, 0.5, 0.5))
        with pytest.raises(ValueError, match="ratios"):
            partition_bounds(100, (0.7, 0.3, 0.0))
        with pytest.raises(ValueError, match="ratios"):
            partition_bounds(100, (0.7, 0.3))

    def test_default_ratios_match_the_training_section(self) -> None:
        """params.yaml training: has 0.70 / 0.15 / 0.15."""
        assert DEFAULT_RATIOS == (0.70, 0.15, 0.15)
        assert PARTITIONS == ("train", "val", "test")


class TestGrid:
    """The grid is 8 types x groups x 3 levels."""

    def test_size_and_labels(self) -> None:
        """Eight fault types, every group, three levels, sorted and unique."""
        modes = grid(GROUPS)
        assert len(modes) == 8 * 3 * 3
        assert len(set(modes)) == len(modes)
        assert modes == sorted(modes)
        assert FaultMode("drift", "flow", "low").label == "drift/flow/low"
        assert len(FAULT_KINDS) == 8
        assert tuple(lv.value for lv in LEVELS) == ("low", "medium", "high")

    def test_duplicated_groups_collapse(self) -> None:
        """A group listed twice is one group."""
        assert len(grid(["flow", "flow", "pressure"])) == 8 * 2 * 3


class TestCollector:
    """Counting readings, episodes and runs per mode."""

    def _collector(self) -> CoverageCollector:
        """Return a collector over two stuck runs and one excursion run."""
        collector = CoverageCollector()
        collector.add_frame(
            synth_frame(
                "sim_run_001",
                "stuck",
                100,
                [
                    Ep("stuck", "low", 10, 12, "TC-A", corrupted_every=1),
                    Ep("stuck", "high", 40, 30, "FT-A"),
                ],
            )
        )
        collector.add_frame(
            synth_frame(
                "sim_run_002",
                "stuck",
                100,
                [Ep("stuck", "low", 80, 20, "TC-B"), Ep("stuck", "medium", 5, 3, "PT-A")],
            )
        )
        collector.add_frame(
            synth_frame(
                "sim_run_003",
                "core_temp_excursion",
                100,
                [Ep("core_temp_excursion", "high", 30, 25, None, axis="process")],
            )
        )
        collector.add_frame(synth_frame("sim_run_004", "normal", 100, []))
        return collector

    def test_types_and_modes(self) -> None:
        """Stuck shows in three groups; the excursion labels every group."""
        report = self._collector().report(k_min_readings=5, n_min_episodes=1, gate=0.95)
        overall = report.overall
        assert overall.types_present == ("stuck", "core_temp_excursion")
        assert overall.n_types == 2
        assert not report.types_ok
        assert overall.stats[FaultMode("stuck", "thermocouple", "low")].episodes == 2
        assert overall.stats[FaultMode("stuck", "thermocouple", "low")].runs == 2
        assert overall.stats[FaultMode("stuck", "thermocouple", "low")].readings == 12 + 20
        excursion = overall.stats[FaultMode("core_temp_excursion", "thermocouple", "high")]
        assert excursion.readings == 25 * 2
        assert excursion.episodes == 1
        flow = overall.stats[FaultMode("core_temp_excursion", "flow", "high")]
        assert flow.readings == 25

    def test_corrupted_counts_come_from_the_frame(self) -> None:
        """The corrupted share of a mode is counted, not inferred."""
        report = self._collector().report(5, 1, 0.95)
        mode = FaultMode("stuck", "thermocouple", "low")
        assert report.overall.stats[mode].corrupted == 12

    def test_thresholds_decide_what_is_covered(self) -> None:
        """k_min readings AND n_min episodes."""
        collector = self._collector()
        loose = collector.report(1, 1, 0.5).overall
        tight_readings = collector.report(30, 1, 0.5).overall
        tight_episodes = collector.report(1, 2, 0.5).overall
        assert len(loose.covered) > len(tight_readings.covered)
        assert FaultMode("stuck", "pressure", "medium") in loose.covered
        assert FaultMode("stuck", "pressure", "medium") not in tight_readings.covered
        assert FaultMode("stuck", "thermocouple", "low") in tight_episodes.covered
        assert FaultMode("stuck", "flow", "high") not in tight_episodes.covered
        assert tight_readings.grid_size == 8 * 3 * 3
        assert set(loose.covered) | set(loose.missing) == set(grid(GROUPS))
        assert not set(loose.covered) & set(loose.missing)

    def test_type_cells(self) -> None:
        """Covered and total cells of one type."""
        overall = self._collector().report(1, 1, 0.5).overall
        covered, total = overall.type_cells("stuck")
        assert total == 9
        assert covered == 3
        assert overall.type_cells("drift") == (0, 9)

    def test_partitions_follow_the_timeline_of_each_type(self) -> None:
        """Two stuck runs make a 200-step timeline: train < 140 <= val < 170 <= test."""
        report = self._collector().report(1, 1, 0.5)
        train, val, test = (report.partitions[p] for p in PARTITIONS)
        # Run 1, steps 10-21 and 40-69 -> train (global 10..69 < 140).
        assert FaultMode("stuck", "thermocouple", "low") in train.covered
        assert FaultMode("stuck", "flow", "high") in train.covered
        # Run 2 offset 100: steps 5-7 -> global 105..107 (train), steps 80-99 -> 180..199 (test).
        assert FaultMode("stuck", "pressure", "medium") in train.covered
        assert FaultMode("stuck", "thermocouple", "low") in test.covered
        assert test.stats[FaultMode("stuck", "thermocouple", "low")].readings == 20
        assert train.stats[FaultMode("stuck", "thermocouple", "low")].readings == 12
        assert val.labelled_readings == 0
        assert val.types_present == ()
        assert train.n_steps + val.n_steps + test.n_steps == report.overall.n_steps == 400
        total = sum(p.labelled_readings for p in report.partitions.values())
        assert total == report.overall.labelled_readings

    def test_partition_steps_are_the_split_arithmetic(self) -> None:
        """Each type contributes int(0.7 n), int(0.15 n) and the rest to test."""
        report = self._collector().report(1, 1, 0.5)
        steps = {name: scope.n_steps for name, scope in report.partitions.items()}
        # stuck (200 steps): 140 / 30 / 30; excursion (100): 70 / 15 / 15; normal (100): 70/15/15.
        assert steps == {"train": 280, "val": 60, "test": 60}

    def test_passed_needs_types_and_grid(self) -> None:
        """The D6 gate is 8/8 types AND the grid gate."""
        report = self._collector().report(1, 1, 0.01)
        assert report.grid_ok
        assert not report.types_ok
        assert not report.passed

    def test_run_order_is_numeric_not_insertion(self) -> None:
        """Runs are laid out by run id: adding run 2 first does not change the timeline."""
        a, b = CoverageCollector(), CoverageCollector()
        first = synth_frame("sim_run_001", "stuck", 100, [Ep("stuck", "low", 90, 10, "TC-A")])
        second = synth_frame("sim_run_002", "stuck", 100, [Ep("stuck", "low", 0, 10, "TC-A")])
        a.add_frame(first)
        a.add_frame(second)
        b.add_frame(second)
        b.add_frame(first)
        ra, rb = a.report(1, 1, 0.5), b.report(1, 1, 0.5)
        assert {k: v.labelled_readings for k, v in ra.partitions.items()} == {
            k: v.labelled_readings for k, v in rb.partitions.items()
        }
        assert ra.partitions["train"].labelled_readings == 20

    def test_errors(self) -> None:
        """Empty collector, repeated run, multi-run frame and bad thresholds."""
        empty = CoverageCollector()
        with pytest.raises(ValueError, match="at least one run"):
            empty.report(1, 1, 0.5)
        collector = CoverageCollector()
        frame = synth_frame("sim_run_001", "stuck", 10, [])
        collector.add_frame(frame)
        with pytest.raises(ValueError, match="already added"):
            collector.add_frame(frame)
        both = pd.concat([frame, synth_frame("sim_run_002", "stuck", 10, [])])
        with pytest.raises(ValueError, match="exactly one run"):
            CoverageCollector().add_frame(both)
        with pytest.raises(ValueError, match="k_min_readings"):
            collector.report(0, 1, 0.5)
        with pytest.raises(ValueError, match="gate"):
            collector.report(1, 1, 1.5)
        with pytest.raises(ValueError, match="ratios"):
            collector.report(1, 1, 0.5, ratios=(0.5, 0.5, 0.5))
        assert collector.n_runs == 1

    def test_normal_only_has_no_types(self) -> None:
        """A dataset without faults has no modes covered."""
        report = coverage_report([synth_frame("sim_run_001", "normal", 20, [])], 1, 1, 0.5)
        assert report.overall.n_types == 0
        assert report.overall.covered == ()
        assert report.overall.grid_fraction == 0.0
        assert report.partitions["test"].stats == {}

    def test_explicit_groups_define_the_grid(self) -> None:
        """The grid can be given (all the groups of the catalog, seen or not)."""
        frame = synth_frame("sim_run_001", "normal", 20, [])
        report = coverage_report([frame], 1, 1, 0.5, groups=["a", "b"])
        assert report.overall.grid_size == 8 * 2 * 3

    def test_mode_stats_defaults(self) -> None:
        """An empty mode is all zeros."""
        assert ModeStats() == ModeStats(readings=0, episodes=0, runs=0, corrupted=0)


@pytest.fixture(scope="module")
def simulator(tmp_path_factory: pytest.TempPathFactory) -> ReactorSimulator:
    """Return a simulator over 30-step runs."""
    return build_simulator(tmp_path_factory.mktemp("coverage"))


class TestPlanEqualsFrame:
    """Measuring the labels of a plan is measuring the labels of its frame."""

    def _plans(self, simulator: ReactorSimulator) -> list[RunPlan]:
        """Return a normal run, a process run, and three instrument runs."""
        params = simulator.params
        rng = np.random.default_rng(0)
        tiers = simulator.scenario_config.instrument
        process = make_run_plan(
            params,
            1,
            ScenarioKind.CORE_TEMP_EXCURSION,
            process_fault=ProcessFaultSpec(
                level=FaultLevel.MEDIUM,
                injected_step=8,
                onset_steps=2,
                hold_steps=6,
                recovery_steps=3,
            ),
        )
        stuck = make_run_plan(
            params,
            2,
            ScenarioKind.STUCK,
            instrument_faults=(
                InstrumentEpisode("TC-HL-1", InstrumentFaultKind.STUCK, FaultLevel.LOW, 7, 9),
                InstrumentEpisode("FT-HL-1", InstrumentFaultKind.STUCK, FaultLevel.HIGH, 9, 11),
            ),
        )
        spikes = make_run_plan(
            params,
            3,
            ScenarioKind.NOISE_SPIKE,
            instrument_faults=tuple(
                build_episode(
                    tiers.for_kind(InstrumentFaultKind.NOISE_SPIKE),
                    FaultLevel.MEDIUM,
                    "PT-PRZ-01",
                    start,
                    params.run_length_steps,
                    rng,
                )
                for start in (7, 12, 20)
            ),
        )
        intermittent = make_run_plan(
            params,
            4,
            ScenarioKind.INTERMITTENT,
            instrument_faults=(
                InstrumentEpisode(
                    "ROD-POS-01",
                    InstrumentFaultKind.INTERMITTENT,
                    FaultLevel.HIGH,
                    6,
                    14,
                    p_enter=0.2,
                    p_exit=0.3,
                ),
            ),
        )
        return [make_run_plan(params, 0, ScenarioKind.NORMAL), process, stuck, spikes, intermittent]

    def test_same_report(self, simulator: ReactorSimulator) -> None:
        """Readings, episodes, runs and partitions agree between plan and frame."""
        plans = self._plans(simulator)
        from_frames, from_plans = CoverageCollector(), CoverageCollector()
        for plan in plans:
            from_frames.add_frame(simulator.generate_run(plan))
            from_plans.add_plan(plan, simulator.catalog)
        a = from_frames.report(1, 1, 0.5)
        b = from_plans.report(1, 1, 0.5)
        assert a.overall.types_present == b.overall.types_present
        assert a.overall.labelled_readings == b.overall.labelled_readings > 0
        for mode, stats in a.overall.stats.items():
            other = b.overall.stats[mode]
            assert (stats.readings, stats.episodes, stats.runs) == (
                other.readings,
                other.episodes,
                other.runs,
            )
        assert set(a.overall.stats) == set(b.overall.stats)
        for name in PARTITIONS:
            assert a.partitions[name].labelled_readings == b.partitions[name].labelled_readings
            assert a.partitions[name].covered == b.partitions[name].covered
        # Only the frame knows which intermittent samples fell in the invalid state.
        assert sum(s.corrupted for s in a.overall.stats.values()) > 0
        assert sum(s.corrupted for s in b.overall.stats.values()) == 0

    def test_add_plan_errors(self, simulator: ReactorSimulator) -> None:
        """A repeated run or an unknown sensor is an error."""
        collector = CoverageCollector()
        plan = make_run_plan(simulator.params, 0, ScenarioKind.NORMAL)
        collector.add_plan(plan, simulator.catalog)
        with pytest.raises(ValueError, match="already added"):
            collector.add_plan(plan, simulator.catalog)
        ghost = make_run_plan(
            simulator.params,
            1,
            ScenarioKind.STUCK,
            instrument_faults=(
                InstrumentEpisode("NOPE-1", InstrumentFaultKind.STUCK, FaultLevel.LOW, 7, 9),
            ),
        )
        with pytest.raises(KeyError, match="NOPE-1"):
            collector.add_plan(ghost, simulator.catalog)

    def test_persistent_process_fault_runs_to_the_end(self, simulator: ReactorSimulator) -> None:
        """A fault with no recovery labels every sensor until the last step."""
        plan = make_run_plan(
            simulator.params,
            0,
            ScenarioKind.VOID_FORMATION,
            process_fault=ProcessFaultSpec(
                level=FaultLevel.HIGH, injected_step=10, onset_steps=2, hold_steps=None
            ),
        )
        collector = CoverageCollector()
        collector.add_plan(plan, simulator.catalog)
        stats = collector.report(1, 1, 0.5).overall.stats
        total = sum(s.readings for s in stats.values())
        assert total == (30 - 10) * len(simulator.catalog.sensors)
