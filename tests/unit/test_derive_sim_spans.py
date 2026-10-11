"""Unit tests for data.generators.derive_sim_spans.

El fichero de spans es lo unico que impide que el validador de Fase 2 quede ciego a
un sensor (H7): hay un test que falla si un sensor del catalogo versionado no tiene
span, otro que comprueba que los spans derivan de una semilla propia, y los de la
aritmetica de los margenes (la misma que `derive_sensor_spans`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.generators.derive_sensor_spans import (
    DEFAULT_ALARM_MARGIN,
    DEFAULT_SPAN_MARGIN,
    MIN_ABSOLUTE_WIDTH,
)
from data.generators.derive_sim_spans import (
    N_SPAN_RUNS,
    SPANS_SEED,
    derive_spans,
    main,
    observe_normal_operation,
    render_yaml,
)
from data.schemas.sensor_spans import load_sensor_spans
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.simulation_params import SimulationParams, load_simulation_params
from tests.support.sim_frames import (
    SIM_SPANS_PATH,
    build_catalog,
    build_model,
    write_sim_params,
)


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned sensor catalog."""
    return build_catalog()


@pytest.fixture(scope="module")
def params(tmp_path_factory: pytest.TempPathFactory) -> SimulationParams:
    """Return a small configuration (runs of 30 steps)."""
    return load_simulation_params(write_sim_params(tmp_path_factory.mktemp("sim")))


class TestVersionedSpans:
    """The committed span file covers the committed catalog."""

    def test_every_catalog_sensor_has_a_span(self, catalog: SensorCatalog) -> None:
        """A sensor without span would pass unvalidated (H7)."""
        spans = load_sensor_spans(SIM_SPANS_PATH)
        assert not [tag for tag in catalog.tags if tag not in spans]

    def test_no_orphan_spans(self, catalog: SensorCatalog) -> None:
        """The span file does not carry sensors the catalog dropped."""
        spans = load_sensor_spans(SIM_SPANS_PATH)
        assert set(spans) == set(catalog.tags)

    def test_units_match_the_catalog(self, catalog: SensorCatalog) -> None:
        """Each span is labelled with the unit of its sensor."""
        spans = load_sensor_spans(SIM_SPANS_PATH)
        for sensor in catalog.sensors:
            assert spans[sensor.tag].unit == sensor.unit.value

    def test_alarm_envelope_sits_inside_the_span(self) -> None:
        """load_sensor_spans enforces it; this documents the committed file honours it."""
        for span in load_sensor_spans(SIM_SPANS_PATH).values():
            assert span.min < span.alarm_min < span.alarm_max < span.max


class TestObservation:
    """The observed runs are normal operation under a seed of their own."""

    def test_shape(self, params: SimulationParams, catalog: SensorCatalog) -> None:
        """N_SPAN_RUNS runs of run_length_steps samples, one column per sensor."""
        observed = observe_normal_operation(params, build_model(), catalog)
        assert observed.shape == (N_SPAN_RUNS * params.run_length_steps, len(catalog.sensors))

    def test_is_deterministic(self, params: SimulationParams, catalog: SensorCatalog) -> None:
        """The same seed gives the same observation."""
        a = observe_normal_operation(params, build_model(), catalog, n_runs=2)
        b = observe_normal_operation(params, build_model(), catalog, n_runs=2)
        assert np.array_equal(a, b)

    def test_seed_is_independent_of_the_dataset_seed(
        self, params: SimulationParams, catalog: SensorCatalog
    ) -> None:
        """The span runs are not the dataset runs: the validator is not tested on its own sample."""
        assert SPANS_SEED != params.openmc_seed
        own = observe_normal_operation(params, build_model(), catalog, n_runs=1)
        dataset_seed = params.openmc_seed
        other = observe_normal_operation(
            params, build_model(), catalog, n_runs=1, seed=dataset_seed
        )
        assert not np.array_equal(own, other)


class TestDerivation:
    """The envelope arithmetic is the one of the TEP span derivation."""

    def test_margins(self, catalog: SensorCatalog) -> None:
        """Alarm and span are the observed range widened by their margins."""
        observed = np.zeros((4, len(catalog.sensors)))
        observed[:, 0] = [10.0, 12.0, 20.0, 15.0]
        spans = derive_spans(observed, catalog)
        first = spans[catalog.tags[0]]
        assert first["alarm_min"] == pytest.approx(10.0 - 10.0 * DEFAULT_ALARM_MARGIN)
        assert first["alarm_max"] == pytest.approx(20.0 + 10.0 * DEFAULT_ALARM_MARGIN)
        assert first["min"] == pytest.approx(10.0 - 10.0 * DEFAULT_SPAN_MARGIN)
        assert first["max"] == pytest.approx(20.0 + 10.0 * DEFAULT_SPAN_MARGIN)

    def test_constant_channel_gets_the_minimum_width(self, catalog: SensorCatalog) -> None:
        """A channel that does not move in normal operation is given a finite span."""
        observed = np.full((5, len(catalog.sensors)), 7.0)
        spans = derive_spans(observed, catalog)
        entry = spans[catalog.tags[3]]
        assert entry["max"] - entry["min"] == pytest.approx(MIN_ABSOLUTE_WIDTH)
        assert entry["alarm_min"] < 7.0 < entry["alarm_max"]

    def test_units_and_order(self, catalog: SensorCatalog) -> None:
        """Entries follow the catalog order and carry the unit."""
        observed = np.random.default_rng(0).normal(size=(10, len(catalog.sensors)))
        spans = derive_spans(observed, catalog)
        assert list(spans) == list(catalog.tags)
        assert spans["TC-HL-1"]["unit"] == "celsius"

    def test_negative_margin(self, catalog: SensorCatalog) -> None:
        """Margins cannot be negative."""
        with pytest.raises(ValueError, match="non-negative"):
            derive_spans(np.zeros((2, len(catalog.sensors))), catalog, alarm_margin=-0.1)

    def test_span_narrower_than_alarm(self, catalog: SensorCatalog) -> None:
        """The alarm envelope must live inside the span."""
        with pytest.raises(ValueError, match="at least alarm_margin"):
            derive_spans(
                np.zeros((2, len(catalog.sensors))), catalog, alarm_margin=0.5, span_margin=0.1
            )

    def test_wrong_observation_shape(self, catalog: SensorCatalog) -> None:
        """The observation must have one column per sensor."""
        with pytest.raises(ValueError, match="shape"):
            derive_spans(np.zeros((5, 3)), catalog)


class TestRenderAndMain:
    """The YAML is loadable and the entry point writes it."""

    def test_rendered_yaml_round_trips(self, catalog: SensorCatalog, tmp_path: Path) -> None:
        """The document is readable by load_sensor_spans and records its provenance."""
        observed = np.random.default_rng(1).normal(100.0, 5.0, size=(50, len(catalog.sensors)))
        spans = derive_spans(observed, catalog)
        text = render_yaml(spans, 5, 1500, SPANS_SEED, 0.2, 2.0)
        assert f"{SPANS_SEED}" in text
        assert "SPANS_SEED" in text
        path = tmp_path / "spans.yaml"
        path.write_text(text, encoding="utf-8")
        assert set(load_sensor_spans(path)) == set(catalog.tags)

    def test_main_writes_the_span_file(self, tmp_path: Path) -> None:
        """main() derives from the params file and writes the output."""
        params_path = write_sim_params(tmp_path)
        output = tmp_path / "out" / "spans.yaml"
        result = main(["--params", str(params_path), "--output", str(output)])
        assert result == output
        assert len(load_sensor_spans(output)) == 27

    def test_main_defaults_to_params_spans_path(self, tmp_path: Path) -> None:
        """Without --output the table goes to simulation.spans_path."""
        target = tmp_path / "default_spans.yaml"
        params_path = write_sim_params(tmp_path, spans_path=str(target))
        assert main(["--params", str(params_path)]) == target
        assert target.exists()
