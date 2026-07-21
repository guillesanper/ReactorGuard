"""Evaluation of the sensor validator over the adapted TEP dataset (T3.6).

Primer criterio de exito medible del proyecto: precision del detector de stuck
por encima de 0,95. Se mide por par (sensor_id, timestep), que es la resolucion a
la que el validador decide, y no por episodio: puntuar por episodio dejaria que
un detector que marca una unica muestra de una ventana de 40 saliera con la misma
nota que uno que las marca todas.

POR QUE INYECCION SINTETICA. El TEP no trae verdad-terreno de fallo de
INSTRUMENTO. Sus 21 ficheros de fallo describen perturbaciones del PROCESO, que
es un eje ortogonal. Medido sobre el dataset, el unico transmisor realmente
congelado es TEP-XMV-04 en d21: 480 pares positivos de 550.160, el 0,087%. El
plan original daba d14 y d15 por positivos de stuck y ESO ES FALSO: d14 es
sticking de valvula (respuesta lenta, la lectura sigue variando) y d15 es
indetectable; el run-length maximo de ambos es 5, el mismo que el de d00. La
clase positiva se construye por tanto sobre d00 degradado a proposito, y d21 se
reserva como validacion cruzada con el unico positivo real que existe.

TRES ASIMETRIAS DE LA MEDICION, declaradas porque cambian como leer los numeros:

  bias_out_of_range sale 1,00 de precision y de recall por construccion, no por
  merito. Su verdad-terreno es "el valor cae fuera del sobre de alarma", que es
  exactamente el predicado que evalua RangeValidator. La cifra no mide deteccion:
  verifica la invariante 2 del validador, que el detector compara contra el sobre
  de alarma y no contra el span calibrado. Si algun dia baja de 1,00, lo que se ha
  roto es esa invariante.

  kalman_anomaly no lleva matriz de confusion. No es un modo de fallo fisico sino
  una senal interna del detector, y etiquetar pares como "aqui deberia haber
  residual anomalo" seria inventarse la verdad. Se reportan en su lugar las dos
  cifras que si significan algo: su tasa de disparo sobre sensores intactos y la
  fraccion de episodios inyectados en los que dispara al menos una vez.

  drift_correlated no se evalua. CrossCorrelationChecker viene sin parejas base y
  con el diccionario vacio no comprueba nada; derivar esa tabla del TEP es trabajo
  que nadie ha hecho todavia.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.generators.tep_adapter import readings_from_frame
from data.schemas.sensor_spans import DEFAULT_SPANS_PATH, SensorSpan, load_sensor_spans
from data.validation.fault_injector import (
    DEFAULT_INJECTION_SEED,
    InjectionKind,
    InjectionReport,
    TEPFaultInjector,
)
from data.validation.sensor_fault import FaultType
from data.validation.sensor_validator import (
    DEFAULT_STUCK_WINDOW,
    SensorValidator,
    StuckValueDetector,
)

PROCESSED_ROOT = Path("data/processed/tep")
RESULTS_PATH = Path("tests/results/validator_metrics_tep.json")

STUCK_PRECISION_GATE = 0.95
"""Criterio de exito de la Fase 2. No lo bajes para que pase la suite."""

_NORMAL_PARTITION = PROCESSED_ROOT / "fault_type=00" / "readings.parquet"
_STUCK_PARTITION = PROCESSED_ROOT / "fault_type=21" / "readings.parquet"
_FROZEN_TAG = "TEP-XMV-04"
"""Unico transmisor congelado de verdad en el dataset, en d21."""

pytestmark = pytest.mark.skipif(
    not _NORMAL_PARTITION.exists() or not _STUCK_PARTITION.exists(),
    reason=(
        "El parquet adaptado del TEP no esta poblado. Generalo con "
        ".\\infra\\scripts\\Invoke-Pipeline.ps1 antes de medir T3.6."
    ),
)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Confusion:
    """Per-fault-type confusion counted over (sensor_id, timestep) pairs.

    Attributes:
        fault_type: Failure mode being scored.
        true_positives: Labelled pairs the validator flagged.
        false_positives: Flagged pairs that are neither labelled nor in a guard
            band. La banda de guarda queda fuera de la clase negativa porque
            junto a una degradacion inyectada la senal esta genuinamente
            perturbada y que detector dispara no esta determinado.
        false_negatives: Labelled pairs the validator missed.
        negatives: Size of the negative class actually scored.
    """

    fault_type: FaultType
    true_positives: int
    false_positives: int
    false_negatives: int
    negatives: int

    @property
    def precision(self) -> float:
        """Return TP / (TP + FP), or 1.0 when the detector never fired."""
        flagged = self.true_positives + self.false_positives
        return 1.0 if flagged == 0 else self.true_positives / flagged

    @property
    def recall(self) -> float:
        """Return TP / (TP + FN), or 1.0 when there was nothing to find."""
        positives = self.true_positives + self.false_negatives
        return 1.0 if positives == 0 else self.true_positives / positives

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 0.0 if total == 0.0 else 2.0 * self.precision * self.recall / total

    @property
    def false_positive_rate(self) -> float:
        """Return FP / negatives, the share of clean pairs wrongly flagged."""
        return 0.0 if self.negatives == 0 else self.false_positives / self.negatives

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the matrix and its derived rates."""
        return {
            "fault_type": self.fault_type.value,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "negatives": self.negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TEPValidatorEvaluator:
    """Streams a long-format TEP frame through the validator and scores the result.

    Las lecturas se emiten en orden (timestep, sensor_id), que es el orden en que
    llegarian de la planta. Importa: los detectores tienen estado por sensor y lo
    construyen segun van llegando, de modo que reordenar el frame cambia lo que ven.
    """

    def __init__(self, spans: dict[str, SensorSpan]) -> None:
        """Build the evaluator.

        Args:
            spans: Calibrated span table covering the TEP tags.
        """
        self.spans = spans
        self.validator = SensorValidator(spans)
        self.predictions: dict[FaultType, set[tuple[str, int]]] = defaultdict(set)
        self.pairs: set[tuple[str, int]] = set()

    def run(self, frame: pd.DataFrame) -> None:
        """Validate every reading of the frame and record what fired where.

        Args:
            frame: Long-format frame with sensor_id, timestep, timestamp and value.
        """
        ordered = frame.sort_values(["timestep", "sensor_id"], kind="stable")
        timestep_of = dict(
            zip(ordered["timestamp"], ordered["timestep"], strict=True)
        )
        self.pairs = {
            (str(sensor), int(step))
            for sensor, step in zip(ordered["sensor_id"], ordered["timestep"], strict=True)
        }

        for reading in readings_from_frame(ordered):
            for fault in self.validator.validate(reading).faults:
                step = int(timestep_of[pd.Timestamp(fault.detected_at)])
                self.predictions[fault.fault_type].add((fault.sensor_id, step))

    def predicted(self, fault_type: FaultType) -> set[tuple[str, int]]:
        """Return the pairs flagged with one fault type.

        Args:
            fault_type: Failure mode.

        Returns:
            The flagged (sensor_id, timestep) pairs.
        """
        return self.predictions.get(fault_type, set())

    def confusion(
        self,
        fault_type: FaultType,
        truth: frozenset[tuple[str, int]],
        guard: frozenset[tuple[str, int]],
    ) -> Confusion:
        """Score one fault type against its ground truth.

        Args:
            fault_type: Failure mode.
            truth: Labelled positive pairs.
            guard: Pairs excluded from the negative class.

        Returns:
            The confusion matrix.
        """
        predicted = self.predicted(fault_type)
        scored_negatives = self.pairs - set(truth) - set(guard)
        return Confusion(
            fault_type=fault_type,
            true_positives=len(predicted & truth),
            false_positives=len(predicted & scored_negatives),
            false_negatives=len(truth - predicted),
            negatives=len(scored_negatives),
        )


def _longest_run(values: list[float]) -> int:
    """Return the longest run of identical consecutive values.

    Args:
        values: Series in sample order.

    Returns:
        Length of the longest run, 0 for an empty series.
    """
    longest = 0
    run = 0
    previous: float | None = None
    for value in values:
        run = run + 1 if previous is not None and value == previous else 1
        previous = value
        longest = max(longest, run)
    return longest


def _runs_by_sensor(frame: pd.DataFrame) -> dict[str, int]:
    """Return the longest identical-value run of every sensor in a frame.

    Args:
        frame: Long-format frame.

    Returns:
        Mapping from sensor tag to its longest run.
    """
    ordered = frame.sort_values(["sensor_id", "timestep"], kind="stable")
    return {
        str(sensor): _longest_run(list(group["value"]))
        for sensor, group in ordered.groupby("sensor_id")
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tep_spans() -> dict[str, SensorSpan]:
    """Load the committed span table."""
    return load_sensor_spans(DEFAULT_SPANS_PATH)


@pytest.fixture(scope="module")
def normal_frame() -> pd.DataFrame:
    """Return d00, the normal-operation partition, straight from the parquet."""
    return pd.read_parquet(_NORMAL_PARTITION)


@pytest.fixture(scope="module")
def stuck_frame() -> pd.DataFrame:
    """Return d21, the partition holding the only genuinely frozen transmitter."""
    return pd.read_parquet(_STUCK_PARTITION)


@pytest.fixture(scope="module")
def injection(
    tep_spans: dict[str, SensorSpan], normal_frame: pd.DataFrame
) -> InjectionReport:
    """Degrade d00 with the fixed seed and return the frame plus its ground truth."""
    return TEPFaultInjector(tep_spans, seed=DEFAULT_INJECTION_SEED).inject(normal_frame)


@pytest.fixture(scope="module")
def evaluation(
    tep_spans: dict[str, SensorSpan], injection: InjectionReport
) -> TEPValidatorEvaluator:
    """Run the full default validator over the degraded d00."""
    evaluator = TEPValidatorEvaluator(tep_spans)
    evaluator.run(injection.frame)
    return evaluator


@pytest.fixture(scope="module")
def d21_evaluation(
    tep_spans: dict[str, SensorSpan], stuck_frame: pd.DataFrame
) -> TEPValidatorEvaluator:
    """Run the full default validator over d21, untouched."""
    evaluator = TEPValidatorEvaluator(tep_spans)
    evaluator.run(stuck_frame)
    return evaluator


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class TestStuckPrecisionGate:
    """El criterio de exito de la Fase 2."""

    def test_stuck_precision_meets_the_phase_gate(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """Stuck precision must exceed 0.95 over the injected ground truth."""
        matrix = evaluation.confusion(
            FaultType.STUCK,
            injection.labels_for(FaultType.STUCK),
            injection.guard,
        )
        assert matrix.true_positives > 0, "El gate no significa nada sin positivos."
        assert matrix.precision >= STUCK_PRECISION_GATE, (
            f"stuck precision {matrix.precision:.4f} < {STUCK_PRECISION_GATE}: "
            f"{matrix.false_positives} falsos positivos sobre {matrix.negatives} "
            "pares negativos."
        )

    def test_stuck_never_fires_on_an_untouched_sensor(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """No sensor left alone by the injector may be accused of being frozen.

        Es la comprobacion mas dura del detector: separa "acierta donde hay fallo"
        de "no inventa fallos donde no los hay", que es lo que un operador nota.
        """
        degraded = injection.degraded_sensors()
        spurious = {
            pair for pair in evaluation.predicted(FaultType.STUCK) if pair[0] not in degraded
        }
        assert spurious == set(), (
            f"{len(spurious)} detecciones de stuck sobre sensores intactos, "
            f"empezando por {sorted(spurious)[:5]}."
        )

    def test_every_miss_lies_in_the_head_of_its_episode(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """The recall shortfall must be the detector window, not missed samples.

        Un congelamiento no se puede marcar desde su primera muestra: hasta que la
        tirada no alcanza window_size no hay evidencia de que la senal se haya
        parado. Cada episodio pierde por tanto una cabecera, y eso es lo unico que
        el recall deberia perder. Se comprueba la afirmacion exacta -- ningun
        falso negativo cae en el interior de un episodio -- en lugar de un techo
        agregado: el techo depende ademas de si el valor congelado empalma por
        casualidad con el inmediatamente anterior, que alarga la tirada y adelanta
        la deteccion, y una cota que dependa de esa coincidencia no dice nada.
        """
        truth = injection.labels_for(FaultType.STUCK)
        missed = truth - evaluation.predicted(FaultType.STUCK)
        freezes = {
            (episode.sensor_id, episode.start_timestep, episode.end_timestep)
            for episode in injection.episodes
            if episode.kind is InjectionKind.FREEZE
        }

        assert missed, "Sin fallos el test no comprueba nada; el detector no es exacto."
        for sensor, step in missed:
            head = next(
                start
                for tag, start, end in freezes
                if tag == sensor and start <= step <= end
            )
            assert step - head < DEFAULT_STUCK_WINDOW, (
                f"{sensor} en timestep {step} esta a {step - head} muestras del "
                "inicio de su congelamiento: el detector se salto una muestra del "
                "interior del episodio, no de su cabecera."
            )


# ---------------------------------------------------------------------------
# The other detectors
# ---------------------------------------------------------------------------


class TestRangeDetectorOnTEP:
    """RangeValidator contra la definicion objetiva de fuera-de-sobre."""

    def test_range_detection_is_exact(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """Every out-of-envelope pair, and only those, must be flagged.

        Precision y recall de 1,00 aqui no miden deteccion: verdad-terreno y
        detector comparten predicado. Lo que se verifica es la invariante 2, que
        el detector usa in_alarm_envelope y no contains. Con contains, el margen
        del 200% del span calibrado dejaria casi todas estas excursiones dentro y
        el recall se hundiria.
        """
        matrix = evaluation.confusion(
            FaultType.BIAS_OUT_OF_RANGE,
            injection.labels_for(FaultType.BIAS_OUT_OF_RANGE),
            injection.guard,
        )
        assert matrix.true_positives > 0
        assert matrix.false_positives == 0
        assert matrix.false_negatives == 0

    def test_clean_d00_has_no_out_of_envelope_readings(
        self, tep_spans: dict[str, SensorSpan], normal_frame: pd.DataFrame
    ) -> None:
        """Normal operation must sit entirely inside its own alarm envelope.

        Consecuencia de como se derivan los sobres: del minimo y el maximo de d00
        con un margen del 20%. Si esto fallara, o los spans no corresponden a este
        parquet o el sobre se derivo de otra cosa.
        """
        injector = TEPFaultInjector(tep_spans)
        assert injector.out_of_envelope_pairs(normal_frame) == set()


class TestDriftDetectorOnTEP:
    """KalmanResidualDetector en su senal de velocidad, tras la calibracion de T3.6."""

    def test_drift_precision_is_reported(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """The recalibrated threshold must keep drift precision usable.

        Sin gate de fase, pero con un suelo: con el 1,0 provisional anterior a
        esta calibracion la precision era 0,032, porque el umbral disparaba en el
        48% de la operacion normal.
        """
        matrix = evaluation.confusion(
            FaultType.SENSOR_DRIFT,
            injection.labels_for(FaultType.SENSOR_DRIFT),
            injection.guard,
        )
        assert matrix.true_positives > 0
        assert matrix.precision > 0.80
        assert matrix.recall > 0.80

    def test_rate_detector_is_reported(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """RateOfChangeDetector must catch the step edges it is labelled for."""
        matrix = evaluation.confusion(
            FaultType.NOISE_SPIKE,
            injection.labels_for(FaultType.NOISE_SPIKE),
            injection.guard,
        )
        assert matrix.true_positives > 0
        assert matrix.recall > 0.60


class TestKalmanSignalOnTEP:
    """El residual de Kalman, medido como lo que es: una senal, no un modo de fallo."""

    def test_residual_false_positive_rate_on_untouched_sensors(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """The residual must stay quiet enough on clean sensors to be usable.

        A 3 sigmas un residual gaussiano daria 0,27%. Medido sobre los sensores
        intactos de d00 sale bastante mas: los residuales del proceso real tienen
        colas mas pesadas que el modelo cinematico. Por eso este detector aporta
        una senal de apoyo y no un veredicto por si solo, y por eso no lleva gate.
        """
        degraded = injection.degraded_sensors()
        clean_pairs = {pair for pair in evaluation.pairs if pair[0] not in degraded}
        fired = {
            pair
            for pair in evaluation.predicted(FaultType.KALMAN_ANOMALY)
            if pair[0] not in degraded
        }

        assert clean_pairs, "Sin sensores intactos no hay tasa que medir."
        assert len(fired) / len(clean_pairs) < 0.05

    def test_residual_covers_most_injected_episodes(
        self, evaluation: TEPValidatorEvaluator, injection: InjectionReport
    ) -> None:
        """The residual must fire at least once inside a fair share of episodes."""
        flagged = evaluation.predicted(FaultType.KALMAN_ANOMALY)
        covered = sum(
            1
            for episode in injection.episodes
            if any(
                (episode.sensor_id, step) in flagged
                for step in range(episode.start_timestep, episode.end_timestep + 2)
            )
        )
        assert covered / len(injection.episodes) > 0.40


# ---------------------------------------------------------------------------
# Cross-validation against the one real positive
# ---------------------------------------------------------------------------


class TestRealFrozenTransmitter:
    """XMV-04 en d21: el unico transmisor congelado que existe en el dataset."""

    def test_the_frozen_transmitter_is_detected(
        self, d21_evaluation: TEPValidatorEvaluator, stuck_frame: pd.DataFrame
    ) -> None:
        """A detector calibrated on synthetic freezes must find the real one.

        Es la validacion cruzada de toda la calibracion: si los umbrales ajustados
        sobre inyeccion sintetica no transfieren al unico caso real, la inyeccion
        no representa lo que pretende representar.
        """
        flagged = {
            sensor for sensor, _ in d21_evaluation.predicted(FaultType.STUCK)
        }
        assert _FROZEN_TAG in flagged

        steps = stuck_frame["timestep"].nunique()
        detected = sum(
            1 for sensor, _ in d21_evaluation.predicted(FaultType.STUCK)
            if sensor == _FROZEN_TAG
        )
        # Se pierde solo la cabecera de la ventana del detector.
        assert detected == steps - (DEFAULT_STUCK_WINDOW - 1)

    def test_no_other_transmitter_in_d21_is_called_frozen(
        self, d21_evaluation: TEPValidatorEvaluator
    ) -> None:
        """The other 51 tags of d21 are perturbed, not frozen, and must stay clean."""
        flagged = {
            sensor for sensor, _ in d21_evaluation.predicted(FaultType.STUCK)
        }
        assert flagged == {_FROZEN_TAG}

    def test_only_xmv_04_holds_a_run_past_the_detector_window(
        self, stuck_frame: pd.DataFrame
    ) -> None:
        """The ground truth of the real case must be a property of the data itself.

        Se comprueba sobre los valores y no sobre el detector: XMV-04 es el unico
        canal de d21 cuya tirada de valores identicos supera la ventana, y su
        tirada son las 480 muestras enteras.
        """
        runs = _runs_by_sensor(stuck_frame)
        assert runs[_FROZEN_TAG] == stuck_frame["timestep"].nunique()

        others = {
            sensor: run
            for sensor, run in runs.items()
            if sensor != _FROZEN_TAG and run >= DEFAULT_STUCK_WINDOW
        }
        assert others == {}


class TestQuantisationFloor:
    """La invariante 3: la ventana de stuck tiene que estar por encima de 6."""

    def test_clean_d00_runs_stay_below_the_detector_window(
        self, normal_frame: pd.DataFrame
    ) -> None:
        """Healthy quantised channels must not reach the stuck window.

        Es lo que justifica el suelo empirico: la cuantizacion del analizador de
        composiciones produce tiradas de valores identicos en operacion
        perfectamente normal. Si esta medida se acercara a la ventana, la ventana
        habria que subirla.
        """
        worst = max(_runs_by_sensor(normal_frame).values())
        assert worst < DEFAULT_STUCK_WINDOW

    def test_the_detector_window_rejects_the_measured_floor(
        self, tep_spans: dict[str, SensorSpan], normal_frame: pd.DataFrame
    ) -> None:
        """A window at the measured floor must be refused by the constructor."""
        worst = max(_runs_by_sensor(normal_frame).values())
        with pytest.raises(ValueError, match="window_size must exceed 6"):
            StuckValueDetector(tep_spans, window_size=worst)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_metrics_report_is_written(
    evaluation: TEPValidatorEvaluator,
    d21_evaluation: TEPValidatorEvaluator,
    injection: InjectionReport,
) -> None:
    """Write the metrics consumed by CRITERIO 2 of Verify-Phase2.ps1 (T4.7).

    El fichero es un artefacto de la evaluacion, no una entrada: si desaparece se
    regenera corriendo esta suite.
    """
    scored = [
        FaultType.STUCK,
        FaultType.NOISE_SPIKE,
        FaultType.BIAS_OUT_OF_RANGE,
        FaultType.SENSOR_DRIFT,
    ]
    matrices = {
        fault_type: evaluation.confusion(
            fault_type, injection.labels_for(fault_type), injection.guard
        )
        for fault_type in scored
    }

    degraded = injection.degraded_sensors()
    clean_pairs = {pair for pair in evaluation.pairs if pair[0] not in degraded}
    clean_kalman = {
        pair
        for pair in evaluation.predicted(FaultType.KALMAN_ANOMALY)
        if pair[0] not in degraded
    }

    report: dict[str, Any] = {
        "dataset": {
            "source": PROCESSED_ROOT.as_posix(),
            "injected_partition": "fault_type=00",
            "cross_validation_partition": "fault_type=21",
            "injection_seed": DEFAULT_INJECTION_SEED,
            "sample_interval_seconds": injection.sample_interval_seconds,
            "episodes_injected": len(injection.episodes),
            "sensors_degraded": len(degraded),
            "pairs_evaluated": len(evaluation.pairs),
        },
        "gate": {
            "criterion": "stuck sensor precision",
            "threshold": STUCK_PRECISION_GATE,
            "measured": matrices[FaultType.STUCK].precision,
            "passed": matrices[FaultType.STUCK].precision >= STUCK_PRECISION_GATE,
        },
        "detectors": {
            fault_type.value: matrix.to_dict()
            for fault_type, matrix in matrices.items()
        },
        "kalman_residual_signal": {
            "note": (
                "Senal interna del detector, no un modo de fallo fisico: sin "
                "matriz de confusion a proposito."
            ),
            "clean_sensor_firing_rate": len(clean_kalman) / len(clean_pairs),
            "clean_pairs": len(clean_pairs),
        },
        "cross_validation": {
            "sensor": _FROZEN_TAG,
            "partition": "fault_type=21",
            "stuck_pairs_detected": sum(
                1
                for sensor, _ in d21_evaluation.predicted(FaultType.STUCK)
                if sensor == _FROZEN_TAG
            ),
            "other_sensors_flagged_stuck": sorted(
                {
                    sensor
                    for sensor, _ in d21_evaluation.predicted(FaultType.STUCK)
                    if sensor != _FROZEN_TAG
                }
            ),
        },
        "latency_ms": evaluation.validator.get_metrics()["latency_ms"],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    reloaded = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert reloaded["gate"]["passed"] is True
