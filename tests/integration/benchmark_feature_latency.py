"""Latency of the feature pipeline over one window (criterio 3 de la Fase 2).

Criterio: menos de 100 ms por ventana BAJO CARGA. La unidad medida es una llamada a
`FeaturePipeline.transform()` sobre una ventana de `longest_window` muestras (60, la
mas larga de params.yaml) por los 52 sensores del TEP: es lo que un servicio en linea
tendria que calcular al llegar cada muestra, y es lo unico que existe hoy, porque no
hay camino de features en linea (el pipeline es por lotes sobre una ventana contigua).

Metodologia:
  - Las ventanas son tiradas reales de d00 (60 timesteps contiguos x 52 sensores),
    desplazadas de una en una a lo largo de la corrida, y se repiten `--repeats` veces.
    Cada llamada se cronometra por separado (perf_counter_ns); se reportan p50, p95,
    p99, maximo y media. El valor del criterio es el p99.
  - El pivot y el troceado de las ventanas quedan FUERA de la medida: solo cuenta
    `transform()`.
  - Un calentamiento de 10 llamadas (imports perezosos, cache de numpy) se descarta.
  - "Bajo carga": con `--load-workers N` se lanzan N procesos que validan lecturas
    reales sin parar (el trabajo de un pod de validacion) mientras se mide, de modo que
    compiten por la CPU. Se reporta cuantas lecturas por segundo consiguio validar la
    carga, para que se vea que estuvo activa. Puede combinarse con el benchmark de Kafka
    en otra terminal para cargar tambien el transporte.

Una medicion `--environment local` es INFORMATIVA; solo `cluster` cierra el criterio.

Uso (desde la raiz del repositorio):

    .\\.venv\\Scripts\\python.exe -m tests.integration.benchmark_feature_latency
    .\\.venv\\Scripts\\python.exe -m tests.integration.benchmark_feature_latency --load-workers 4

Salida: tests/results/feature_latency.json.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import platform
import statistics
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.generators.tep_adapter import readings_from_frame
from data.schemas.sensor_spans import DEFAULT_SPANS_PATH, load_sensor_spans
from data.validation.sensor_validator import SensorValidator
from ml.features.batch_featurizer import measure_sample_interval
from ml.features.feature_params import DEFAULT_PARAMS_PATH, FeatureParams, load_feature_params
from ml.features.pipeline import FeaturePipeline
from tests.integration.benchmark_report import ENVIRONMENTS, build_report, write_report

_LOG = logging.getLogger("benchmark_feature_latency")

CRITERION = "feature_latency"
UNIT = "milliseconds"
LATENCY_THRESHOLD_MS = 100.0
"""Criterio de la Fase 2 (TDD): milisegundos por ventana, medidos en el p99."""

DEFAULT_REPEATS = 1
DEFAULT_WARMUP_CALLS = 10
DEFAULT_OUTPUT = Path("tests/results/feature_latency.json")
_PARTITION = Path("data/processed/tep/fault_type=00/readings.parquet")
_LOAD_SETTLE_S = 1.0
_LOAD_JOIN_S = 10.0

Window = tuple[pd.DataFrame, pd.Series]
"""(valores anchos, timestamps) de una ventana contigua."""


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Return a percentile of an ascending series, linearly interpolated.

    Args:
        sorted_values: Values in ascending order.
        pct: Percentile between 0 and 100.

    Returns:
        The interpolated percentile.

    Raises:
        ValueError: If the series is empty or pct is outside [0, 100].
    """
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty series.")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be within [0, 100], got {pct}.")
    index = pct / 100.0 * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Summarize call latencies.

    Args:
        latencies_ms: One latency per call, in milliseconds.

    Returns:
        count, mean, p50, p95, p99 and max, all in milliseconds (count excepted).

    Raises:
        ValueError: If there are no latencies.
    """
    ordered = sorted(latencies_ms)
    return {
        "count": float(len(ordered)),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(ordered, 50.0),
        "p95_ms": percentile(ordered, 95.0),
        "p99_ms": percentile(ordered, 99.0),
        "max_ms": float(ordered[-1]),
    }


def sliding_windows(frame: pd.DataFrame, window_samples: int) -> Iterator[Window]:
    """Cut a long-format run into contiguous windows, one step apart.

    Args:
        frame: Long-format readings of one run (timestep, sensor_id, value, timestamp).
        window_samples: Timesteps per window.

    Yields:
        (values, timestamps) of each window, indexed by timestep as
        FeaturePipeline.transform expects.

    Raises:
        ValueError: If the run has fewer timesteps than a window.
    """
    wide = frame.pivot(index="timestep", columns="sensor_id", values="value")
    stamps = (
        frame.drop_duplicates(subset="timestep")
        .set_index("timestep")
        .sort_index()
        .loc[:, "timestamp"]
    )
    if len(wide) < window_samples:
        raise ValueError(
            f"The run has {len(wide)} timesteps, fewer than the window of {window_samples}."
        )
    for start in range(len(wide) - window_samples + 1):
        rows = wide.index[start : start + window_samples]
        yield wide.loc[rows], stamps.loc[rows]


def time_transform(
    pipeline: FeaturePipeline,
    windows: Sequence[Window],
    sensor_types: Mapping[str, str],
    *,
    repeats: int,
    warmup_calls: int,
) -> list[float]:
    """Time every transform() call.

    Args:
        pipeline: Pipeline under test.
        windows: Windows to transform, in order.
        sensor_types: Sensor tag to type, as transform requires.
        repeats: Passes over all the windows.
        warmup_calls: Initial calls whose time is discarded.

    Returns:
        One latency in milliseconds per measured call.

    Raises:
        ValueError: If there are no windows or repeats is not positive.
    """
    if not windows or repeats <= 0:
        raise ValueError("windows must not be empty and repeats must be positive.")
    for index in range(warmup_calls):
        values, stamps = windows[index % len(windows)]
        pipeline.transform(values, stamps, sensor_types)
    latencies: list[float] = []
    for _ in range(repeats):
        for values, stamps in windows:
            started = time.perf_counter_ns()
            pipeline.transform(values, stamps, sensor_types)
            latencies.append((time.perf_counter_ns() - started) / 1e6)
    return latencies


def _load_worker(partition: str, stop: EventType, counter: Synchronized[int]) -> None:
    """Validate real readings in a loop until told to stop (the background load).

    Args:
        partition: Parquet file of the run whose readings are validated.
        stop: Set by the parent when the measurement ends.
        counter: Shared count of readings validated by this worker.
    """
    spans = load_sensor_spans(DEFAULT_SPANS_PATH)
    frame = pd.read_parquet(partition).sort_values(["timestep", "sensor_id"], kind="stable")
    readings = list(readings_from_frame(frame))
    while not stop.is_set():
        # Estado limpio en cada vuelta: el reloj del TEP se reinicia al repetir la corrida.
        validator = SensorValidator(spans)
        for reading in readings:
            if stop.is_set():
                return
            validator.validate(reading)
            with counter.get_lock():
                counter.value += 1


def start_load(
    workers: int, partition: Path
) -> tuple[list[Any], EventType, list[Synchronized[int]]]:
    """Start the background validation workers.

    Args:
        workers: Number of processes.
        partition: Parquet file the workers validate.

    Returns:
        The processes, the stop event and one shared counter per worker.
    """
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    counters: list[Synchronized[int]] = [context.Value("q", 0) for _ in range(workers)]
    processes = [
        context.Process(target=_load_worker, args=(str(partition), stop, counter), daemon=True)
        for counter in counters
    ]
    for process in processes:
        process.start()
    return processes, stop, counters


def stop_load(processes: Sequence[Any], stop: EventType) -> None:
    """Stop the background workers and wait for them.

    Args:
        processes: Processes from start_load.
        stop: Their stop event.
    """
    stop.set()
    for process in processes:
        process.join(timeout=_LOAD_JOIN_S)
        if process.is_alive():
            process.terminate()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments; sys.argv[1:] when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Feature pipeline latency per window.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup-calls", type=int, default=DEFAULT_WARMUP_CALLS)
    parser.add_argument(
        "--load-workers",
        type=int,
        default=0,
        help="validator processes competing for the CPU while measuring (0 = idle machine)",
    )
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="local")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--partition", type=Path, default=_PARTITION)
    parser.add_argument(
        "--fail-above-threshold",
        action="store_true",
        help="exit 2 when the p99 does not meet the criterion (default: exit 0)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its report.

    Args:
        argv: Command-line arguments; sys.argv[1:] when omitted.

    Returns:
        0 when the measurement completed, 2 when --fail-above-threshold is set and
        the p99 is above the criterion.

    Raises:
        FileNotFoundError: If the parquet run is not populated.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if not args.partition.exists():
        raise FileNotFoundError(
            f"{args.partition} not found. Generate it with .\\infra\\scripts\\Invoke-Pipeline.ps1."
        )
    params: FeatureParams = load_feature_params(args.params)
    frame = pd.read_parquet(args.partition)
    interval = measure_sample_interval(frame)
    sensor_types = dict(zip(frame["sensor_id"], frame["sensor_type"], strict=True))
    pipeline = FeaturePipeline(params, sample_interval_seconds=interval)
    window_samples = params.longest_window
    windows = list(sliding_windows(frame, window_samples))
    _LOG.info(
        "%d windows of %d samples x %d sensors, %d repeats, load workers: %d",
        len(windows), window_samples, frame["sensor_id"].nunique(), args.repeats,
        args.load_workers,
    )

    processes: list[Any] = []
    stop: EventType | None = None
    counters: list[Synchronized[int]] = []
    if args.load_workers > 0:
        processes, stop, counters = start_load(args.load_workers, args.partition)
        time.sleep(_LOAD_SETTLE_S)
    load_started = time.perf_counter()
    try:
        latencies = time_transform(
            pipeline, windows, sensor_types,
            repeats=args.repeats, warmup_calls=args.warmup_calls,
        )
    finally:
        load_elapsed = time.perf_counter() - load_started
        if stop is not None:
            stop_load(processes, stop)

    summary = summarize(latencies)
    validated = sum(counter.value for counter in counters)
    report = build_report(
        criterion=CRITERION,
        environment=args.environment,
        value=round(summary["p99_ms"], 3),
        unit=UNIT,
        threshold=LATENCY_THRESHOLD_MS,
        direction="at_most",
        details={
            "statistic": "p99",
            "calls": int(summary["count"]),
            "mean_ms": round(summary["mean_ms"], 3),
            "p50_ms": round(summary["p50_ms"], 3),
            "p95_ms": round(summary["p95_ms"], 3),
            "p99_ms": round(summary["p99_ms"], 3),
            "max_ms": round(summary["max_ms"], 3),
            "window_samples": window_samples,
            "sensors": int(frame["sensor_id"].nunique()),
            "features_per_sensor": len(pipeline.feature_names()),
            "sample_interval_s": interval,
            "repeats": args.repeats,
            "warmup_calls": args.warmup_calls,
            "payload_source": str(args.partition),
            "load": {
                "workers": args.load_workers,
                "readings_validated": validated,
                "readings_validated_per_s": (
                    round(validated / load_elapsed, 1) if args.load_workers > 0 else 0.0
                ),
            },
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpus": os.cpu_count(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    )
    write_report(args.output, report)
    _LOG.info(
        "p50 %.2f ms, p95 %.2f ms, p99 %.2f ms, max %.2f ms (load workers %d), "
        "environment=%s, threshold %.0f ms -> %s. Report: %s",
        summary["p50_ms"], summary["p95_ms"], summary["p99_ms"], summary["max_ms"],
        args.load_workers, args.environment, LATENCY_THRESHOLD_MS,
        "meets" if report["passed"] else "ABOVE", args.output,
    )
    if args.fail_above_threshold and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
