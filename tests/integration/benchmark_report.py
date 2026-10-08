"""Shared result format of the Phase 2 benchmarks.

`benchmark_kafka.py` (criterio 1) y `benchmark_feature_latency.py` (criterio 3)
escriben el mismo esquema de JSON para que `Verify-Phase2.ps1` (M9) los lea igual:

    {
      "schema_version": 1,
      "criterion": "kafka_throughput" | "feature_latency" | "kafka_latency",
      "environment": "local" | "cluster",
      "measured_at": "2026-10-07T12:00:00Z",
      "commit": "a428e71" | "a428e71-dirty" | "unknown",
      "value": <number>,
      "unit": "messages_per_second" | "milliseconds",   (kafka_latency: criterio de la Fase 1)
      "threshold": <number>,
      "direction": "at_least" | "at_most",
      "passed": <bool>,
      "details": { ... }
    }

`environment` es lo que decide si la cifra cierra el criterio: una medicion `local` es
informativa y solo una `cluster` lo da por CUMPLIDO o FALLADO. `direction` dice como se
compara el valor con el umbral (el throughput debe SUPERARLO, la latencia NO EXCEDERLO).
Los resultados van a tests/results/, que esta en .gitignore: no se versionan.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ENVIRONMENTS = ("local", "cluster")
DIRECTIONS = ("at_least", "at_most")
_GIT_TIMEOUT_S = 10


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a Z suffix.

    Returns:
        The timestamp, to the second.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(root: Path | None = None) -> str:
    """Return the abbreviated HEAD commit, marked when the tree has changes.

    Args:
        root: Repository directory; defaults to the current directory.

    Returns:
        The short hash, the hash plus "-dirty" when the working tree differs from
        HEAD, or "unknown" when git is unavailable or the directory is not a repo.
    """
    cwd = None if root is None else str(root)
    try:
        head = subprocess.run(  # noqa: S603  (argumentos fijos, sin entrada externa)
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_S,
            cwd=cwd,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_S,
            cwd=cwd,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not head:
        return "unknown"
    return f"{head}-dirty" if status else head


def meets_threshold(value: float, threshold: float, direction: str) -> bool:
    """Compare a measurement with its threshold.

    Args:
        value: Measured value.
        threshold: Criterion threshold.
        direction: "at_least" (value >= threshold) or "at_most" (value <= threshold).

    Returns:
        Whether the criterion is met by this measurement.

    Raises:
        ValueError: If direction is not one of DIRECTIONS.
    """
    if direction == "at_least":
        return value >= threshold
    if direction == "at_most":
        return value <= threshold
    raise ValueError(f"direction must be one of {DIRECTIONS}, got '{direction}'.")


def build_report(
    *,
    criterion: str,
    environment: str,
    value: float,
    unit: str,
    threshold: float,
    direction: str,
    details: dict[str, Any],
    commit: str | None = None,
    measured_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a result document in the shared format.

    Args:
        criterion: Name of the Phase 2 criterion measured.
        environment: "local" or "cluster".
        value: The measured figure.
        unit: Unit of value and threshold.
        threshold: Criterion threshold.
        direction: "at_least" or "at_most".
        details: Free-form supporting numbers and the settings of the run.
        commit: Commit to record; resolved from git when omitted.
        measured_at: Timestamp to record; the current time when omitted.

    Returns:
        The JSON-serialisable document.

    Raises:
        ValueError: If environment or direction is not recognised.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {ENVIRONMENTS}, got '{environment}'.")
    return {
        "schema_version": SCHEMA_VERSION,
        "criterion": criterion,
        "environment": environment,
        "measured_at": utc_now_iso() if measured_at is None else measured_at,
        "commit": git_commit() if commit is None else commit,
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "direction": direction,
        "passed": meets_threshold(value, threshold, direction),
        "details": details,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a result document, creating the directory when needed.

    Args:
        path: Destination file.
        report: Document from build_report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
