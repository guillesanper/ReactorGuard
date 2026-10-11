"""Verify-Phase2.ps1 (T4.7): tres estados, tres codigos de salida y su contrato.

El script decide si la Fase 2 esta cerrada, asi que se prueba como el codigo del que
depende la verdad del proyecto: no con mocks de su logica sino ejecutando PowerShell 7
sobre artefactos sinteticos con el formato EXACTO que producen los benchmarks
(`build_report`), pytest (JUnit) y DVC (`dvc.lock`).

Tres grupos:

- Contrato estatico: lo que el script lee (nombres de fichero, criterios, umbrales,
  Deployments, dashboard, Feast) coincide con lo que el resto del repositorio produce.
  Un nombre desalineado no daria un error sino un NO MEDIDO permanente.
- Decisiones (capa 2): cada regla de estado con su caso borde. El que mas importa es
  el de un test saltado, que pytest devuelve con exit code 0.
- Orquestacion: el script completo sobre un proyecto sintetico, con un `dvc` falso y
  con `kubectl` que lanza si alguien lo llama, de modo que ningun test toca un cluster.

Los tests de PowerShell se saltan si `pwsh` no esta instalado (CI Linux ya lo trae).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from ml.features.store.feature_views import SENSOR_ANALYSIS_SERVICE, SENSOR_FEATURES_VIEW
from tests.integration import benchmark_feature_latency as latency_tool
from tests.integration import benchmark_kafka as kafka_bench
from tests.integration.benchmark_report import build_report, write_report

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "infra" / "scripts" / "Verify-Phase2.ps1"
VALIDATOR_TEST = REPO_ROOT / "tests" / "unit" / "test_validator_on_tep.py"
DASHBOARD = REPO_ROOT / "observability" / "grafana" / "dashboards" / "data-pipeline.json"
DASHBOARDS_DIR = REPO_ROOT / "observability" / "grafana" / "dashboards"
DASHBOARD_KUSTOMIZATION = DASHBOARDS_DIR / "kustomization.yaml"
FEAST_LOCAL_CONFIG = REPO_ROOT / "ml" / "features" / "store" / "feature_store.local.yaml"
INGESTION_DIR = REPO_ROOT / "k8s" / "base" / "ingestion"

PWSH = shutil.which("pwsh")
requires_pwsh = pytest.mark.skipif(PWSH is None, reason="PowerShell 7 no esta instalado")

MET = "CUMPLIDO"
FAILED = "FALLADO"
NOT_MEASURED = "NO MEDIDO"

# Cada script de prueba se ejecuta dot-sourced: el guard del punto de entrada evita que
# el script corra, y las funciones y variables quedan disponibles. `kubectl` lanza para
# que ningun test pueda consultar un cluster por accidente.
_HARNESS = r"""
$ErrorActionPreference = "Stop"
function kubectl { throw "kubectl no debe llamarse desde los tests" }
. "__SCRIPT__" -ProjectRoot "__ROOT__" -PythonPath "__PYTHON__" -SkipCluster __SWITCHES__
__BODY__
"""

_FAKE_DVC_CLEAN = r"""
function dvc { $global:LASTEXITCODE = 0; "{}" }
"""


def _read(path: Path) -> str:
    """Return the UTF-8 text of a repository file."""
    return path.read_text(encoding="utf-8")


def _run(
    tmp_path: Path,
    body: str,
    *,
    root: Path | None = None,
    switches: str = "",
    timeout: int = 180,
) -> dict[str, list[str]]:
    """Run a PowerShell body after dot-sourcing the script; return its KEY=value lines.

    Args:
        tmp_path: Scratch directory for the harness file.
        body: PowerShell code to run once the script's functions are loaded.
        root: Value of -ProjectRoot; defaults to tmp_path.
        switches: Extra switches for the dot-sourced script.
        timeout: Seconds before the run is abandoned.

    Returns:
        Every `KEY=value` line of stdout, grouped by key and in output order.
    """
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        _HARNESS.replace("__SCRIPT__", str(SCRIPT))
        .replace("__ROOT__", str(root or tmp_path))
        .replace("__PYTHON__", sys.executable)
        .replace("__SWITCHES__", switches)
        .replace("__BODY__", body),
        encoding="utf-8",
    )
    # pytest-cov instrumenta los subprocesos de Python a traves de estas variables; aqui
    # solo anadiria ficheros .coverage sueltos por un pytest sintetico.
    env = {k: v for k, v in os.environ.items() if not k.startswith("COV_CORE")}
    result = subprocess.run(  # noqa: S603  (pwsh resuelto con shutil.which, script propio)
        [str(PWSH), "-NoProfile", "-File", str(harness)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    lines: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and re.fullmatch(r"[A-Z_]+", key):
            lines.setdefault(key, []).append(value)
    return lines


def _cases(output: dict[str, list[str]], key: str = "CASE") -> dict[str, tuple[str, str]]:
    """Parse `name|state|extra` lines into {name: (state, extra)}."""
    parsed: dict[str, tuple[str, str]] = {}
    for line in output[key]:
        name, state, extra = line.split("|", 2)
        parsed[name] = (state, extra)
    return parsed


# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


class TestScriptContract:
    """What the script reads is what the rest of the repository produces."""

    @staticmethod
    def _variable(name: str) -> str:
        """Return the first quoted or numeric value assigned to a script variable."""
        pattern = rf'^\s*(?:\[\w+\])?\${name}\s*=\s*"?([^"\s#,]+)"?'
        match = re.search(pattern, _read(SCRIPT), re.MULTILINE)
        assert match, f"Verify-Phase2.ps1 no declara ${name}"
        return match.group(1)

    @staticmethod
    def _array(name: str) -> list[str]:
        """Return the quoted strings of a `$name = @(...)` assignment."""
        match = re.search(rf"^\${name}\s*=\s*@\(([^)]*)\)", _read(SCRIPT), re.MULTILINE)
        assert match, f"Verify-Phase2.ps1 no declara ${name}"
        return re.findall(r'"([^"]+)"', match.group(1))

    def test_follows_the_script_conventions(self) -> None:
        """PowerShell 7, cuatro capas y solo ASCII (sin emojis)."""
        text = _read(SCRIPT)
        assert text.startswith("#Requires -Version 7.0")
        for layer in ("CAPA 1", "CAPA 2", "CAPA 3", "CAPA 4"):
            assert layer in text, f"Falta la {layer}"
        assert text.isascii()

    def test_project_id_is_the_real_one(self) -> None:
        """El contraste general de ProjectId vive en test_k8s_manifests; aqui el valor."""
        assert self._variable("ProjectId") == "sentinel-platform-485714"

    def test_documents_the_four_exit_codes(self) -> None:
        """La cabecera explica 0, 1, 2 y 3, y las variables llevan esos valores."""
        text = _read(SCRIPT)
        for code in ("0  todos", "1  algun", "2  ninguno", "3  error interno"):
            assert code in text
        names = ("ExitMet", "ExitFailed", "ExitNotMeasured")
        assert [int(self._variable(n)) for n in names] == [0, 1, 2]
        assert int(self._variable("ExitError")) == 3

    def test_thresholds_match_the_benchmarks_and_the_gate(self) -> None:
        """Los umbrales del verificador son los que los productores de evidencia declaran."""
        assert float(self._variable("ThroughputThreshold")) == kafka_bench.THROUGHPUT_THRESHOLD
        assert float(self._variable("LatencyThresholdMs")) == latency_tool.LATENCY_THRESHOLD_MS
        gate = re.search(
            r"^STUCK_PRECISION_GATE\s*=\s*([0-9.]+)", _read(VALIDATOR_TEST), re.MULTILINE
        )
        assert gate
        assert float(self._variable("StuckPrecisionThreshold")) == float(gate.group(1))

    def test_report_files_match_what_the_producers_write(self) -> None:
        """Un nombre distinto dejaria el criterio en NO MEDIDO para siempre."""
        assert self._variable("ThroughputReportFile") == kafka_bench.DEFAULT_OUTPUT.name
        assert self._variable("LatencyReportFile") == latency_tool.DEFAULT_OUTPUT.name
        results = re.search(
            r'^RESULTS_PATH\s*=\s*Path\("([^"]+)"\)', _read(VALIDATOR_TEST), re.MULTILINE
        )
        assert results
        assert Path(results.group(1)).name == self._variable("MetricsFile")
        assert kafka_bench.DEFAULT_OUTPUT.parent.as_posix() == self._variable("ResultsDir")

    def test_the_producers_use_the_criterion_names_the_script_expects(self) -> None:
        """Resolve-BenchmarkVerdict rechaza un informe de otro criterio."""
        text = _read(SCRIPT)
        assert f'-Criterion "{kafka_bench.CRITERION}"' in text
        assert f'-Criterion "{latency_tool.CRITERION}"' in text

    def test_stuck_suite_exists_and_writes_the_metrics_it_reads(self) -> None:
        """Los ficheros de la suite existen y el test que escribe las metricas tambien."""
        files = self._array("StuckTestFiles")
        assert len(files) >= 2
        for relative in files:
            assert (REPO_ROOT / relative).is_file(), f"No existe {relative}"
        name = self._variable("MetricsTestName")
        assert f"def {name}(" in _read(VALIDATOR_TEST)

    def test_data_guard_covers_the_partitions_the_suite_skips_on(self) -> None:
        """La comprobacion previa de datos usa los mismos parquets que el skipif."""
        partitions = {
            Path(relative).as_posix()
            for relative in re.findall(
                r'Path\("(data/processed/tep/fault_type=\d+/readings\.parquet)"\)',
                _read(REPO_ROOT / "tests" / "unit" / "test_stream_invariants.py"),
            )
        }
        declared = set(self._array("StuckDataFiles"))
        assert partitions
        assert partitions <= declared
        validator = _read(VALIDATOR_TEST)
        for relative in declared:
            fault = re.search(r"fault_type=\d+", relative)
            assert fault
            assert fault.group(0) in validator

    def test_deliverables_point_at_real_artifacts(self) -> None:
        """Dashboard, Deployments y Feast que el script busca existen con ese nombre."""
        assert (REPO_ROOT / self._variable("DashboardFile")).is_file()
        assert (REPO_ROOT / self._variable("FeastConfigFile")).is_file()
        assert self._variable("FeastView") == SENSOR_FEATURES_VIEW
        assert self._variable("FeastService") == SENSOR_ANALYSIS_SERVICE
        assert re.search(r"^registry:\s*\S+", _read(FEAST_LOCAL_CONFIG), re.MULTILINE)

        deployed: set[str] = set()
        for manifest in INGESTION_DIR.glob("*-deployment.yaml"):
            for document in yaml.safe_load_all(_read(manifest)):
                if document and document.get("kind") == "Deployment":
                    deployed.add(document["metadata"]["name"])
                    assert document["metadata"]["namespace"] == self._variable("Namespace")
        assert set(self._array("ExpectedDeployments")) <= deployed

        kustomization = yaml.safe_load(_read(DASHBOARD_KUSTOMIZATION))
        assert kustomization["namespace"] == self._variable("DashboardNamespace")
        label = kustomization["generatorOptions"]["labels"]
        key, value = next(iter(label.items()))
        assert f"{key}={value}" == self._variable("DashboardLabel")


# ---------------------------------------------------------------------------
# Criteria 1 and 3: benchmark reports
# ---------------------------------------------------------------------------


def _benchmark_report(
    tmp_path: Path,
    name: str,
    *,
    criterion: str,
    environment: str,
    value: float,
    threshold: float,
    direction: str,
    unit: str,
    bootstrap: str | None = None,
    passed: bool | None = None,
    commit: str = "abc1234",
) -> Path:
    """Write a report in the exact format of build_report and return its path."""
    document = build_report(
        criterion=criterion,
        environment=environment,
        value=value,
        unit=unit,
        threshold=threshold,
        direction=direction,
        details={"bootstrap_servers": [bootstrap]} if bootstrap else {},
        commit=commit,
        measured_at="2026-10-11T00:00:00Z",
    )
    if passed is not None:
        document["passed"] = passed
    path = tmp_path / f"{name}.json"
    write_report(path, document)
    return path


@requires_pwsh
class TestBenchmarkCriteria:
    """Criteria 1 and 3 only close with a measurement taken in the cluster."""

    _THROUGHPUT = {
        "criterion": "kafka_throughput",
        "threshold": 50_000.0,
        "direction": "at_least",
        "unit": "messages_per_second",
    }
    _LATENCY = {
        "criterion": "feature_latency",
        "threshold": 100.0,
        "direction": "at_most",
        "unit": "milliseconds",
    }

    def test_each_rule_of_the_state_table(self, tmp_path: Path) -> None:
        """Entorno, umbral, informes rotos y un informe que se aprueba a si mismo."""
        t, lat = self._THROUGHPUT, self._LATENCY
        reports: dict[str, tuple[Path | None, str, str]] = {
            # name: (path, kind, expected state)
            "cluster_met": (
                _benchmark_report(tmp_path, "a", environment="cluster", value=62_000.0, **t),
                "T",
                MET,
            ),
            "cluster_failed": (
                _benchmark_report(tmp_path, "b", environment="cluster", value=30_000.0, **t),
                "T",
                FAILED,
            ),
            "cluster_exactly_at_threshold": (
                _benchmark_report(tmp_path, "c", environment="cluster", value=50_000.0, **t),
                "T",
                MET,
            ),
            # Una cifra local que SUPERA el umbral sigue sin cerrar el criterio.
            "local_above_threshold": (
                _benchmark_report(tmp_path, "d", environment="local", value=99_999.0, **t),
                "T",
                NOT_MEASURED,
            ),
            "local_below_threshold": (
                _benchmark_report(tmp_path, "e", environment="local", value=13_993.0, **t),
                "T",
                NOT_MEASURED,
            ),
            # Declarado cluster pero con el bootstrap del equipo: no es evidencia de cluster.
            "cluster_with_loopback_bootstrap": (
                _benchmark_report(
                    tmp_path,
                    "f",
                    environment="cluster",
                    value=62_000.0,
                    bootstrap="127.0.0.1:9092",
                    **t,
                ),
                "T",
                NOT_MEASURED,
            ),
            "cluster_with_real_bootstrap": (
                _benchmark_report(
                    tmp_path,
                    "g",
                    environment="cluster",
                    value=62_000.0,
                    bootstrap="reactorguard-cluster-kafka-bootstrap.kafka-operator:9093",
                    **t,
                ),
                "T",
                MET,
            ),
            "latency_cluster_met": (
                _benchmark_report(tmp_path, "h", environment="cluster", value=80.0, **lat),
                "L",
                MET,
            ),
            "latency_cluster_failed": (
                _benchmark_report(tmp_path, "i", environment="cluster", value=120.0, **lat),
                "L",
                FAILED,
            ),
            "latency_local_met": (
                _benchmark_report(tmp_path, "j", environment="local", value=40.0, **lat),
                "L",
                NOT_MEASURED,
            ),
            # Un informe de latencia no puede cerrar el criterio de throughput.
            "wrong_criterion": (
                _benchmark_report(tmp_path, "k", environment="cluster", value=80.0, **lat),
                "T",
                NOT_MEASURED,
            ),
            "absent": (tmp_path / "absent.json", "T", NOT_MEASURED),
        }
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not json", encoding="utf-8")
        reports["corrupt"] = (corrupt, "T", NOT_MEASURED)
        unknown_env = tmp_path / "unknown_env.json"
        document = json.loads(_read(tmp_path / "a.json"))
        document["environment"] = "staging"
        unknown_env.write_text(json.dumps(document), encoding="utf-8")
        reports["unknown_environment"] = (unknown_env, "T", NOT_MEASURED)

        body_lines = []
        for name, (path, kind, _) in reports.items():
            kind_args = (
                '-Criterion "kafka_throughput" -Threshold $ThroughputThreshold -Direction at_least '
                "-Unit messages_per_second"
                if kind == "T"
                else '-Criterion "feature_latency" -Threshold $LatencyThresholdMs '
                "-Direction at_most -Unit milliseconds"
            )
            body_lines.append(
                f'$v = Resolve-BenchmarkVerdict -Id "x" -Name "n" {kind_args} -HowToMeasure "how" '
                f'-Read (Read-JsonFile -Path "{path}") -HeadCommit "abc1234"\n'
                f'Write-Output ("CASE={name}|" + $v.state + "|" + ($v.warnings -join ";"))'
            )
        output = _cases(_run(tmp_path, "\n".join(body_lines)))

        for name, (_, _, expected) in reports.items():
            assert output[name][0] == expected, f"{name}: {output[name]}"

    def test_the_script_threshold_wins_over_the_one_in_the_report(self, tmp_path: Path) -> None:
        """Un informe con el umbral rebajado no puede aprobarse a si mismo."""
        lowered = _benchmark_report(
            tmp_path,
            "lowered",
            criterion="kafka_throughput",
            environment="cluster",
            value=30_000.0,
            threshold=10_000.0,
            direction="at_least",
            unit="messages_per_second",
        )
        lying = _benchmark_report(
            tmp_path,
            "lying",
            criterion="kafka_throughput",
            environment="cluster",
            value=30_000.0,
            threshold=50_000.0,
            direction="at_least",
            unit="messages_per_second",
            passed=True,
        )
        body = "\n".join(
            f'$v = Resolve-BenchmarkVerdict -Id "x" -Name "n" -Criterion "kafka_throughput" '
            f'-Threshold $ThroughputThreshold -Direction at_least -Unit messages_per_second '
            f'-HowToMeasure "how" -Read (Read-JsonFile -Path "{path}")\n'
            f'Write-Output ("CASE={name}|" + $v.state + "|" + ($v.warnings -join ";"))'
            for name, path in (("lowered", lowered), ("lying", lying))
        )
        output = _cases(_run(tmp_path, body))
        assert output["lowered"][0] == FAILED
        assert "umbral 10000" in output["lowered"][1]
        assert output["lying"][0] == FAILED
        assert "passed=True" in output["lying"][1]

    def test_warns_when_the_measurement_comes_from_another_commit(self, tmp_path: Path) -> None:
        """La evidencia de otro commit o de un arbol sucio se senala, sin cambiar el estado."""
        stale = _benchmark_report(
            tmp_path,
            "stale",
            environment="cluster",
            value=62_000.0,
            commit="deadbee-dirty",
            **self._THROUGHPUT,
        )
        body = (
            f'$v = Resolve-BenchmarkVerdict -Id "x" -Name "n" -Criterion "kafka_throughput" '
            f'-Threshold $ThroughputThreshold -Direction at_least -Unit messages_per_second '
            f'-HowToMeasure "how" -Read (Read-JsonFile -Path "{stale}") -HeadCommit "abc1234"\n'
            'Write-Output ("CASE=stale|" + $v.state + "|" + ($v.warnings -join ";"))'
        )
        state, warnings = _cases(_run(tmp_path, body))["stale"]
        assert state == MET
        assert "deadbee-dirty" in warnings
        assert "sin commitear" in warnings


# ---------------------------------------------------------------------------
# Criterion 2: passed is not skipped
# ---------------------------------------------------------------------------

_JUNIT_START = datetime(2026, 10, 11, 10, 0, 0, tzinfo=UTC)
_VALIDATOR_CLASS = "tests.unit.test_validator_on_tep"
_TEST_METRICS = f"{_VALIDATOR_CLASS}.test_metrics_report_is_written"


def _junit(path: Path, cases: list[tuple[str, str]], *, start: datetime = _JUNIT_START) -> Path:
    """Write a JUnit report with the shape pytest produces.

    Args:
        path: Destination file.
        cases: (classname.name, outcome) pairs; outcome is passed, skipped, failure or error.
        start: Suite start, written with a UTC offset like pytest does.

    Returns:
        The path written.
    """
    rows = []
    for full_name, outcome in cases:
        classname, _, name = full_name.rpartition(".")
        child = {
            "passed": "",
            "skipped": '<skipped type="pytest.skip" message="faltan los datos">x</skipped>',
            "failure": '<failure message="assert 0.5 >= 0.95">trace</failure>',
            "error": '<error message="collection">trace</error>',
        }[outcome]
        rows.append(
            f'<testcase classname="{classname}" name="{name}" time="0.01">{child}</testcase>'
        )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="{len(cases)}" '
        f'time="1.0" timestamp="{start.isoformat()}" hostname="h">{"".join(rows)}'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return path


def _metrics(path: Path, measured: float, *, written: datetime | None = None) -> Path:
    """Write a validator metrics report; optionally back-date its modification time."""
    path.write_text(
        json.dumps(
            {
                "gate": {"measured": measured, "threshold": 0.95, "passed": measured >= 0.95},
                "cross_validation": {"stuck_pairs_detected": 473},
            }
        ),
        encoding="utf-8",
    )
    if written is not None:
        stamp = written.timestamp()
        os.utime(path, (stamp, stamp))
    return path


@requires_pwsh
class TestStuckCriterion:
    """El exit code de pytest no basta: un skipif devuelve 0 y no mide nada."""

    @staticmethod
    def _passed(count: int = 14) -> list[tuple[str, str]]:
        """Return `count` passing tests plus the one that writes the metrics."""
        base = [(f"tests.unit.test_validator_on_tep.test_{n}", "passed") for n in range(count)]
        return [*base, (_TEST_METRICS, "passed")]

    def test_each_rule_of_the_state_table(self, tmp_path: Path) -> None:
        """Cada combinacion de datos, informe JUnit y metricas, con su estado."""
        fresh = _JUNIT_START.replace(second=30)
        old = _JUNIT_START.replace(hour=8)
        scenarios: dict[str, dict[str, Any]] = {
            "all_passed_gate_met": {
                "junit": self._passed(),
                "metrics": _metrics(tmp_path / "m_ok.json", 1.0, written=fresh),
                "expected": MET,
            },
            "precision_below_gate": {
                "junit": self._passed(),
                "metrics": _metrics(tmp_path / "m_low.json", 0.90, written=fresh),
                "expected": FAILED,
            },
            # El caso de la trampa: 15 skipped, exit code 0, JSON viejo con 1,0.
            "everything_skipped": {
                "junit": [(f"{_VALIDATOR_CLASS}.test_{n}", "skipped") for n in range(15)],
                "metrics": _metrics(tmp_path / "m_old1.json", 1.0, written=old),
                "expected": NOT_MEASURED,
            },
            "one_skipped_among_passed": {
                "junit": [*self._passed(), ("tests.unit.test_stream_invariants.test_x", "skipped")],
                "metrics": _metrics(tmp_path / "m_ok2.json", 1.0, written=fresh),
                "expected": NOT_MEASURED,
            },
            "a_test_fails": {
                "junit": [*self._passed(), ("tests.unit.test_stream_invariants.test_y", "failure")],
                "metrics": _metrics(tmp_path / "m_ok3.json", 1.0, written=fresh),
                "expected": FAILED,
            },
            "a_test_errors": {
                "junit": [*self._passed(), ("tests.unit.test_stream_invariants.test_z", "error")],
                "metrics": _metrics(tmp_path / "m_ok4.json", 1.0, written=fresh),
                "expected": FAILED,
            },
            "metrics_test_not_run": {
                "junit": [(f"{_VALIDATOR_CLASS}.test_{n}", "passed") for n in range(5)],
                "metrics": _metrics(tmp_path / "m_ok5.json", 1.0, written=fresh),
                "expected": NOT_MEASURED,
            },
            # Tests verdes pero el JSON es de una ejecucion anterior: cifra vieja.
            "stale_metrics": {
                "junit": self._passed(),
                "metrics": _metrics(tmp_path / "m_stale.json", 1.0, written=old),
                "expected": NOT_MEASURED,
            },
            "metrics_missing": {
                "junit": self._passed(),
                "metrics": tmp_path / "m_absent.json",
                "expected": NOT_MEASURED,
            },
            "no_junit": {
                "junit": None,
                "metrics": _metrics(tmp_path / "m_ok6.json", 1.0, written=fresh),
                "expected": NOT_MEASURED,
            },
        }

        body = []
        for name, scenario in scenarios.items():
            junit_path = tmp_path / f"{name}.xml"
            if scenario["junit"] is not None:
                _junit(junit_path, scenario["junit"])
            body.append(
                f'$v = Resolve-StuckPrecisionVerdict -DataPresent $true '
                f'-JUnit (Read-JUnitSummary -Path "{junit_path}") '
                f'-Metrics (Read-JsonFile -Path "{scenario["metrics"]}")\n'
                f'Write-Output ("CASE={name}|" + $v.state + "|" + $v.detail)'
            )
        body.append(
            '$v = Resolve-StuckPrecisionVerdict -DataPresent $false -JUnit $null '
            '-Metrics (Read-JsonFile -Path "absent")\n'
            'Write-Output ("CASE=no_data|" + $v.state + "|" + $v.detail)'
        )
        output = _cases(_run(tmp_path, "\n".join(body)))

        for name, scenario in scenarios.items():
            assert output[name][0] == scenario["expected"], f"{name}: {output[name]}"
        assert output["no_data"][0] == NOT_MEASURED
        assert "Faltan los datos" in output["no_data"][1]
        assert "SALTADOS" in output["everything_skipped"][1]
        assert "anterior a esta ejecucion" in output["stale_metrics"][1]

    def test_a_real_pytest_run_is_classified_by_what_it_executed(self, tmp_path: Path) -> None:
        """Invoke-StuckSuite sobre una suite sintetica: el JUnit REAL de pytest se interpreta."""
        passing = tmp_path / "ok" / "tests" / "unit"
        skipping = tmp_path / "skip" / "tests" / "unit"
        passing.mkdir(parents=True)
        skipping.mkdir(parents=True)
        writer = (
            "import json\nfrom pathlib import Path\n\n"
            "def test_something():\n    assert True\n\n"
            "def test_metrics_report_is_written():\n"
            '    target = Path("tests/results/validator_metrics_tep.json")\n'
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            '    target.write_text(json.dumps({"gate": {"measured": 1.0},'
            ' "cross_validation": {"stuck_pairs_detected": 473}}))\n'
        )
        (passing / "test_fake_stuck.py").write_text(writer, encoding="utf-8")
        (skipping / "test_fake_stuck.py").write_text(
            "import pytest\n\npytestmark = pytest.mark.skipif(True, reason='faltan los datos')\n\n"
            "def test_something():\n    assert True\n\n"
            "def test_metrics_report_is_written():\n    assert True\n",
            encoding="utf-8",
        )

        body = r"""
$StuckTestFiles = @("tests/unit/test_fake_stuck.py")
foreach ($name in @("ok", "skip")) {
    $root = Join-Path "__BASE__" $name
    $junit = Join-Path $root "tests/results/junit.xml"
    $run = Invoke-StuckSuite -Root $root -Python "__PYTHON__" -JUnitPath $junit
    $summary = Read-JUnitSummary -Path $junit
    $metrics = Read-JsonFile -Path (Join-Path $root "tests/results/validator_metrics_tep.json")
    $v = Resolve-StuckPrecisionVerdict -DataPresent $true -JUnit $summary -Metrics $metrics
    $facts = "exit=" + $run.ExitCode + " passed=" + $summary.Passed + " skipped=" + $summary.Skipped
    $dated = $null -ne $summary.StartUtc
    Write-Output ("CASE=$name|" + $v.state + "|" + $facts + " start=" + $dated)
}
""".replace("__BASE__", str(tmp_path)).replace("__PYTHON__", sys.executable)
        output = _cases(_run(tmp_path, body))

        assert output["ok"][0] == MET
        assert "passed=2 skipped=0 start=True" in output["ok"][1]
        # pytest sale con codigo 0 aunque todo se haya saltado: el estado lo da el JUnit.
        assert output["skip"][0] == NOT_MEASURED
        assert "exit=0" in output["skip"][1]
        assert "passed=0 skipped=2" in output["skip"][1]


# ---------------------------------------------------------------------------
# Criterion 4: dvc
# ---------------------------------------------------------------------------

_DVC_YAML = "schema: '2.0'\nstages:\n  one:\n    cmd: x\n  two:\n    cmd: y\n"


@requires_pwsh
class TestDvcCriterion:
    """dvc status limpio y dvc.lock completo; lo desactualizado no es lo roto."""

    def test_stage_names_and_hashes_agree_with_the_real_files(self, tmp_path: Path) -> None:
        """El analizador de dvc.yaml y dvc.lock devuelve lo mismo que un parser de YAML."""
        body = r"""
$yaml = Get-Content -LiteralPath "__YAML__" -Raw
$lock = Get-Content -LiteralPath "__LOCK__" -Raw
Write-Output ("STAGES=" + ((Get-StageNames -Text $yaml) -join ","))
Write-Output ("LOCKED=" + ((Get-StageNames -Text $lock) -join ","))
$missing = @(Get-MissingLockStages -DvcYamlText $yaml -DvcLockText $lock)
Write-Output ("MISSING=" + ($missing -join ","))
$hashes = Get-LockOutputHashes -Text $lock
foreach ($key in $hashes.Keys) { Write-Output ("HASH=" + $key + "=" + $hashes[$key]) }
""".replace("__YAML__", str(REPO_ROOT / "dvc.yaml")).replace(
            "__LOCK__", str(REPO_ROOT / "dvc.lock")
        )
        output = _run(tmp_path, body)

        declared = list(yaml.safe_load(_read(REPO_ROOT / "dvc.yaml"))["stages"])
        lock = yaml.safe_load(_read(REPO_ROOT / "dvc.lock"))["stages"]
        assert output["STAGES"][0].split(",") == declared
        assert output["LOCKED"][0].split(",") == list(lock)
        # El dvc.lock del repositorio debe registrar todas las etapas de dvc.yaml.
        assert output["MISSING"] == [""]
        expected = {
            f"{stage}:{out['path']}": out["md5"]
            for stage, body_ in lock.items()
            for out in body_.get("outs", [])
        }
        parsed = dict(entry.rsplit("=", 1) for entry in output["HASH"])
        assert parsed == expected
        assert len(expected) >= 5

    def test_a_missing_stage_in_the_lock_is_reported(self, tmp_path: Path) -> None:
        """Una etapa nueva sin `dvc repro` registrada no puede pasar por reproducible."""
        lock = "schema: '2.0'\nstages:\n  one:\n    cmd: x\n"
        body = (
            f'$missing = Get-MissingLockStages -DvcYamlText @"\n{_DVC_YAML}"@ '
            f'-DvcLockText @"\n{lock}"@\n'
            'Write-Output ("MISSING=" + ($missing -join ","))\n'
            '$none = Get-MissingLockStages -DvcYamlText @"\n' + _DVC_YAML + '"@ -DvcLockText ""\n'
            'Write-Output ("EMPTYLOCK=" + ($none -join ","))'
        )
        output = _run(tmp_path, body)
        assert output["MISSING"] == ["two"]
        assert output["EMPTYLOCK"] == ["one,two"]

    def test_hash_comparison_names_the_differing_outputs(self, tmp_path: Path) -> None:
        """Compare-LockHashes distingue coincide, difiere y no se produjo."""
        body = r"""
$expected = @{ "a:out1" = "111"; "a:out2" = "222"; "b:out3" = "333" }
$actual = @{ "a:out1" = "111"; "a:out2" = "999" }
$diff = Compare-LockHashes -Expected $expected -Actual $actual
foreach ($line in $diff) { Write-Output ("DIFF=" + $line) }
Write-Output ("SAME=" + @(Compare-LockHashes -Expected $expected -Actual $expected).Count)
"""
        output = _run(tmp_path, body)
        assert output["DIFF"] == [
            "a:out2 (esperado 222, obtenido 999)",
            "b:out3 (esperado 333, no se produjo)",
        ]
        assert output["SAME"] == ["0"]

    def test_a_network_failure_is_told_apart_from_a_pipeline_failure(self, tmp_path: Path) -> None:
        """Sin red la reproduccion queda sin medir; cualquier otro fallo es del pipeline."""
        body = r"""
$net = "ERROR: failed to reproduce 'download_tep': Max retries exceeded with url: /file"
$pipe = "ERROR: failed to reproduce 'download_tep': [WinError 3] ruta no encontrada"
$bar = "d00.dat: 93%|#####| 48.0k/51.7k [00:00<00:00, 449kB/s]"
$noise = $bar + "`nINFO Download complete`nERROR: boom"
Write-Output ("NET=" + (Test-NetworkFailure -Text $net))
Write-Output ("PIPE=" + (Test-NetworkFailure -Text $pipe))
Write-Output ("TAIL=" + (Get-OutputTail -Text $noise))
"""
        output = _run(tmp_path, body)
        assert output["NET"] == ["True"]
        assert output["PIPE"] == ["False"]
        assert output["TAIL"] == ["INFO Download complete | ERROR: boom"]

    def test_each_rule_of_the_state_table(self, tmp_path: Path) -> None:
        """dvc status limpio, dependencias cambiadas, salidas ausentes o modificadas."""
        body = r"""
function Status($json) { Get-DvcStatusSummary -Status ($json | ConvertFrom-Json -AsHashtable) }
function Case($name, $v) { Write-Output ("CASE=$name|" + $v.state + "|" + $v.detail) }
function Scratch($state, $detail) { [pscustomobject]@{ Status = $state; Detail = $detail } }
$clean = Status '{}'
$deps = Status '{"featurize": [{"changed deps": {"pipeline.py": "modified"}}]}'
$deleted = Status '{"adapt_tep": [{"changed outs": {"data/processed/tep": "deleted"}}]}'
$modified = Status '{"adapt_tep": [{"changed outs": {"data/processed/tep": "modified"}}]}'
$always = Status '{"s": ["always changed"]}'

Case "clean" (Resolve-DvcVerdict -StatusSummary $clean)
Case "changed_deps" (Resolve-DvcVerdict -StatusSummary $deps)
Case "out_deleted" (Resolve-DvcVerdict -StatusSummary $deleted)
Case "out_modified" (Resolve-DvcVerdict -StatusSummary $modified)
Case "always_changed" (Resolve-DvcVerdict -StatusSummary $always)
Case "lock_incomplete" (Resolve-DvcVerdict -MissingLockStages @("feast_source") `
    -StatusSummary $clean)
Case "status_unavailable" (Resolve-DvcVerdict -StatusSummary $null -StatusProblem "sin dvc")
$matched = Scratch "matched" "9"
$mismatch = Scratch "mismatch" "a:b"
$failed = Scratch "failed" "boom"
$offline = Scratch "unavailable" "red"
Case "scratch_matched" (Resolve-DvcVerdict -StatusSummary $clean -Scratch $matched)
Case "scratch_mismatch" (Resolve-DvcVerdict -StatusSummary $clean -Scratch $mismatch)
Case "scratch_failed" (Resolve-DvcVerdict -StatusSummary $clean -Scratch $failed)
Case "scratch_no_network" (Resolve-DvcVerdict -StatusSummary $clean -Scratch $offline)
"""
        output = _cases(_run(tmp_path, body))
        expected = {
            "clean": MET,
            "changed_deps": NOT_MEASURED,
            "out_deleted": NOT_MEASURED,
            "out_modified": FAILED,
            "always_changed": NOT_MEASURED,
            "lock_incomplete": FAILED,
            "status_unavailable": NOT_MEASURED,
            "scratch_matched": MET,
            "scratch_mismatch": FAILED,
            "scratch_failed": FAILED,
            # Si se pidio la prueba fuerte y no se pudo completar, no se da por cumplido.
            "scratch_no_network": NOT_MEASURED,
        }
        for name, state in expected.items():
            assert output[name][0] == state, f"{name}: {output[name]}"
        assert "feast_source" in output["lock_incomplete"][1]

    def test_get_dvc_status_reads_json_through_the_dvc_command(self, tmp_path: Path) -> None:
        """Get-DvcStatus usa `dvc status --json` y distingue fallo de comando y JSON invalido."""
        body = r"""
function dvc {
    $global:LASTEXITCODE = 0
    if ($env:FAKE_DVC -eq "dirty") { '{"split": [{"changed deps": {"params.yaml": "modified"}}]}' }
    elseif ($env:FAKE_DVC -eq "garbage") { 'not json' }
    elseif ($env:FAKE_DVC -eq "error") { $global:LASTEXITCODE = 255; 'ERROR: unexpected' }
    else { '{}' }
}
foreach ($mode in @("clean", "dirty", "garbage", "error")) {
    $env:FAKE_DVC = $mode
    $r = Get-DvcStatus -Root "__ROOT__"
    $deps = if ($null -eq $r.Summary) { "none" } else { @($r.Summary.ChangedDeps).Count }
    Write-Output ("CASE=$mode|" + ($null -ne $r.Summary) + "|deps=$deps problem=" + $r.Problem)
}
""".replace("__ROOT__", str(tmp_path))
        output = _cases(_run(tmp_path, body))
        assert output["clean"] == ("True", "deps=0 problem=")
        assert output["dirty"][0] == "True"
        assert "deps=1" in output["dirty"][1]
        assert output["garbage"][0] == "False"
        assert "no devolvio JSON" in output["garbage"][1]
        assert output["error"][0] == "False"
        assert "255" in output["error"][1]


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------


@requires_pwsh
class TestDeliverables:
    """Pods, dashboard y Feast: los mismos tres estados, sin bloquear."""

    @staticmethod
    def _deployments(**ready: tuple[int, int]) -> str:
        """Build the JSON of `kubectl get deployments` with (desired, ready) replicas."""
        items = [
            {
                "metadata": {"name": name},
                "spec": {"replicas": desired},
                "status": {"readyReplicas": have} if have else {},
            }
            for name, (desired, have) in ready.items()
        ]
        return json.dumps({"items": items})

    def test_deployments(self, tmp_path: Path) -> None:
        """Listos, sin replicas listas, ausentes y sin cluster."""
        scenarios = {
            "all_ready": self._deployments(**{"tep-streamer": (1, 1), "sensor-validator": (2, 2)}),
            "validator_half_ready": self._deployments(
                **{"tep-streamer": (1, 1), "sensor-validator": (2, 1)}
            ),
            "validator_crashing": self._deployments(
                **{"tep-streamer": (1, 1), "sensor-validator": (2, 0)}
            ),
            "validator_absent": self._deployments(**{"tep-streamer": (1, 1)}),
            "nothing_deployed": self._deployments(),
            "scaled_to_zero": self._deployments(
                **{"tep-streamer": (0, 0), "sensor-validator": (0, 0)}
            ),
        }
        body = []
        for name, document in scenarios.items():
            path = tmp_path / f"{name}.json"
            path.write_text(document, encoding="utf-8")
            body.append(
                f'$d = Get-Content -LiteralPath "{path}" -Raw | ConvertFrom-Json\n'
                f'$v = Resolve-DeploymentsVerdict -Deployments $d\n'
                f'Write-Output ("CASE={name}|" + $v.state + "|" + $v.detail)'
            )
        body.append(
            '$v = Resolve-DeploymentsVerdict -Deployments $null -Problem "sin cluster"\n'
            'Write-Output ("CASE=no_cluster|" + $v.state + "|" + $v.detail + "|" + $v.blocking)'
        )
        output = _cases(_run(tmp_path, "\n".join(body)))
        expected = {
            "all_ready": MET,
            "validator_half_ready": FAILED,
            "validator_crashing": FAILED,
            # Faltando uno de los dos hay un despliegue a medias, que es un fallo.
            "validator_absent": FAILED,
            "nothing_deployed": NOT_MEASURED,
            "scaled_to_zero": FAILED,
            "no_cluster": NOT_MEASURED,
        }
        for name, state in expected.items():
            assert output[name][0] == state, f"{name}: {output[name]}"
        assert output["no_cluster"][1].endswith("|False"), "los entregables no bloquean"

    def test_dashboard(self, tmp_path: Path) -> None:
        """Valido sin desplegar, desplegado y roto."""
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({"uid": "", "panels": []}), encoding="utf-8")
        body = rf"""
$file = Read-JsonFile -Path "{DASHBOARD}"
$none = Resolve-DashboardVerdict -File $file -ClusterConfigMaps $null -ClusterProblem "sin cluster"
Write-Output ("CASE=valid_not_deployed|" + $none.state + "|" + $none.detail)
$empty = [pscustomobject]@{{ items = @() }}
$v = Resolve-DashboardVerdict -File $file -ClusterConfigMaps $empty
Write-Output ("CASE=no_configmap|" + $v.state + "|" + $v.detail)
$entry = [pscustomobject]@{{ metadata = @{{ name = "dashboard-data-pipeline" }} }}
$one = [pscustomobject]@{{ items = @($entry) }}
$v = Resolve-DashboardVerdict -File $file -ClusterConfigMaps $one
Write-Output ("CASE=deployed|" + $v.state + "|" + $v.detail)
$v = Resolve-DashboardVerdict -File (Read-JsonFile -Path "{broken}")
Write-Output ("CASE=broken|" + $v.state + "|" + $v.detail)
$v = Resolve-DashboardVerdict -File (Read-JsonFile -Path "{tmp_path / 'absent.json'}")
Write-Output ("CASE=absent|" + $v.state + "|" + $v.detail)
"""
        output = _cases(_run(tmp_path, body))
        assert output["valid_not_deployed"][0] == NOT_MEASURED
        assert "18 paneles" in output["valid_not_deployed"][1]
        assert output["no_configmap"][0] == NOT_MEASURED
        assert "no hay Grafana en el cluster" in output["no_configmap"][1]
        assert output["deployed"][0] == MET
        assert output["broken"][0] == FAILED
        assert output["absent"][0] == FAILED

    def test_feast_registry(self, tmp_path: Path) -> None:
        """Sin registro, registro correcto y registro sin la vista esperada."""
        body = r"""
$v = Resolve-FeastVerdict -RegistryExists $false
Write-Output ("CASE=no_registry|" + $v.state + "|" + $v.detail)
$good = [pscustomobject]@{
    feature_views = @("sensor_features"); feature_services = @("sensor_analysis")
}
$v = Resolve-FeastVerdict -RegistryExists $true -Probe $good
Write-Output ("CASE=applied|" + $v.state + "|" + $v.detail)
$bad = [pscustomobject]@{ feature_views = @("other"); feature_services = @() }
$v = Resolve-FeastVerdict -RegistryExists $true -Probe $bad
Write-Output ("CASE=wrong_views|" + $v.state + "|" + $v.detail)
$v = Resolve-FeastVerdict -RegistryExists $true -Probe $null -Problem "no se pudo leer"
Write-Output ("CASE=unreadable|" + $v.state + "|" + $v.detail)
"""
        output = _cases(_run(tmp_path, body))
        assert output["no_registry"][0] == NOT_MEASURED
        assert output["applied"][0] == MET
        assert output["wrong_views"][0] == FAILED
        assert output["unreadable"][0] == NOT_MEASURED

    def test_cluster_is_only_queried_with_a_context_of_this_project(self, tmp_path: Path) -> None:
        """Un contexto ajeno (docker-desktop) no es evidencia; -Skip ni llama a kubectl."""
        body = r"""
$script:calls = 0
function kubectl {
    $script:calls++
    $global:LASTEXITCODE = 0
    $env:FAKE_CONTEXT
}
$env:FAKE_CONTEXT = "docker-desktop"
$other = Get-ClusterAccess -Project "sentinel-platform-485714"
Write-Output ("CASE=foreign|" + $other.Available + "|" + $other.Reason)
$env:FAKE_CONTEXT = "gke_sentinel-platform-485714_europe-southwest1_reactorguard-cluster"
$own = Get-ClusterAccess -Project "sentinel-platform-485714"
Write-Output ("CASE=own|" + $own.Available + "|" + $own.Reason)
$before = $script:calls
$skipped = Get-ClusterAccess -Project "sentinel-platform-485714" -Skip
Write-Output ("CASE=skipped|" + $skipped.Available + "|" + $skipped.Reason)
Write-Output ("EXTRACALLS=" + ($script:calls - $before))
"""
        output = _run(tmp_path, body)
        cases = _cases(output)
        assert cases["foreign"][0] == "False"
        assert "docker-desktop" in cases["foreign"][1]
        assert cases["own"][0] == "True"
        assert cases["skipped"][0] == "False"
        assert output["EXTRACALLS"] == ["0"]


# ---------------------------------------------------------------------------
# Exit codes and the whole script
# ---------------------------------------------------------------------------


def _project(
    tmp_path: Path,
    *,
    throughput_env: str,
    throughput_value: float,
    latency_env: str,
    latency_value: float,
    suite_skips: bool = False,
) -> Path:
    """Create a synthetic repository with every artifact Verify-Phase2.ps1 reads."""
    root = tmp_path / "project"
    (root / "tests" / "results").mkdir(parents=True)
    (root / "tests" / "unit").mkdir(parents=True)
    (root / "data" / "processed" / "tep" / "fault_type=00").mkdir(parents=True)
    (root / "data" / "processed" / "tep" / "fault_type=21").mkdir(parents=True)
    for partition in ("00", "21"):
        folder = root / "data" / "processed" / "tep" / f"fault_type={partition}"
        (folder / "readings.parquet").write_bytes(b"")
    (root / "dvc.yaml").write_text(_DVC_YAML, encoding="utf-8")
    (root / "dvc.lock").write_text(
        "schema: '2.0'\nstages:\n  one:\n    cmd: x\n  two:\n    cmd: y\n", encoding="utf-8"
    )
    dashboard = root / "observability" / "grafana" / "dashboards"
    dashboard.mkdir(parents=True)
    shutil.copy(DASHBOARD, dashboard / DASHBOARD.name)

    skip_line = (
        "pytestmark = pytest.mark.skipif(True, reason='faltan los datos')\n" if suite_skips else ""
    )
    (root / "tests" / "unit" / "test_fake_stuck.py").write_text(
        "import json\nfrom pathlib import Path\n\nimport pytest\n\n"
        f"{skip_line}\n"
        "def test_metrics_report_is_written():\n"
        '    target = Path("tests/results/validator_metrics_tep.json")\n'
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        '    target.write_text(json.dumps({"gate": {"measured": 1.0},'
        ' "cross_validation": {"stuck_pairs_detected": 473}}))\n',
        encoding="utf-8",
    )
    bootstrap = "reactorguard-cluster-kafka-bootstrap.kafka-operator:9093"
    write_report(
        root / "tests" / "results" / "kafka_throughput.json",
        build_report(
            criterion="kafka_throughput",
            environment=throughput_env,
            value=throughput_value,
            unit="messages_per_second",
            threshold=50_000.0,
            direction="at_least",
            details={"bootstrap_servers": [bootstrap]},
            commit="abc1234",
        ),
    )
    write_report(
        root / "tests" / "results" / "feature_latency.json",
        build_report(
            criterion="feature_latency",
            environment=latency_env,
            value=latency_value,
            unit="milliseconds",
            threshold=100.0,
            direction="at_most",
            details={},
            commit="abc1234",
        ),
    )
    return root


_RUN_ALL = (
    '$StuckTestFiles = @("tests/unit/test_fake_stuck.py")\n'
    + _FAKE_DVC_CLEAN
    + '$code = Invoke-VerifyPhase2\nWrite-Output ("EXIT=" + $code)'
)


@requires_pwsh
class TestExitCodes:
    """0 todos cumplidos, 1 algun fallado, 2 ninguno fallado pero alguno sin medir."""

    @staticmethod
    def _report(root: Path) -> dict[str, Any]:
        """Return the verification report the script wrote."""
        return json.loads(_read(root / "tests" / "results" / "phase2_verification.json"))

    @staticmethod
    def _states(report: dict[str, Any]) -> dict[str, str]:
        """Return {criterion id: state} from a verification report."""
        return {item["id"]: item["state"] for item in report["criteria"]}

    def test_exit_code_function(self, tmp_path: Path) -> None:
        """Get-ExitCode: FALLADO gana a NO MEDIDO y los entregables no cuentan."""
        body = r"""
function V($state, $blocking) {
    New-Verdict -Id "i" -Name "n" -Blocking $blocking -State $state -Detail "d"
}
Write-Output ("EXIT=" + (Get-ExitCode -Verdicts @((V "CUMPLIDO" $true), (V "CUMPLIDO" $true))))
Write-Output ("EXIT=" + (Get-ExitCode -Verdicts @((V "CUMPLIDO" $true), (V "NO MEDIDO" $true))))
Write-Output ("EXIT=" + (Get-ExitCode -Verdicts @((V "FALLADO" $true), (V "NO MEDIDO" $true))))
Write-Output ("EXIT=" + (Get-ExitCode -Verdicts @((V "CUMPLIDO" $true), (V "FALLADO" $false))))
"""
        assert _run(tmp_path, body)["EXIT"] == ["0", "2", "1", "0"]

    def test_everything_met_exits_zero(self, tmp_path: Path) -> None:
        """Con evidencia de cluster y la suite pasando, los cuatro criterios cierran."""
        root = _project(
            tmp_path,
            throughput_env="cluster",
            throughput_value=62_000.0,
            latency_env="cluster",
            latency_value=72.0,
        )
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["0"]
        report = self._report(root)
        assert self._states(report) == {"1": MET, "2": MET, "3": MET, "4": MET}
        assert report["exit_code"] == 0
        assert report["summary"]["criteria"][MET] == 4
        # Los entregables que dependen del cluster quedan sin medir y no cambian la salida.
        deliverables = {item["id"]: item for item in report["deliverables"]}
        assert deliverables["pods"]["state"] == NOT_MEASURED
        assert deliverables["dashboard"]["state"] == NOT_MEASURED
        assert deliverables["feast"]["state"] == NOT_MEASURED
        assert not any(item["blocking"] for item in deliverables.values())

    def test_a_failed_criterion_exits_one(self, tmp_path: Path) -> None:
        """Un throughput de cluster por debajo del umbral es FALLADO, aunque lo demas cierre."""
        root = _project(
            tmp_path,
            throughput_env="cluster",
            throughput_value=12_000.0,
            latency_env="local",
            latency_value=72.0,
        )
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["1"]
        report = self._report(root)
        assert self._states(report) == {"1": FAILED, "2": MET, "3": NOT_MEASURED, "4": MET}
        assert report["exit_code"] == 1

    def test_local_measurements_exit_two(self, tmp_path: Path) -> None:
        """El estado esperado hasta que el cluster mida: 2 y 4 cumplidos, 1 y 3 sin medir."""
        root = _project(
            tmp_path,
            throughput_env="local",
            throughput_value=99_000.0,
            latency_env="local",
            latency_value=10.0,
        )
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["2"]
        report = self._report(root)
        assert self._states(report) == {"1": NOT_MEASURED, "2": MET, "3": NOT_MEASURED, "4": MET}
        # Las cifras locales se muestran, pero como informativas.
        criterion_1 = report["criteria"][0]
        assert "INFORMATIVA" not in criterion_1["detail"]
        assert "informativa" in criterion_1["detail"]
        assert criterion_1["evidence"]["value"] == 99_000.0
        assert criterion_1["evidence"]["environment"] == "local"

    def test_skipped_suite_is_not_a_pass(self, tmp_path: Path) -> None:
        """El caso que motiva el script: pytest sale con 0 y no se ha medido nada."""
        root = _project(
            tmp_path,
            throughput_env="cluster",
            throughput_value=62_000.0,
            latency_env="cluster",
            latency_value=72.0,
            suite_skips=True,
        )
        # Un JSON de una ejecucion anterior con 1,0 no debe rescatar el criterio.
        _metrics(root / "tests" / "results" / "validator_metrics_tep.json", 1.0)
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["2"]
        report = self._report(root)
        assert self._states(report)["2"] == NOT_MEASURED
        assert "SALTADOS" in report["criteria"][1]["detail"]

    def test_missing_dataset_is_not_a_failure(self, tmp_path: Path) -> None:
        """Sin data/processed/tep el criterio 2 queda sin medir y pytest ni se ejecuta."""
        root = _project(
            tmp_path,
            throughput_env="cluster",
            throughput_value=62_000.0,
            latency_env="cluster",
            latency_value=72.0,
        )
        shutil.rmtree(root / "data")
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["2"]
        report = self._report(root)
        assert self._states(report)["2"] == NOT_MEASURED
        assert "Faltan los datos" in report["criteria"][1]["detail"]
        assert not (root / "tests" / "results" / "phase2_stuck.junit.xml").exists()

    def test_a_crash_while_measuring_one_criterion_does_not_stop_the_others(
        self, tmp_path: Path
    ) -> None:
        """Invoke-Guarded convierte la excepcion de un criterio en NO MEDIDO."""
        root = _project(
            tmp_path,
            throughput_env="cluster",
            throughput_value=62_000.0,
            latency_env="cluster",
            latency_value=72.0,
        )
        (root / "dvc.yaml").unlink()
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["2"]
        states = self._states(self._report(root))
        assert states["4"] == NOT_MEASURED
        assert states["1"] == MET
        assert states["2"] == MET

    def test_an_internal_error_exits_three(self, tmp_path: Path) -> None:
        """Un fallo del propio verificador no se disfraza de veredicto sobre la fase."""
        root = _project(
            tmp_path,
            throughput_env="local",
            throughput_value=1.0,
            latency_env="local",
            latency_value=1.0,
        )
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("un fichero donde deberia haber un directorio", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.startswith("COV_CORE")}
        result = subprocess.run(  # noqa: S603  (pwsh resuelto con shutil.which, script propio)
            [
                str(PWSH),
                "-NoProfile",
                "-File",
                str(SCRIPT),
                "-ProjectRoot",
                str(root),
                "-PythonPath",
                sys.executable,
                "-ResultsDir",
                str(blocker / "results"),
                "-SkipPytest",
                "-SkipCluster",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=env,
        )
        assert result.returncode == 3, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "ERROR del verificador" in result.stdout
