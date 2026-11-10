"""Verify-Phase3.ps1 (F3-M7): tres estados, tres codigos de salida y su contrato.

El script decide si la Fase 3 esta cerrada, asi que se prueba como Verify-Phase2.ps1:
ejecutando PowerShell 7 sobre artefactos sinteticos con el formato EXACTO que
producen el stage `simulate` (parquet, `manifest.json`, metricas), pytest (JUnit) y
DVC (`dvc.lock`), sin mocks de su logica.

Cuatro grupos:

- Contrato estatico: lo que el script lee (rutas, umbrales, columnas, etapa, tests)
  coincide con lo que el codigo de simulacion produce y declara. Un nombre
  desalineado no daria un error sino un NO MEDIDO permanente.
- Decisiones (capa 2): cada regla de estado de los cuatro criterios con su caso
  borde, incluido el que motiva el patron: una suite saltada sale con codigo 0.
- Servicio (capa 3): el lector de parquet real sobre parquet sinteticos y la
  reproduccion desde cero con `git` y `dvc` falsos (no se ejecuta el stage).
- Orquestacion: el script completo sobre un proyecto sintetico.

Los tests de PowerShell se saltan si `pwsh` no esta instalado (CI Linux ya lo trae).
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from data.simulation.dataset_builder import LABEL_COLUMNS, MANIFEST_NAME, file_sha256
from data.simulation.plant_params import load_plant_params
from data.simulation.scenario import FAULT_KINDS, FaultAxis
from data.simulation.scenario_planner import parse_planner_settings
from data.simulation.simulation_params import load_simulation_params

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "infra" / "scripts" / "Verify-Phase3.ps1"
PHASE2_SCRIPT = REPO_ROOT / "infra" / "scripts" / "Verify-Phase2.ps1"
SAFETY_SUITE = REPO_ROOT / "tests" / "safety" / "test_safety_constraints.py"
COMMITTED_METRICS = REPO_ROOT / "metrics" / "simulation_metrics.json"

PWSH = shutil.which("pwsh")
requires_pwsh = pytest.mark.skipif(PWSH is None, reason="PowerShell 7 no esta instalado")

MET = "CUMPLIDO"
FAILED = "FALLADO"
NOT_MEASURED = "NO MEDIDO"

# Cada script de prueba se ejecuta dot-sourced: el guard del punto de entrada evita que
# el script corra, y las funciones y variables quedan disponibles.
_HARNESS = r"""
$ErrorActionPreference = "Stop"
. "__SCRIPT__" -ProjectRoot "__ROOT__" -PythonPath "__PYTHON__" __SWITCHES__
function Case($name, $v) { Write-Output ("CASE=$name|" + $v.state + "|" + $v.detail) }
__BODY__
"""

_FAKE_DVC_CLEAN = r"""
function dvc { $global:LASTEXITCODE = 0; "{}" }
"""

_SIMULATE_YAML = (
    "stages:\n"
    "  other:\n"
    "    cmd: x\n"
    "  simulate:\n"
    "    cmd: python data/generators/reactor_simulator.py --workers 0\n"
)
_SIMULATE_LOCK = (
    "schema: '2.0'\n"
    "stages:\n"
    "  other:\n"
    "    cmd: x\n"
    "    outs:\n"
    "    - path: data/other\n"
    "      hash: md5\n"
    "      md5: 0000.dir\n"
    "  simulate:\n"
    "    cmd: python data/generators/reactor_simulator.py --workers 0\n"
    "    deps:\n"
    "    - path: data/simulation\n"
    "      hash: md5\n"
    "      md5: dddd.dir\n"
    "    outs:\n"
    "    - path: data/raw/simulation\n"
    "      hash: md5\n"
    "      md5: aaaa.dir\n"
    "    - path: metrics/simulation_metrics.json\n"
    "      hash: md5\n"
    "      md5: bbbb\n"
)

_COMPLIANCE_TESTS = (
    "test_conservation_under_tolerance_in_every_run",
    "test_expected_violation_matrix",
)


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


def _script_variable(name: str) -> str:
    """Return the first quoted or numeric value assigned to a script variable."""
    pattern = rf'^\s*(?:\[\w+\])?\${name}\s*=\s*"?([^"\s#,]+)"?'
    match = re.search(pattern, _read(SCRIPT), re.MULTILINE)
    assert match, f"Verify-Phase3.ps1 no declara ${name}"
    return match.group(1)


def _script_array(name: str) -> list[str]:
    """Return the quoted strings of a `$name = @(...)` assignment."""
    match = re.search(rf"^\${name}\s*=\s*@\(([^)]*)\)", _read(SCRIPT), re.MULTILINE)
    assert match, f"Verify-Phase3.ps1 no declara ${name}"
    return re.findall(r'"([^"]+)"', match.group(1))


def _committed_metrics() -> dict[str, Any]:
    """Return the metrics the `simulate` stage committed (`cache: false`)."""
    document: dict[str, Any] = json.loads(_read(COMMITTED_METRICS))
    return document


def _write_metrics(path: Path, mutate: Callable[[dict[str, Any]], None] | None = None) -> Path:
    """Write a copy of the committed metrics, optionally mutated, and return its path."""
    document = copy.deepcopy(_committed_metrics())
    if mutate is not None:
        mutate(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _junit(path: Path, cases: list[tuple[str, str]]) -> Path:
    """Write a JUnit report with the shape pytest produces.

    Args:
        path: Destination file.
        cases: (test name, outcome) pairs; outcome is passed, skipped, failure or error.

    Returns:
        The path written.
    """
    children = {
        "passed": "",
        "skipped": '<skipped type="pytest.skip" message="sin pyarrow">x</skipped>',
        "failure": '<failure message="assert False">trace</failure>',
        "error": '<error message="collection">trace</error>',
    }
    rows = "".join(
        f'<testcase classname="tests.safety.test_safety_constraints" name="{name}" '
        f'time="0.01">{children[outcome]}</testcase>'
        for name, outcome in cases
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="{len(cases)}" time="1.0">{rows}'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return path


def _safety_passed(extra: int = 11) -> list[tuple[str, str]]:
    """Return a passing safety suite: the compliance tests plus `extra` others."""
    others = [(f"test_other_{n}", "passed") for n in range(extra)]
    return [*[(name, "passed") for name in _COMPLIANCE_TESTS], *others]


# ---------------------------------------------------------------------------
# Synthetic dataset (the layout of the `simulate` stage, a few rows per run)
# ---------------------------------------------------------------------------


def _run_frame(n_rows: int, n_faulty: int) -> pd.DataFrame:
    """Return a long frame with the ground-truth columns and `n_faulty` labelled rows."""
    healthy = n_rows - n_faulty
    injected = [pd.Timestamp("2026-01-01", tz="UTC")] * n_faulty + [pd.NaT] * healthy
    return pd.DataFrame(
        {
            "value": np.arange(n_rows, dtype=np.float64),
            "scenario": pd.array(["void_formation"] * n_rows, dtype="string"),
            "true_value": np.arange(n_rows, dtype=np.float64),
            "corrupted": np.zeros(n_rows, dtype=bool),
            "process_fault": pd.array(["void_formation"] * n_faulty + ["none"] * healthy, "string"),
            "process_severity": pd.array(["CRITICAL"] * n_faulty + [None] * healthy, "string"),
            "instrument_fault": pd.array(["none"] * n_rows, dtype="string"),
            "instrument_severity": pd.array([None] * n_rows, dtype="string"),
            "injected_at": pd.Series(injected, dtype="datetime64[ns, UTC]"),
        }
    )


def _dataset(
    root: Path,
    frames: dict[str, pd.DataFrame],
    *,
    tamper: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Write runs and their manifest under `root/data/raw/simulation`.

    Args:
        root: Project root.
        frames: run_id -> long frame.
        tamper: Optional edit of the manifest before it is written.

    Returns:
        The dataset directory.
    """
    dataset = root / "data" / "raw" / "simulation"
    runs = []
    for run_id, frame in frames.items():
        relative = f"fault_type=void_formation/seed=42/{run_id}.parquet"
        target = dataset / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        labelled = int(
            ((frame["process_fault"] != "none") | (frame["instrument_fault"] != "none"))
            .fillna(value=False)
            .sum()
        )
        runs.append(
            {
                "run_id": run_id,
                "path": relative,
                "sha256": file_sha256(target),
                "rows": len(frame),
                "labelled": labelled,
            }
        )
    manifest: dict[str, Any] = {
        "version": 1,
        "n_runs": len(runs),
        "total_readings": sum(r["rows"] for r in runs),
        "labelled_readings": sum(r["labelled"] for r in runs),
        "runs": runs,
    }
    if tamper is not None:
        tamper(manifest)
    (dataset / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def _volume_case(name: str, root: Path, threshold: int = 10) -> str:
    """Return the PowerShell lines that evaluate criterion 3 over a synthetic dataset."""
    dataset = root / "data" / "raw" / "simulation"
    return (
        f"$VolumeThreshold = {threshold}\n"
        f'$m = Read-JsonFile -Path "{dataset / MANIFEST_NAME}"\n'
        "$paths = @($m.Data.runs | ForEach-Object { $_.path })\n"
        f'$f = Get-ParquetFacts -DatasetRoot "{dataset}" -Python "{sys.executable}" '
        "-RelativePaths $paths\n"
        f'Case "{name}" (Resolve-VolumeVerdict -Manifest $m -Probe $f.Probe '
        "-ProbeProblem $f.Problem)\n"
    )


# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


class TestScriptContract:
    """What the script reads is what the simulation code produces and declares."""

    def test_follows_the_script_conventions(self) -> None:
        """PowerShell 7, cuatro capas, solo ASCII y sin ProjectId (la fase no toca GCP)."""
        text = _read(SCRIPT)
        assert text.startswith("#Requires -Version 7.0")
        for layer in ("CAPA 1", "CAPA 2", "CAPA 3", "CAPA 4"):
            assert layer in text, f"Falta la {layer}"
        assert text.isascii()
        assert "kubectl" not in text
        assert "ProjectId" not in text

    def test_documents_the_four_exit_codes(self) -> None:
        """La cabecera explica 0, 1, 2 y 3, y las variables llevan esos valores."""
        text = _read(SCRIPT)
        for code in ("0  todos", "1  algun", "2  ninguno", "3  error interno"):
            assert code in text
        names = ("ExitMet", "ExitFailed", "ExitNotMeasured", "ExitError")
        assert [int(_script_variable(n)) for n in names] == [0, 1, 2, 3]

    def test_states_are_the_ones_of_phase_2(self) -> None:
        """Los tres estados se escriben igual que en Verify-Phase2.ps1 y que en estos tests."""

        def states(text: str) -> list[str]:
            names = ("StateMet", "StateFailed", "StateNotMeasured")
            found = [re.search(rf'^\${n}\s*=\s*"([^"]+)"', text, re.MULTILINE) for n in names]
            return [match.group(1) if match else "" for match in found]

        assert states(_read(SCRIPT)) == states(_read(PHASE2_SCRIPT)) == [MET, FAILED, NOT_MEASURED]

    def test_thresholds_are_the_ones_the_simulation_declares(self) -> None:
        """Tipos del TDD, puerta D6, tolerancia D7 y el volumen del plan de fases."""
        params = load_simulation_params(REPO_ROOT / "params.yaml")
        plant = load_plant_params(REPO_ROOT / params.plant_parameters_path)
        planner = parse_planner_settings(plant.scenario_planner)
        assert int(_script_variable("FaultTypesRequired")) == len(FAULT_KINDS) == 8
        assert float(_script_variable("CoverageGate")) == planner.coverage_gate
        assert float(_script_variable("ConservationTolerance")) == (
            plant.limits.conservation_tolerance
        )
        assert int(_script_variable("VolumeThreshold")) == 1_000_000
        plan = _read(REPO_ROOT / "Docs" / "ReactorGuard_Plan_Fases.md")
        assert "> 1M readings etiquetados" in plan

    def test_evidence_paths_are_the_ones_of_params_yaml(self) -> None:
        """Metricas, dataset y manifiesto son los que escribe el stage."""
        params = load_simulation_params(REPO_ROOT / "params.yaml")
        assert _script_variable("MetricsFile") == params.metrics_path.as_posix()
        assert _script_variable("DatasetDir") == params.output_dir.as_posix()
        assert _script_variable("ManifestName") == MANIFEST_NAME

    def test_the_stage_and_its_outputs_are_declared_in_dvc(self) -> None:
        """La etapa existe y sus salidas son el dataset y las metricas que se leen."""
        stage = _script_variable("SimulateStage")
        declared = yaml.safe_load(_read(REPO_ROOT / "dvc.yaml"))["stages"][stage]
        metrics = {name for entry in declared["metrics"] for name in entry}
        assert declared["outs"] == [_script_variable("DatasetDir")]
        assert metrics == {_script_variable("MetricsFile")}

    def test_ground_truth_columns_are_label_columns_of_the_frame(self) -> None:
        """Las columnas que el criterio 3 exige estan en el frame largo del dataset."""
        columns = _script_array("GroundTruthColumns")
        axes = _script_array("FaultAxisColumns")
        assert set(columns) <= set(LABEL_COLUMNS)
        assert set(axes) <= set(columns)
        assert axes == ["process_fault", "instrument_fault"]
        assert _script_variable("NoFaultLabel") == FaultAxis.NONE.value

    def test_safety_suite_runs_the_compliance_tests(self) -> None:
        """Los ficheros de la suite existen y definen los tests que el criterio exige."""
        files = _script_array("SafetyTestFiles")
        assert files == [SAFETY_SUITE.relative_to(REPO_ROOT).as_posix()]
        names = _script_array("ComplianceTestNames")
        assert names == list(_COMPLIANCE_TESTS)
        for name in names:
            assert f"def {name}(" in _read(SAFETY_SUITE)
        assert "pytestmark" not in _read(SAFETY_SUITE), "la suite no debe llevar skip de modulo"

    def test_deliverables_point_at_real_artifacts(self) -> None:
        """El informe de Fase 2 y los documentos que se buscan existen con ese nombre."""
        for relative in _script_array("RequiredDocs"):
            assert (REPO_ROOT / relative).is_file(), f"No existe {relative}"
        assert re.search(
            rf'^\$ReportFile\s*=\s*"{_script_variable("Phase2ReportFile")}"',
            _read(PHASE2_SCRIPT),
            re.MULTILINE,
        )
        assert _script_variable("ResultsDir") == "tests/results"

    def test_committed_metrics_carry_every_field_the_script_reads(self) -> None:
        """Las metricas versionadas tienen los campos de los criterios 1 y 2."""
        metrics = _committed_metrics()
        coverage, compliance = metrics["coverage"], metrics["compliance"]
        for key in ("n_types", "gate", "passed", "partitions"):
            assert key in coverage
        for key in ("covered", "grid_size"):
            assert isinstance(coverage["overall"][key], int)
        for key in (
            "conservation_failures",
            "max_instant_residual",
            "max_interval_residual",
            "conservation_tolerance",
            "unexpected_violations",
            "violations_as_expected",
        ):
            assert key in compliance


# ---------------------------------------------------------------------------
# Criterion 1: coverage
# ---------------------------------------------------------------------------


@requires_pwsh
class TestCoverageCriterion:
    """8/8 tipos y la rejilla, recalculados; el `passed` del informe no decide."""

    def test_each_rule_of_the_state_table(self, tmp_path: Path) -> None:
        """Metricas del stage, tipo ausente, rejilla corta, borde y metricas rotas."""

        def overall(covered: int, grid: int) -> Callable[[dict[str, Any]], None]:
            def mutate(doc: dict[str, Any]) -> None:
                doc["coverage"]["overall"].update(covered=covered, grid_size=grid)

            return mutate

        def seven_types(doc: dict[str, Any]) -> None:
            doc["coverage"]["n_types"] = 7

        def lying(doc: dict[str, Any]) -> None:
            doc["coverage"].update(n_types=7, passed=True, gate=0.5)

        def no_overall(doc: dict[str, Any]) -> None:
            del doc["coverage"]["overall"]

        scenarios: dict[str, tuple[Path, str]] = {
            "committed": (COMMITTED_METRICS, MET),
            "seven_types": (_write_metrics(tmp_path / "a.json", seven_types), FAILED),
            "grid_below_gate": (_write_metrics(tmp_path / "b.json", overall(136, 144)), FAILED),
            "grid_exactly_at_gate": (_write_metrics(tmp_path / "c.json", overall(19, 20)), MET),
            "report_approves_itself": (_write_metrics(tmp_path / "d.json", lying), FAILED),
            "no_overall_block": (_write_metrics(tmp_path / "e.json", no_overall), NOT_MEASURED),
            "covered_above_grid": (
                _write_metrics(tmp_path / "f.json", overall(150, 144)),
                NOT_MEASURED,
            ),
            "absent": (tmp_path / "absent.json", NOT_MEASURED),
        }
        body = "\n".join(
            f'$v = Resolve-CoverageVerdict -Metrics (Read-JsonFile -Path "{path}")\n'
            f'Case "{name}" $v\n'
            f'Write-Output ("WARN={name}|" + ($v.warnings -join " / "))'
            for name, (path, _) in scenarios.items()
        )
        output = _run(tmp_path, body)
        cases = _cases(output)

        for name, (_, expected) in scenarios.items():
            assert cases[name][0] == expected, f"{name}: {cases[name]}"
        assert cases["committed"][1] == "8/8 tipos; rejilla 144/144 (100.0 %, umbral >= 95 %)"
        warnings = dict(line.split("|", 1) for line in output["WARN"])
        assert warnings["committed"] == ""
        assert "passed=True" in warnings["report_approves_itself"]
        assert "gate 0.5" in warnings["report_approves_itself"]

    def test_partitions_are_reported_and_do_not_block(self, tmp_path: Path) -> None:
        """Val y test cubren el 38-40 % de la rejilla (M5-5): se informa, no bloquea."""
        body = (
            f'$v = Resolve-CoverageVerdict -Metrics (Read-JsonFile -Path "{COMMITTED_METRICS}")\n'
            "foreach ($k in $v.evidence.partitions.Keys) {\n"
            '    Write-Output ("PART=" + $k + "=" + $v.evidence.partitions[$k]) }\n'
            'Write-Output ("STATE=" + $v.state)'
        )
        output = _run(tmp_path, body)
        partitions = {k: float(v) for k, v in (line.split("=") for line in output["PART"])}
        expected = _committed_metrics()["coverage"]["partitions"]
        assert partitions == pytest.approx({k: v["grid_fraction"] for k, v in expected.items()})
        assert partitions["val"] < 0.95
        assert output["STATE"] == [MET]


# ---------------------------------------------------------------------------
# Criterion 2: compliance
# ---------------------------------------------------------------------------


@requires_pwsh
class TestComplianceCriterion:
    """Conservacion, matriz y suite: un FALLADO de cualquier fuente gana."""

    def test_each_rule_of_the_state_table(self, tmp_path: Path) -> None:
        """Cada combinacion de metricas y JUnit, con su estado."""

        def conservation(doc: dict[str, Any]) -> None:
            doc["compliance"]["conservation_failures"] = 3

        def residual(doc: dict[str, Any]) -> None:
            doc["compliance"]["max_interval_residual"] = 2.0e-6

        def unexpected(doc: dict[str, Any]) -> None:
            doc["compliance"]["unexpected_violations"] = {"sim_run_007": {"power_flux": 2}}

        def broken(doc: dict[str, Any]) -> None:
            del doc["compliance"]["max_instant_residual"]

        good = COMMITTED_METRICS
        scenarios: dict[str, tuple[Path, list[tuple[str, str]] | None, str]] = {
            "all_good": (good, _safety_passed(), MET),
            "conservation_failures": (
                _write_metrics(tmp_path / "a.json", conservation),
                _safety_passed(),
                FAILED,
            ),
            "residual_above_tolerance": (
                _write_metrics(tmp_path / "b.json", residual),
                _safety_passed(),
                FAILED,
            ),
            "violation_outside_matrix": (
                _write_metrics(tmp_path / "c.json", unexpected),
                _safety_passed(),
                FAILED,
            ),
            # Un fallo de las metricas no se esconde tras una suite sin ejecutar.
            "metrics_fail_without_junit": (
                _write_metrics(tmp_path / "d.json", conservation),
                None,
                FAILED,
            ),
            "a_safety_test_fails": (good, [*_safety_passed(), ("test_x", "failure")], FAILED),
            "a_safety_test_errors": (good, [*_safety_passed(), ("test_y", "error")], FAILED),
            "metrics_absent_suite_fails": (
                tmp_path / "absent.json",
                [*_safety_passed(), ("test_x", "failure")],
                FAILED,
            ),
            # El caso de la trampa: todo saltado, pytest sale con 0.
            "everything_skipped": (
                good,
                [(name, "skipped") for name, _ in _safety_passed()],
                NOT_MEASURED,
            ),
            "compliance_test_not_run": (
                good,
                [("test_expected_violation_matrix", "passed"), ("test_other", "passed")],
                NOT_MEASURED,
            ),
            "no_junit": (good, None, NOT_MEASURED),
            "metrics_absent": (tmp_path / "absent.json", _safety_passed(), NOT_MEASURED),
            "metrics_broken": (
                _write_metrics(tmp_path / "e.json", broken),
                _safety_passed(),
                NOT_MEASURED,
            ),
        }
        body = []
        for name, (metrics, junit, _) in scenarios.items():
            junit_path = tmp_path / f"{name}.xml"
            if junit is not None:
                _junit(junit_path, junit)
            body.append(
                f'Case "{name}" (Resolve-ComplianceVerdict '
                f'-Metrics (Read-JsonFile -Path "{metrics}") '
                f'-JUnit (Read-JUnitSummary -Path "{junit_path}"))'
            )
        cases = _cases(_run(tmp_path, "\n".join(body)))

        for name, (_, _, expected) in scenarios.items():
            assert cases[name][0] == expected, f"{name}: {cases[name]}"
        assert "13 tests de seguridad pasan sin saltos" in cases["all_good"][1]
        assert "sim_run_007 power_flux=2" in cases["violation_outside_matrix"][1]
        assert "SALTADOS" in cases["everything_skipped"][1]
        not_run = cases["compliance_test_not_run"][1]
        assert "test_conservation_under_tolerance_in_every_run" in not_run
        assert "2e-6" in cases["residual_above_tolerance"][1]

    def test_a_real_pytest_run_is_classified_by_what_it_executed(self, tmp_path: Path) -> None:
        """Invoke-SafetySuite sobre una suite sintetica: el JUnit REAL de pytest se interpreta."""
        tests = "\n\n".join(f"def {name}():\n    assert True" for name in _COMPLIANCE_TESTS)
        for variant, header in (
            ("ok", ""),
            ("skip", "import pytest\n\npytestmark = pytest.mark.skip(reason='sin datos')\n\n"),
        ):
            suite = tmp_path / variant / "tests" / "safety"
            suite.mkdir(parents=True)
            (suite / "test_safety_constraints.py").write_text(header + tests + "\n", "utf-8")

        body = rf"""
foreach ($name in @("ok", "skip")) {{
    $root = Join-Path "{tmp_path}" $name
    $junit = Join-Path $root "tests/results/junit.xml"
    $run = Invoke-SafetySuite -Root $root -Python "{sys.executable}" -JUnitPath $junit
    $summary = Read-JUnitSummary -Path $junit
    $metrics = Read-JsonFile -Path "{COMMITTED_METRICS}"
    $v = Resolve-ComplianceVerdict -Metrics $metrics -JUnit $summary
    $facts = "exit=" + $run.ExitCode + " passed=" + $summary.Passed + " skipped=" + $summary.Skipped
    Write-Output ("CASE=$name|" + $v.state + "|" + $facts)
}}
"""
        cases = _cases(_run(tmp_path, body))

        assert cases["ok"] == (MET, "exit=0 passed=2 skipped=0")
        # pytest sale con codigo 0 aunque todo se haya saltado: el estado lo da el JUnit.
        assert cases["skip"] == (NOT_MEASURED, "exit=0 passed=0 skipped=2")


# ---------------------------------------------------------------------------
# Criterion 3: volume, counted in the parquet files
# ---------------------------------------------------------------------------


@requires_pwsh
class TestVolumeCriterion:
    """Las lecturas se cuentan en los parquet y se contrastan con el manifiesto."""

    def test_each_rule_of_the_state_table(self, tmp_path: Path) -> None:
        """Dataset coherente, borde del umbral, manipulaciones y dataset incompleto."""
        two_runs = {"sim_run_000": _run_frame(6, 2), "sim_run_001": _run_frame(6, 0)}
        roots = {
            name: tmp_path / name
            for name in (
                "met",
                "at_threshold",
                "sha_tampered",
                "rows_tampered",
                "labelled_tampered",
                "total_tampered",
                "missing_column",
                "unlabelled_rows",
                "absent_parquet",
            )
        }
        for name in ("met", "at_threshold"):
            _dataset(roots[name], two_runs)

        def first_run(**changes: Any) -> Callable[[dict[str, Any]], None]:
            def tamper(manifest: dict[str, Any]) -> None:
                manifest["runs"][0].update(changes)

            return tamper

        def total(manifest: dict[str, Any]) -> None:
            manifest["total_readings"] = 13

        _dataset(roots["sha_tampered"], two_runs, tamper=first_run(sha256="0" * 64))
        _dataset(roots["rows_tampered"], two_runs, tamper=first_run(rows=7))
        _dataset(roots["labelled_tampered"], two_runs, tamper=first_run(labelled=1))
        _dataset(roots["total_tampered"], two_runs, tamper=total)
        _dataset(
            roots["missing_column"],
            {**two_runs, "sim_run_000": _run_frame(6, 2).drop(columns="injected_at")},
        )
        unlabelled = _run_frame(6, 2)
        unlabelled.loc[5, "instrument_fault"] = pd.NA
        _dataset(roots["unlabelled_rows"], {**two_runs, "sim_run_000": unlabelled})
        dataset = _dataset(roots["absent_parquet"], two_runs)
        (dataset / "fault_type=void_formation" / "seed=42" / "sim_run_001.parquet").unlink()

        body = "".join(
            _volume_case(name, root, threshold=12 if name == "at_threshold" else 10)
            for name, root in roots.items()
        )
        body += (
            f'Case "no_manifest" (Resolve-VolumeVerdict -Manifest '
            f'(Read-JsonFile -Path "{tmp_path / "absent.json"}"))\n'
            f'Case "probe_failed" (Resolve-VolumeVerdict -Manifest '
            f'(Read-JsonFile -Path "{roots["met"] / "data/raw/simulation" / MANIFEST_NAME}") '
            '-Probe $null -ProbeProblem "sin pyarrow")\n'
        )
        cases = _cases(_run(tmp_path, body))

        expected = {
            "met": MET,
            # "> 1M": el umbral exacto no basta.
            "at_threshold": FAILED,
            "sha_tampered": FAILED,
            "rows_tampered": FAILED,
            "labelled_tampered": FAILED,
            "total_tampered": FAILED,
            "missing_column": FAILED,
            "unlabelled_rows": FAILED,
            "absent_parquet": NOT_MEASURED,
            "no_manifest": NOT_MEASURED,
            "probe_failed": NOT_MEASURED,
        }
        for name, state in expected.items():
            assert cases[name][0] == state, f"{name}: {cases[name]}"
        assert cases["met"][1] == (
            "12 lecturas con verdad-terreno en 2 runs (umbral > 10); 2 con fallo (16.67 %)"
        )
        assert "sha256 distinto" in cases["sha_tampered"][1]
        assert "7 del manifiesto" in cases["rows_tampered"][1]
        assert "injected_at" in cases["missing_column"][1]
        assert "1 lecturas sin etiqueta" in cases["unlabelled_rows"][1]
        assert "declara 13 lecturas" in cases["total_tampered"][1]
        assert "sim_run_001.parquet" in cases["absent_parquet"][1]
        assert "sin pyarrow" in cases["probe_failed"][1]

    def test_the_probe_reads_what_pandas_reads(self, tmp_path: Path) -> None:
        """Filas, sha256, columnas y etiquetas del lector coinciden con pandas y hashlib."""
        frame = _run_frame(9, 4)
        frame.loc[[0, 6], "instrument_fault"] = "drift"
        dataset = _dataset(tmp_path, {"sim_run_000": frame})
        relative = "fault_type=void_formation/seed=42/sim_run_000.parquet"
        body = (
            f'$f = Get-ParquetFacts -DatasetRoot "{dataset}" -Python "{sys.executable}" '
            f'-RelativePaths @("{relative}", "missing.parquet")\n'
            "foreach ($file in $f.Probe.files) {\n"
            '    Write-Output ("FILE=" + ($file | ConvertTo-Json -Compress -Depth 4)) }\n'
        )
        files = [json.loads(line) for line in _run(tmp_path, body)["FILE"]]

        assert files[1] == {"path": "missing.parquet", "exists": False}
        facts = files[0]
        assert facts["sha256"] == file_sha256(dataset / relative)
        assert facts["rows"] == 9
        assert facts["columns"] == list(frame.columns)
        assert facts["ground_truth"] == 9
        # 4 de proceso y 2 de instrumento, una de ellas en una fila ya con fallo de proceso.
        assert facts["labelled"] == 5


# ---------------------------------------------------------------------------
# Criterion 4: reproducibility of the simulate stage
# ---------------------------------------------------------------------------


@requires_pwsh
class TestReproducibilityCriterion:
    """dvc status de la etapa, lock completo y, si se pide, la reproduccion desde cero."""

    def test_the_parsers_agree_with_the_real_dvc_files(self, tmp_path: Path) -> None:
        """La etapa esta declarada y registrada, y sus hashes son los del YAML."""
        body = rf"""
$yaml = Get-Content -LiteralPath "{REPO_ROOT / "dvc.yaml"}" -Raw
$lock = Get-Content -LiteralPath "{REPO_ROOT / "dvc.lock"}" -Raw
Write-Output ("DECLARED=" + ($SimulateStage -in @(Get-StageNames -Text $yaml)))
Write-Output ("LOCKED=" + ($SimulateStage -in @(Get-StageNames -Text $lock)))
$hashes = Select-StageHashes -Hashes (Get-LockOutputHashes -Text $lock) -Stage $SimulateStage
foreach ($key in $hashes.Keys) {{ Write-Output ("HASH=" + $key + "=" + $hashes[$key]) }}
"""
        output = _run(tmp_path, body)

        assert output["DECLARED"] == ["True"]
        assert output["LOCKED"] == ["True"]
        lock = yaml.safe_load(_read(REPO_ROOT / "dvc.lock"))["stages"]["simulate"]
        expected = {f"simulate:{out['path']}": out["md5"] for out in lock["outs"]}
        assert dict(entry.rsplit("=", 1) for entry in output["HASH"]) == expected
        assert set(expected) == {
            "simulate:data/raw/simulation",
            "simulate:metrics/simulation_metrics.json",
        }

    def test_each_rule_of_the_state_table(self, tmp_path: Path) -> None:
        """Etapa sin declarar o sin lock, status sucio o roto y los cuatro desenlaces."""
        body = r"""
function Status($json) { Get-DvcStatusSummary -Status ($json | ConvertFrom-Json -AsHashtable) }
function Scratch($state, $detail) { [pscustomobject]@{ Status = $state; Detail = $detail } }
function Repro([bool]$d, [bool]$l, $s, $p = "", $x = $null) {
    Resolve-ReproducibilityVerdict -Declared $d -Locked $l -StatusSummary $s `
        -StatusProblem $p -Scratch $x
}
$clean = Status '{}'
$deps = Status '{"simulate": [{"changed deps": {"params.yaml": "modified"}}]}'
$deleted = Status '{"simulate": [{"changed outs": {"data/raw/simulation": "deleted"}}]}'
$modified = Status '{"simulate": [{"changed outs": {"data/raw/simulation": "modified"}}]}'
Case "clean" (Repro $true $true $clean)
Case "not_declared" (Repro $false $false $null)
Case "not_locked" (Repro $true $false $null)
Case "changed_deps" (Repro $true $true $deps)
Case "out_deleted" (Repro $true $true $deleted)
Case "out_modified" (Repro $true $true $modified)
Case "status_unavailable" (Repro $true $true $null "sin dvc")
Case "scratch_matched" (Repro $true $true $clean "" (Scratch "matched" "2 salidas"))
Case "scratch_mismatch" (Repro $true $true $clean "" (Scratch "mismatch" "a:b"))
Case "scratch_failed" (Repro $true $true $clean "" (Scratch "failed" "boom"))
Case "scratch_unavailable" (Repro $true $true $clean "" (Scratch "unavailable" "sin git"))
"""
        cases = _cases(_run(tmp_path, body))
        expected = {
            "clean": MET,
            "not_declared": FAILED,
            "not_locked": FAILED,
            "changed_deps": NOT_MEASURED,
            "out_deleted": NOT_MEASURED,
            "out_modified": FAILED,
            "status_unavailable": NOT_MEASURED,
            "scratch_matched": MET,
            "scratch_mismatch": FAILED,
            "scratch_failed": FAILED,
            # Si se pidio la prueba fuerte y no se pudo completar, no se da por cumplida.
            "scratch_unavailable": NOT_MEASURED,
        }
        for name, state in expected.items():
            assert cases[name][0] == state, f"{name}: {cases[name]}"
        assert "sin -FromScratch" in cases["clean"][1]
        assert "bit a bit" in cases["scratch_matched"][1]

    def test_get_dvc_status_asks_only_for_the_stage(self, tmp_path: Path) -> None:
        """`dvc status simulate --json`: el resto del pipeline no cuenta en la Fase 3."""
        body = r"""
function dvc {
    $global:LASTEXITCODE = 0
    Write-Output ($args -join " ") | Out-File -LiteralPath "__ARGS__" -Encoding utf8
    if ($env:FAKE_DVC -eq "error") { $global:LASTEXITCODE = 255; "ERROR: unexpected" }
    elseif ($env:FAKE_DVC -eq "garbage") { "not json" }
    else { "{}" }
}
foreach ($mode in @("clean", "garbage", "error")) {
    $env:FAKE_DVC = $mode
    $r = Get-DvcStatus -Root "__ROOT__" -Stage $SimulateStage
    Write-Output ("CASE=$mode|" + ($null -ne $r.Summary) + "|" + $r.Problem)
}
""".replace("__ARGS__", str(tmp_path / "args.txt")).replace("__ROOT__", str(tmp_path))
        cases = _cases(_run(tmp_path, body))

        assert _read(tmp_path / "args.txt").strip() == "status simulate --json"
        assert cases["clean"] == ("True", "")
        assert cases["garbage"][0] == "False"
        assert "no devolvio JSON" in cases["garbage"][1]
        assert cases["error"][0] == "False"
        assert "255" in cases["error"][1]

    def test_from_scratch_rebuilds_only_the_stage_in_a_copy(self, tmp_path: Path) -> None:
        """Copia sin dvc.lock, repro de la etapa, comparacion de SUS hashes y limpieza."""
        root = tmp_path / "project"
        (root / "configs").mkdir(parents=True)
        (root / "dvc.yaml").write_text(_SIMULATE_YAML, encoding="utf-8")
        (root / "dvc.lock").write_text(_SIMULATE_LOCK, encoding="utf-8")
        (root / "configs" / "x.yaml").write_text("a: 1\n", encoding="utf-8")
        record = tmp_path / "record.txt"
        # Otra etapa con un hash distinto en la copia no debe contar: solo simulate.
        same = _SIMULATE_LOCK.replace("0000.dir", "9999.dir")
        different = _SIMULATE_LOCK.replace("aaaa.dir", "ffff.dir")
        body = r"""
function git { $global:LASTEXITCODE = 0; "dvc.yaml"; "dvc.lock"; ".dvc/config"; "configs/x.yaml" }
function dvc {
    $global:LASTEXITCODE = 0
    if ($args[0] -eq "repro") {
        $line = "cwd=" + (Get-Location).Path + " lock=" + (Test-Path "dvc.lock") +
            " config=" + (Test-Path "configs/x.yaml") + " args=" + ($args -join " ")
        Add-Content -LiteralPath "__RECORD__" -Value $line
        if ($env:FAKE_REPRO -eq "fail") { $global:LASTEXITCODE = 1; "ERROR: failed to reproduce" }
        elseif ($env:FAKE_REPRO -eq "same") { Set-Content -LiteralPath "dvc.lock" $env:SAME_LOCK }
        else { Set-Content -LiteralPath "dvc.lock" -Value $env:DIFFERENT_LOCK }
    }
}
foreach ($mode in @("same", "different", "fail")) {
    $env:FAKE_REPRO = $mode
    $s = Invoke-FromScratch -Root "__ROOT__" -Stage $SimulateStage
    Write-Output ("CASE=$mode|" + $s.Status + "|" + $s.Detail)
}
""".replace("__RECORD__", str(record)).replace("__ROOT__", str(root))
        env_body = (
            f"$env:SAME_LOCK = @'\n{same}'@\n$env:DIFFERENT_LOCK = @'\n{different}'@\n" + body
        )
        cases = _cases(_run(tmp_path, env_body))

        assert cases["same"] == ("matched", "2 salidas de simulate con el mismo hash.")
        assert cases["different"][0] == "mismatch"
        difference = cases["different"][1]
        assert "simulate:data/raw/simulation (esperado aaaa.dir, obtenido ffff.dir)" in difference
        assert "other" not in difference
        assert cases["fail"] == ("failed", "ERROR: failed to reproduce")
        runs = _read(record).splitlines()
        assert len(runs) == 3
        for line in runs:
            # La copia no lleva el dvc.lock del repositorio y si los ficheros seguidos.
            assert "lock=False config=True args=repro --no-run-cache simulate" in line
            copy_dir = Path(line.split("cwd=", 1)[1].split(" lock=", 1)[0])
            assert copy_dir.name.startswith("rg-")
            assert not copy_dir.exists(), "la copia temporal debe borrarse"
        # El repositorio no se toca.
        assert _read(root / "dvc.lock") == _SIMULATE_LOCK
        assert sorted(p.name for p in root.iterdir()) == ["configs", "dvc.lock", "dvc.yaml"]

    def test_from_scratch_without_hashes_to_compare_is_unavailable(self, tmp_path: Path) -> None:
        """Sin dvc.lock o sin salidas de la etapa no hay nada que reproducir."""
        empty = tmp_path / "empty"
        empty.mkdir()
        no_stage = tmp_path / "no_stage"
        no_stage.mkdir()
        lock = "schema: '2.0'\nstages:\n  other:\n    cmd: x\n"
        (no_stage / "dvc.lock").write_text(lock, encoding="utf-8")
        body = (
            f'$a = Invoke-FromScratch -Root "{empty}" -Stage $SimulateStage\n'
            'Write-Output ("CASE=no_lock|" + $a.Status + "|" + $a.Detail)\n'
            f'$b = Invoke-FromScratch -Root "{no_stage}" -Stage $SimulateStage\n'
            'Write-Output ("CASE=no_stage|" + $b.Status + "|" + $b.Detail)\n'
        )
        cases = _cases(_run(tmp_path, body))
        assert cases["no_lock"][0] == "unavailable"
        assert cases["no_stage"] == ("unavailable", "dvc.lock no registra salidas de simulate.")


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------


@requires_pwsh
class TestDeliverables:
    """Regresion de Fase 2 y documentos: los mismos tres estados, sin bloquear."""

    @staticmethod
    def _phase2_report(path: Path, states: dict[str, str], commit: str = "a8bb24f") -> Path:
        """Write a report with the shape New-ReportDocument of Verify-Phase2.ps1 produces."""
        criteria = [{"id": cid, "state": state} for cid, state in states.items()]
        path.write_text(json.dumps({"commit": commit, "criteria": criteria}), encoding="utf-8")
        return path

    def test_phase2_regression(self, tmp_path: Path) -> None:
        """2 y 4 cumplidos, uno fallado, uno sin medir, otro commit y sin informe."""
        local = {"1": NOT_MEASURED, "2": MET, "3": NOT_MEASURED, "4": MET}
        reports = {
            "guarded": self._phase2_report(tmp_path / "a.json", local),
            "dvc_failed": self._phase2_report(tmp_path / "b.json", {**local, "4": FAILED}),
            "stuck_skipped": self._phase2_report(tmp_path / "c.json", {**local, "2": NOT_MEASURED}),
            "other_commit": self._phase2_report(tmp_path / "d.json", local, commit="d04b1f6"),
            "absent": tmp_path / "absent.json",
        }
        body = "\n".join(
            f'$v = Resolve-Phase2RegressionVerdict -Report (Read-JsonFile -Path "{path}") '
            '-HeadCommit "a8bb24f-dirty"\n'
            f'Case "{name}" $v\n'
            f'Write-Output ("WARN={name}|" + $v.warnings.Count + "|" + $v.blocking)'
            for name, path in reports.items()
        )
        output = _run(tmp_path, body)
        cases = _cases(output)
        warnings = _cases(output, "WARN")

        assert cases["guarded"][0] == MET
        assert cases["dvc_failed"][0] == FAILED
        assert cases["stuck_skipped"][0] == NOT_MEASURED
        assert cases["other_commit"][0] == MET
        assert cases["absent"][0] == NOT_MEASURED
        assert warnings["guarded"] == ("0", "False")
        assert warnings["other_commit"][0] == "1"

    def test_docs(self, tmp_path: Path) -> None:
        """Presentes, ausentes y vacios."""
        body = r"""
Case "present" (Resolve-DocsVerdict -Sizes ([ordered]@{ "a.md" = 10; "b.md" = 3 }))
Case "absent" (Resolve-DocsVerdict -Sizes ([ordered]@{ "a.md" = 10; "b.md" = -1 }))
Case "empty" (Resolve-DocsVerdict -Sizes ([ordered]@{ "a.md" = 0; "b.md" = 3 }))
"""
        cases = _cases(_run(tmp_path, body))
        assert cases["present"] == (MET, "Presentes: a.md, b.md.")
        assert cases["absent"] == (FAILED, "Documentos ausentes: b.md.")
        assert cases["empty"] == (FAILED, "Documentos vacios: a.md.")


# ---------------------------------------------------------------------------
# Exit codes and the whole script
# ---------------------------------------------------------------------------


def _project(tmp_path: Path, *, suite_skips: bool = False) -> Path:
    """Create a synthetic repository with every artifact Verify-Phase3.ps1 reads."""
    root = tmp_path / "project"
    _dataset(root, {"sim_run_000": _run_frame(6, 2), "sim_run_001": _run_frame(6, 0)})
    _write_metrics(root / "metrics" / "simulation_metrics.json")
    (root / "dvc.yaml").write_text(_SIMULATE_YAML, encoding="utf-8")
    (root / "dvc.lock").write_text(_SIMULATE_LOCK, encoding="utf-8")
    for relative in ("Docs/physics_constraints.md", "Docs/ReactorGuard_Fase3_Simulacion.md"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("# doc\n", encoding="utf-8")
    suite = root / "tests" / "safety"
    suite.mkdir(parents=True)
    header = "import pytest\n\npytestmark = pytest.mark.skip(reason='x')\n\n" if suite_skips else ""
    tests = "\n\n".join(f"def {name}():\n    assert True" for name in _COMPLIANCE_TESTS)
    (suite / "test_safety_constraints.py").write_text(header + tests + "\n", encoding="utf-8")
    return root


_RUN_ALL = (
    "$VolumeThreshold = 10\n"
    + _FAKE_DVC_CLEAN
    + '$code = Invoke-VerifyPhase3\nWrite-Output ("EXIT=" + $code)'
)


@requires_pwsh
class TestExitCodes:
    """0 todos cumplidos, 1 algun fallado, 2 ninguno fallado pero alguno sin medir."""

    @staticmethod
    def _report(root: Path) -> dict[str, Any]:
        """Return the verification report the script wrote."""
        document: dict[str, Any] = json.loads(
            _read(root / "tests" / "results" / "phase3_verification.json")
        )
        return document

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
        """Dataset, metricas, suite y DVC coherentes: los cuatro criterios cierran."""
        root = _project(tmp_path)
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["0"]
        report = self._report(root)
        assert self._states(report) == {"1": MET, "2": MET, "3": MET, "4": MET}
        assert report["exit_code"] == 0
        assert report["phase"] == 3
        deliverables = {item["id"]: item for item in report["deliverables"]}
        assert deliverables["docs"]["state"] == MET
        # Sin informe de Fase 2 el entregable queda sin medir y no cambia la salida.
        assert deliverables["fase2"]["state"] == NOT_MEASURED
        assert not any(item["blocking"] for item in deliverables.values())

    def test_a_failed_criterion_exits_one(self, tmp_path: Path) -> None:
        """Un parquet manipulado es FALLADO aunque lo demas cierre."""
        root = _project(tmp_path)
        target = root / "data/raw/simulation/fault_type=void_formation/seed=42/sim_run_000.parquet"
        _run_frame(6, 3).to_parquet(target, index=False)
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["1"]
        assert self._states(self._report(root)) == {"1": MET, "2": MET, "3": FAILED, "4": MET}

    def test_skipped_suite_is_not_a_pass(self, tmp_path: Path) -> None:
        """pytest sale con 0 y no se ha medido nada: criterio 2 sin medir, salida 2."""
        root = _project(tmp_path, suite_skips=True)
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["2"]
        report = self._report(root)
        assert self._states(report)["2"] == NOT_MEASURED
        assert "SALTADOS" in report["criteria"][1]["detail"]

    def test_missing_dataset_is_not_a_failure(self, tmp_path: Path) -> None:
        """Sin data/raw/simulation el volumen queda sin medir y el lector ni se ejecuta."""
        root = _project(tmp_path)
        shutil.rmtree(root / "data")
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["2"]
        states = self._states(self._report(root))
        assert states == {"1": MET, "2": MET, "3": NOT_MEASURED, "4": MET}

    def test_a_crash_while_measuring_one_criterion_does_not_stop_the_others(
        self, tmp_path: Path
    ) -> None:
        """Invoke-Guarded convierte la excepcion de un criterio en NO MEDIDO."""
        root = _project(tmp_path)
        (root / "dvc.yaml").unlink()
        output = _run(tmp_path, _RUN_ALL, root=root)

        assert output["EXIT"] == ["2"]
        report = self._report(root)
        assert self._states(report) == {"1": MET, "2": MET, "3": MET, "4": NOT_MEASURED}
        assert "Error al medir" in report["criteria"][3]["detail"]

    def test_an_internal_error_exits_three(self, tmp_path: Path) -> None:
        """Un fallo del propio verificador no se disfraza de veredicto sobre la fase."""
        root = _project(tmp_path)
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
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=env,
        )
        assert result.returncode == 3, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "ERROR del verificador" in result.stdout
