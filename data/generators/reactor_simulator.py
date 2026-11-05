"""Simulador de reactor: orquesta runs de operacion normal sobre el modelo ODE.

Reescritura de F3-M3 (T5.3). Sustituye al esqueleto anterior, que construia el
esquema plano previo al commit f130d27 y lanzaba ValidationError contra el
contrato canonico. Ahora:

    - el motor es `data/simulation/thermal_model.ReactorModel` (scipy LSODA), no
      OpenMC (D4); `openmc_seed` se conserva como semilla maestra (D12);
    - la salida es el frame largo del dataset, con las 11 columnas de
      `readings_from_frame` mas la verdad-terreno (`dataset_builder`);
    - todo lo configurable llega por `SimulationParams` (seccion simulation: de
      params.yaml), nada vive como constante de modulo.

`plans` es el dataset de operacion normal (F3-M3). Los fallos de proceso (F3-M4)
se piden con `process_plan` y se integran con el mismo `simulate`: perturban las
entradas del modelo, no la medicion. Los de instrumento (F3-M5) tampoco tocan la
planta: `generate_run` les aplica sus episodios a la serie medida, despues del
ruido. `planned` es el dataset con fallos que fija el planificador por
presupuesto (D11); es el que escribe el stage `simulate` (F3-M6).

`build_dataset` genera el dataset planificado entero: escribe un parquet por run
(TDD 4.4), `manifest.json` y las metricas. Los runs son independientes (cada uno
cuelga de su propia semilla), asi que se pueden repartir entre procesos: el
resultado es el mismo byte a byte con uno o con varios trabajadores. El coste lo
domina el vacio: F3-M5 midio ~840 s por run en Windows y F3-M6, 7 min de CPU el
dataset entero en Linux (`Docs/physics_constraints.md`, 14.1).

Este modulo NO importa nada de `data/validation` (la validacion offline vive en
los tests). El punto de entrada del stage `simulate` es `main`.

Uso:
    python data/generators/reactor_simulator.py
    python data/generators/reactor_simulator.py --params params.yaml --workers 4
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from data.schemas.sensor_spans import SensorSpan, load_sensor_spans
from data.simulation.dataset_builder import (
    RunOutput,
    RunSimulation,
    SimulatedDataset,
    build_dataset,
    build_run_frame,
    run_parquet_path,
    simulate_run,
    write_dataset,
    write_metrics,
    write_run_output,
    write_run_parquet,
)
from data.simulation.physics_constraints import ConstraintLimits
from data.simulation.plant_params import load_neutronics, load_plant_params
from data.simulation.process_faults import (
    FaultTiers,
    build_fault_inputs,
    parse_fault_tiers,
    process_run_plan,
)
from data.simulation.scenario import (
    FaultAxis,
    FaultLevel,
    LoadProfile,
    RunPlan,
    ScenarioKind,
    normal_run_plans,
)
from data.simulation.scenario_planner import (
    PlannedDataset,
    ScenarioConfig,
    plan_dataset,
    planner_rng,
)
from data.simulation.sensor_catalog import SensorCatalog, load_sensor_catalog
from data.simulation.simulation_params import (
    DEFAULT_PARAMS_PATH,
    SimulationParams,
    load_simulation_params,
)
from data.simulation.thermal_model import ModelInputs, ReactorModel
from data.simulation.water_properties import load_water_table

_LOG = logging.getLogger(__name__)


class ReactorSimulator:
    """Generates labelled sensor streams from the thermal model, run by run.

    Args:
        params: Resolved simulation parameters.
        model: Thermal model.
        catalog: Sensor catalog.
        spans: Calibrated span per sensor tag (scales readings to ADC counts).

    Raises:
        KeyError: If a catalog sensor has no span.
    """

    def __init__(
        self,
        params: SimulationParams,
        model: ReactorModel,
        catalog: SensorCatalog,
        spans: dict[str, SensorSpan],
    ) -> None:
        missing = [tag for tag in catalog.tags if tag not in spans]
        if missing:
            raise KeyError(
                f"Sensor span table is missing {len(missing)} simulation tags: "
                f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}. "
                "Regenerate it with data/generators/derive_sim_spans.py."
            )
        self.params = params
        self.model = model
        self.catalog = catalog
        self.spans = spans
        self.profile = LoadProfile.from_params(params)
        self._fault_tiers: FaultTiers | None = None
        self._scenario_config: ScenarioConfig | None = None
        self._limits: ConstraintLimits | None = None

    @classmethod
    def from_params(cls, params: SimulationParams) -> ReactorSimulator:
        """Build the simulator from the files the simulation: section points to.

        Args:
            params: Resolved simulation parameters.

        Returns:
            A ReactorSimulator with model, catalog and spans loaded.

        Raises:
            FileNotFoundError: If a configuration file does not exist.
            KeyError: If a required key or a sensor span is missing.
            ValueError: If a configuration file is not valid.
        """
        plant = load_plant_params(params.plant_parameters_path)
        neutronics = load_neutronics(params.neutronics_path)
        table = load_water_table(params.water_table_path)
        model = ReactorModel(plant, neutronics, table)
        catalog = load_sensor_catalog(params.sensors_path)
        spans = load_sensor_spans(params.spans_path)
        _LOG.info(
            "Simulator ready: %d sensors, %d runs of %d steps at %.0f s.",
            len(catalog.sensors),
            params.n_runs,
            params.run_length_steps,
            params.dt_seconds,
        )
        return cls(params, model, catalog, spans)

    @property
    def plans(self) -> list[RunPlan]:
        """Return the plan of every normal-operation run of the dataset."""
        return normal_run_plans(self.params)

    @property
    def fault_tiers(self) -> FaultTiers:
        """Return the process-fault levels of plant_parameters.yaml (parsed once).

        Raises:
            KeyError: If a fault, level or required key is missing.
            ValueError: If a magnitude is outside its validity range.
        """
        if self._fault_tiers is None:
            self._fault_tiers = parse_fault_tiers(self.model.plant.fault_tiers)
        return self._fault_tiers

    @property
    def scenario_config(self) -> ScenarioConfig:
        """Return the fault levels and planner caps of plant_parameters.yaml (parsed once).

        Raises:
            KeyError: If a fault, level or required key is missing.
            ValueError: If a magnitude is out of range.
        """
        if self._scenario_config is None:
            self._scenario_config = ScenarioConfig.from_plant(self.model.plant)
        return self._scenario_config

    @property
    def limits(self) -> ConstraintLimits:
        """Return the rule thresholds and the conservation tolerance (parsed once)."""
        if self._limits is None:
            self._limits = ConstraintLimits.from_plant(self.model.plant)
        return self._limits

    def planned(
        self,
        *,
        fault_injection_rate: float | None = None,
        severity_weights: dict[str, float] | None = None,
    ) -> PlannedDataset:
        """Plan the dataset with faults under the D11 label budget.

        Args:
            fault_injection_rate: Sustituye al de `params` (estudios de capacidad).
            severity_weights: Sustituye a los pesos de `params`.

        Returns:
            The PlannedDataset (plans, budget and per-type decisions).

        Raises:
            ValueError: If the quotas do not fit under the density caps; the
                message carries the numbers (D11: the planner does not force).
        """
        return plan_dataset(
            self.params,
            self.catalog,
            planner_rng(self.params),
            self.scenario_config,
            fault_injection_rate=fault_injection_rate,
            severity_weights=severity_weights,
        )

    def process_plan(
        self,
        scenario: ScenarioKind,
        level: FaultLevel,
        *,
        run_index: int = 0,
        seed: int | None = None,
        injected_step: int | None = None,
        hold_steps: int | None = None,
        recovery_steps: int | None = None,
        persistent: bool = False,
    ) -> RunPlan:
        """Return the plan of one process-fault run.

        Args:
            scenario: Uno de los tres escenarios de proceso.
            level: Nivel de magnitud.
            run_index: Posicion del run (instantes y semilla).
            seed: Semilla maestra; por defecto `params.openmc_seed`.
            injected_step: Primera muestra del fallo; por defecto el fin del
                calentamiento.
            hold_steps: Pasos a magnitud plena; por defecto los de `fault_tiers`.
            recovery_steps: Pasos de recuperacion; por defecto los de `fault_tiers`.
            persistent: Fallo hasta el final del run, sin recuperacion.

        Returns:
            The RunPlan.

        Raises:
            KeyError: If the scenario is not a process fault.
            ValueError: If the fault window does not fit in the run.
        """
        return process_run_plan(
            self.params,
            self.fault_tiers.for_kind(scenario),
            level,
            run_index=run_index,
            seed=seed,
            injected_step=injected_step,
            hold_steps=hold_steps,
            recovery_steps=recovery_steps,
            persistent=persistent,
        )

    def simulate(self, plan: RunPlan) -> RunSimulation:
        """Integrate one run (load profile from the run's own generator).

        Un run de operacion normal solo lleva el programa de carga; un run con
        fallo de proceso le superpone las entradas del fallo (rampa de
        reactividad, demanda de bomba o fuga) segun su envolvente.

        Args:
            plan: Plan del run.

        Returns:
            The RunSimulation.
        """
        dt_s = self.params.dt_seconds
        load_rng, _, _ = plan.rngs()
        loads = self.profile.sample(load_rng, plan.n_steps, dt_s)
        spec = plan.process_fault
        if spec is None:
            inputs = [ModelInputs(load=float(load)) for load in loads]
            return simulate_run(self.model, self.catalog, plan, inputs, dt_s)
        fault = build_fault_inputs(
            self.model,
            self.fault_tiers.for_kind(plan.scenario),
            spec.level,
            spec,
            loads.tolist(),
            dt_s,
        )
        return simulate_run(
            self.model,
            self.catalog,
            plan,
            fault.sample,
            dt_s,
            fine_inputs=fault.fine,
            substeps=fault.substeps,
        )

    def generate_run(self, plan: RunPlan) -> pd.DataFrame:
        """Return the long frame of one run (normal, process or instrument fault).

        Args:
            plan: Plan del run.

        Returns:
            The frame with the columns of `dataset_builder.LONG_COLUMNS`.
        """
        return self.simulate_frame(plan)[1]

    def simulate_frame(self, plan: RunPlan) -> tuple[RunSimulation, pd.DataFrame]:
        """Integrate one run and return it with its long frame.

        Args:
            plan: Plan del run.

        Returns:
            (RunSimulation, frame with the columns of `dataset_builder.LONG_COLUMNS`).
        """
        sim = self.simulate(plan)
        _, noise_rng, _ = plan.rngs()
        frame = build_run_frame(sim, self.catalog, self.spans, self.params.plant_id, noise_rng)
        return sim, frame

    def write_run(self, plan: RunPlan, output_dir: str | Path) -> RunOutput:
        """Generate one run, write its parquet and summarise it.

        Args:
            plan: Plan del run.
            output_dir: Raiz del dataset.

        Returns:
            The RunOutput (manifest entry and coverage of the run).
        """
        sim, frame = self.simulate_frame(plan)
        return write_run_output(
            sim,
            frame,
            output_dir,
            self.params.openmc_seed,
            catalog=self.catalog,
            table=self.model.table,
            limits=self.limits,
        )

    def build_dataset(
        self, output_dir: str | Path | None = None, *, workers: int = 1
    ) -> SimulatedDataset:
        """Generate the planned dataset (D11) and summarise it; writes no JSON.

        Args:
            output_dir: Raiz del dataset; por defecto `params.output_dir`.
            workers: Procesos que integran runs en paralelo; 1 = en este proceso,
                0 = uno por CPU (sin pasar del numero de runs). El resultado no
                depende de este valor.

        Returns:
            The SimulatedDataset (runs in run order).

        Raises:
            ValueError: If the planner refuses the budget (D11) or workers < 0.
        """
        if workers < 0:
            raise ValueError(f"workers must be >= 0, got {workers}.")
        root = Path(output_dir) if output_dir is not None else self.params.output_dir
        plans = self.planned().plans
        n_workers = resolve_workers(workers, len(plans))
        _LOG.info("Generating %d runs with %d worker(s) under %s", len(plans), n_workers, root)
        if n_workers == 1:
            outputs = [self.write_run(plan, root) for plan in plans]
        else:
            outputs = _run_in_pool(self.params, plans, root, n_workers)
        settings = self.scenario_config.planner
        return build_dataset(
            outputs,
            self.params,
            n_sensors=len(self.catalog.sensors),
            k_min_readings=settings.k_min_readings,
            n_min_episodes=settings.n_min_episodes,
            coverage_gate=settings.coverage_gate,
            conservation_tolerance=self.limits.conservation_tolerance,
        )

    def iter_runs(self) -> Iterator[tuple[RunPlan, pd.DataFrame]]:
        """Yield every run with its frame, in run order.

        Yields:
            (plan, long frame) per run.
        """
        for plan in self.plans:
            yield plan, self.generate_run(plan)

    def generate(self) -> pd.DataFrame:
        """Return the long frame of the whole dataset.

        Returns:
            All runs concatenated in run order, with a fresh RangeIndex.
        """
        frames = [frame for _, frame in self.iter_runs()]
        return pd.concat(frames, ignore_index=True)

    def write(self, output_dir: str | Path | None = None) -> list[Path]:
        """Write every run as Parquet under the TDD 4.4 layout.

        Args:
            output_dir: Destination root; defaults to `params.output_dir`.

        Returns:
            The paths written, in run order.
        """
        root = Path(output_dir) if output_dir is not None else self.params.output_dir
        written: list[Path] = []
        for plan, frame in self.iter_runs():
            path = run_parquet_path(root, plan.scenario, self.params.openmc_seed, plan.run_id)
            written.append(write_run_parquet(frame, path))
        return written


# ----------------------------------------------------------------------
# Ejecucion en paralelo (F3-M6)
# ----------------------------------------------------------------------

_WORKER_SIMULATOR: ReactorSimulator | None = None
"""Simulador de cada proceso trabajador (lo crea `_init_worker`)."""

_COST_ORDER: dict[FaultAxis, int] = {FaultAxis.PROCESS: 1, FaultAxis.INSTRUMENT: 2}
"""Orden de envio por coste: vacio, resto de proceso, instrumento y normal."""


def resolve_workers(workers: int, n_runs: int) -> int:
    """Return the worker processes to use: 0 means one per CPU; never more than runs.

    Args:
        workers: Valor pedido (0 = automatico).
        n_runs: Runs a generar.

    Returns:
        A count in [1, max(1, n_runs)].
    """
    wanted = (os.cpu_count() or 1) if workers == 0 else workers
    return max(1, min(wanted, n_runs))


def _cost_key(plan: RunPlan) -> tuple[int, str]:
    """Return the submission key: the slow runs first, so they overlap the rest."""
    if plan.scenario is ScenarioKind.VOID_FORMATION:
        return 0, plan.run_id
    return _COST_ORDER.get(plan.scenario.axis, 3), plan.run_id


def _init_worker(params: SimulationParams) -> None:
    """Build the simulator of a worker process once."""
    global _WORKER_SIMULATOR
    _WORKER_SIMULATOR = ReactorSimulator.from_params(params)


def _write_run_in_worker(plan: RunPlan, output_dir: Path) -> RunOutput:
    """Generate and write one run in a worker process."""
    if _WORKER_SIMULATOR is None:  # pragma: no cover - set by the pool initializer
        raise RuntimeError("worker simulator is not initialised.")
    return _WORKER_SIMULATOR.write_run(plan, output_dir)


def _run_in_pool(
    params: SimulationParams, plans: Sequence[RunPlan], root: Path, workers: int
) -> list[RunOutput]:
    """Generate the runs in `workers` spawned processes.

    `spawn` en todas las plataformas: es el unico metodo de Windows y asi Linux
    ejecuta lo mismo. Cada run depende solo de su plan y de su semilla, de modo
    que el orden de terminacion no cambia nada; los resultados vuelven al orden
    de run en `build_dataset`.
    """
    ordered = sorted(plans, key=_cost_key)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(params,)
    ) as pool:
        futures = [pool.submit(_write_run_in_worker, plan, root) for plan in ordered]
        outputs = []
        for future in futures:
            output = future.result()
            _LOG.info("Run %s (%s) written.", output.record.run_id, output.record.scenario)
            outputs.append(output)
    return outputs


def main(params_path: str | Path = DEFAULT_PARAMS_PATH, *, workers: int = 0) -> SimulatedDataset:
    """Run the `simulate` stage: dataset, manifest and metrics.

    Args:
        params_path: Path to the params file holding the simulation: section.
        workers: Worker processes (0 = one per CPU, 1 = in this process).

    Returns:
        The SimulatedDataset written.

    Raises:
        FileNotFoundError: If params_path or a referenced file does not exist.
        KeyError: If a required key or a sensor span is missing.
        ValueError: If a value is out of range or the planner refuses the budget.
    """
    params = load_simulation_params(params_path)
    simulator = ReactorSimulator.from_params(params)
    dataset = simulator.build_dataset(workers=workers)
    write_dataset(dataset, params.output_dir)
    write_metrics(dataset, params.metrics_path)
    _LOG.info(
        "Wrote %d runs (%d steps x %d sensors) under %s",
        len(dataset.records),
        params.total_steps,
        len(simulator.catalog.sensors),
        params.output_dir,
    )
    return dataset


def cli(argv: list[str] | None = None) -> SimulatedDataset:
    """Parse the command line and run `main`.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        The SimulatedDataset written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes (0 = one per CPU, 1 = no pool); the output does not depend on it",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return main(args.params, workers=args.workers)


if __name__ == "__main__":
    # Se llama a `cli` del modulo importado, no del script: los procesos `spawn`
    # deserializan las funciones del pool por su modulo, y asi resuelven a
    # `data.generators.reactor_simulator` y no a `__main__`.
    from data.generators.reactor_simulator import cli as _module_cli

    _module_cli()
