"""Entry point for the prepare_sim DVC stage (F4-M1).

Lee el dataset del stage `simulate` (`simulation.output_dir`), lo reparte por run
(D4-7) y escribe en `pinn.data_dir` un parquet ancho por particion y la asignacion
de runs, mas las metricas en `pinn.metrics_path`. La logica vive en
`ml/training/sim_dataset.py`; este modulo solo resuelve parametros, orquesta y
reporta, como `data/generators/adapt_tep.py`.

`featurize`, `split` y el TEP no se tocan (D4-8).
"""

from __future__ import annotations

import logging
from pathlib import Path

from data.simulation.plant_params import load_plant_params
from data.simulation.scenario_planner import parse_planner_settings
from data.simulation.sensor_catalog import load_sensor_catalog
from data.simulation.simulation_params import load_simulation_params
from ml.training.pinn_params import DEFAULT_PARAMS_PATH, load_pinn_params
from ml.training.sim_dataset import (
    SPLITS,
    PreparedDataset,
    prepare_dataset,
    write_metrics,
    write_prepared,
)

_LOG = logging.getLogger(__name__)


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> PreparedDataset:
    """Run the prepare_sim stage end to end.

    Args:
        params_path: Path to the params file holding the pinn: and simulation:
            sections.

    Returns:
        The PreparedDataset that was written.

    Raises:
        FileNotFoundError: If params_path, a configuration file, the manifest or
            a run parquet does not exist.
        KeyError: If a required section or key is missing.
        ValueError: If a value, the manifest, a digest or a run is invalid.
    """
    pinn = load_pinn_params(params_path)
    simulation = load_simulation_params(params_path)
    catalog = load_sensor_catalog(simulation.sensors_path)
    planner = parse_planner_settings(
        load_plant_params(simulation.plant_parameters_path).scenario_planner
    )
    _LOG.info("Preparing PINN data from %s", simulation.output_dir.as_posix())

    prepared = prepare_dataset(
        simulation.output_dir,
        catalog,
        val_per_scenario=pinn.val_runs_per_scenario,
        test_per_scenario=pinn.test_runs_per_scenario,
        k_min_readings=planner.k_min_readings,
        n_min_episodes=planner.n_min_episodes,
    )
    write_prepared(prepared, pinn.data_dir)
    write_metrics(prepared, pinn.metrics_path)

    for split in SPLITS:
        summary = prepared.summary(split)
        _LOG.info(
            "  %-5s: %2d runs, %6d instants, %5d CRITICAL, %6d trainable pairs, grid %.1f %%",
            split,
            summary.runs,
            summary.instants,
            summary.critical_instants,
            summary.trainable_pairs,
            100.0 * summary.coverage_fraction,
        )
    return prepared


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
