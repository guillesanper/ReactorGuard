"""Derive the calibrated sensor span table from normal-operation TEP data.

Genera configs/sensor_spans.yaml a partir de d00.dat EXCLUSIVAMENTE. La eleccion
es deliberada: d00 es el unico fichero de operacion normal del dataset, y derivar
los spans del pool completo filtraria informacion de los ficheros de fallo hacia
la escala de medida. El detector de fuera-de-rango consume ese mismo fichero, asi
que un span ajustado sobre datos de fallo le impediria por construccion senalar
justo los valores que debe senalar.

Este modulo NO es un stage de dvc.yaml. Se ejecuta a mano, el fichero resultante
se revisa y se commitea, y los stages que lo consumen lo declaran como `deps`.
Un span es una decision de ingenieria de instrumentacion, no un artefacto
regenerable en cada `dvc repro`.

Uso:
    python data/generators/derive_sensor_spans.py
    python data/generators/derive_sensor_spans.py --margin 0.25 --output otro.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from data.generators.tep_adapter import SENSOR_TYPE_MAP, UNIT_MAP, sensor_id
from data.generators.tep_loader import N_COLUMNS, load_dat_file
from data.schemas.sensor_spans import DEFAULT_SPANS_PATH

_LOG = logging.getLogger(__name__)

NORMAL_OPERATION_FILE = "d00.dat"

DEFAULT_ALARM_MARGIN = 0.20
"""Margen del sobre de operacion normal, a cada lado del rango observado en d00.

Un umbral fijado en el minimo y el maximo exactos de d00 marcaria como fallo la
primera excursion benigna que las 500 muestras normales no llegaron a visitar.
El 20% cubre esos transitorios y aun asi produce 0,000% de falsos positivos
sobre d00 en la medicion de las 550.160 lecturas.
"""

DEFAULT_SPAN_MARGIN = 2.00
"""Margen del span calibrado del transmisor, mucho mas ancho que el de alarma.

El span existe para escalar a cuentas ADC, y todo lo que caiga fuera se recorta
irreversiblemente. Medido sobre el dataset completo, el margen del sobre de
alarma (20%) recortaria el 12,76% de las lecturas; con el 200% baja al 4,06%, y
ese resto son excursiones extremas de los ficheros de fallo, donde un transmisor
real tambien saturaria. Se deriva igualmente solo de d00: ensanchar el span con
los ficheros de fallo seria hacer que la escala dependa de lo que se pretende
detectar.
"""

MIN_ABSOLUTE_WIDTH = 1.0
"""Ancho minimo cuando el canal es constante en operacion normal.

Algunas variables manipuladas del TEP no se mueven en d00. Un span de ancho cero
es indivisible y romperia tanto el escalado a cuentas como la comprobacion de
rango, asi que se les asigna este ancho minimo alrededor del valor observado.
"""


def derive_spans(
    dat_path: str | Path,
    alarm_margin: float = DEFAULT_ALARM_MARGIN,
    span_margin: float = DEFAULT_SPAN_MARGIN,
) -> dict[str, dict[str, Any]]:
    """Derive a span entry per sensor from a normal-operation TEP file.

    Args:
        dat_path: Path to d00.dat, the normal-operation file.
        alarm_margin: Fractional margin of the normal-operation envelope,
            applied to each side of the observed range. 0.2 widens an observed
            range of [10, 20] to an envelope of [8, 22].
        span_margin: Fractional margin of the calibrated transmitter span. Must
            be at least alarm_margin, since a transmitter cannot alarm on a
            value it cannot represent.

    Returns:
        Mapping of sensor_id to a {min, max, alarm_min, alarm_max, unit}
        mapping, ordered by column index.

    Raises:
        FileNotFoundError: If dat_path does not exist.
        ValueError: If a margin is negative, if span_margin < alarm_margin, or
            if the file does not hold 52 variables in either orientation.
    """
    if alarm_margin < 0.0 or span_margin < 0.0:
        raise ValueError(
            f"Margins must be non-negative, got alarm_margin={alarm_margin}, "
            f"span_margin={span_margin}."
        )
    if span_margin < alarm_margin:
        raise ValueError(
            f"span_margin ({span_margin}) must be at least alarm_margin "
            f"({alarm_margin}): the alarm envelope lives inside the span."
        )

    frame = load_dat_file(dat_path)
    _LOG.info(
        "Deriving spans from %s (%d samples x %d variables).",
        Path(dat_path).name,
        frame.shape[0],
        frame.shape[1],
    )

    spans: dict[str, dict[str, Any]] = {}
    for col_idx in range(N_COLUMNS):
        observed = frame.iloc[:, col_idx]
        low = float(observed.min())
        high = float(observed.max())
        observed_width = high - low

        if observed_width <= 0.0:
            _LOG.warning(
                "%s is constant at %.6f in normal operation; assigning the "
                "minimum span width of %.1f.",
                sensor_id(col_idx),
                low,
                MIN_ABSOLUTE_WIDTH,
            )
            alarm_pad = MIN_ABSOLUTE_WIDTH / 2.0
            span_pad = MIN_ABSOLUTE_WIDTH / 2.0
        else:
            alarm_pad = observed_width * alarm_margin
            span_pad = observed_width * span_margin

        spans[sensor_id(col_idx)] = {
            "min": round(low - span_pad, 6),
            "max": round(high + span_pad, 6),
            "alarm_min": round(low - alarm_pad, 6),
            "alarm_max": round(high + alarm_pad, 6),
            "unit": UNIT_MAP[SENSOR_TYPE_MAP[col_idx]].value,
        }

    return spans


def render_yaml(
    spans: dict[str, dict[str, Any]],
    source_file: str,
    alarm_margin: float,
    span_margin: float,
) -> str:
    """Render the span table as a commented YAML document.

    Args:
        spans: Mapping produced by derive_spans.
        source_file: Name of the file the spans were derived from, for the header.
        alarm_margin: Alarm margin used, recorded for traceability.
        span_margin: Span margin used, recorded for traceability.

    Returns:
        The YAML document as a string, ready to be written to disk.
    """
    header = (
        "# ==========================================================================\n"
        "# ReactorGuard - Spans calibrados por transmisor\n"
        "#\n"
        "# GENERADO por data/generators/derive_sensor_spans.py. Se revisa a mano y se\n"
        "# commitea; no es un artefacto de `dvc repro`.\n"
        "#\n"
        f"#   Derivado de : {source_file} (operacion normal, exclusivamente)\n"
        f"#   Margen span : {span_margin:.0%} a cada lado del rango observado\n"
        f"#   Margen alarma: {alarm_margin:.0%} a cada lado del rango observado\n"
        "#\n"
        "# Dos rangos por sensor, porque sus consumidores necesitan anchos distintos:\n"
        "#\n"
        "#   [min, max]              span calibrado del transmisor. Escalado a cuentas\n"
        "#                           ADC en data/generators/tep_adapter.py. Todo lo que\n"
        "#                           cae fuera se recorta de forma irreversible, asi que\n"
        "#                           es deliberadamente ancho.\n"
        "#   [alarm_min, alarm_max]  sobre de operacion normal. Umbral del detector de\n"
        "#                           fuera-de-rango en data/validation/sensor_validator.py.\n"
        "#\n"
        "# Ambos se leen a traves de data/schemas/sensor_spans.py, de modo que no\n"
        "# existan dos respuestas a que es un valor valido para un canal dado.\n"
        "# ==========================================================================\n\n"
    )
    body = yaml.safe_dump(
        {"sensors": spans}, sort_keys=False, default_flow_style=False, allow_unicode=False
    )
    return header + body


def main(argv: list[str] | None = None) -> Path:
    """Derive the span table and write it to disk.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        The path the span table was written to.

    Raises:
        FileNotFoundError: If the normal-operation file does not exist.
        ValueError: If margin_fraction is negative.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data/raw/tep",
        help="Directory holding the raw TEP .dat files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SPANS_PATH),
        help="Destination path for the generated span table.",
    )
    parser.add_argument(
        "--alarm-margin",
        type=float,
        default=DEFAULT_ALARM_MARGIN,
        help="Fractional margin of the normal-operation envelope.",
    )
    parser.add_argument(
        "--span-margin",
        type=float,
        default=DEFAULT_SPAN_MARGIN,
        help="Fractional margin of the calibrated transmitter span.",
    )
    args = parser.parse_args(argv)

    dat_path = Path(args.raw_dir) / NORMAL_OPERATION_FILE
    spans = derive_spans(dat_path, args.alarm_margin, args.span_margin)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_yaml(spans, NORMAL_OPERATION_FILE, args.alarm_margin, args.span_margin),
        encoding="utf-8",
    )

    _LOG.info("Wrote %d sensor spans to %s.", len(spans), output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
