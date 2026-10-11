"""Catalogo de sensores de la simulacion y su validacion contra el contrato.

El catalogo vive en configs/sim_sensors.yaml. Este modulo lo carga y comprueba
que cada entrada cabe en `SensorReading` (enums, elevation_m en [-10, 100]),
que su `source` es una magnitud que el modelo entrega y que los `roles` que
leen las constraints apuntan a tags que existen. Un catalogo mal formado falla
aqui, no cuando el validador de Fase 2 deja de ver un sensor (ver H7 del plan).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from data.schemas.sensor_reading import MeasurementUnit, SensorLocation, SensorType
from data.simulation.thermal_model import OBSERVABLE_NAMES

DEFAULT_SENSORS_PATH = Path("configs/sim_sensors.yaml")

MIN_SENSORS = 21
"""Minimo del catalogo (D2): menos de 21 canales no cubren los 5 tipos y 4 zonas."""

ELEVATION_RANGE_M = (-10.0, 100.0)
"""Rango de `SensorInfo.elevation_m` en el contrato."""

REQUIRED_ROLES: tuple[str, ...] = ("T_out", "T_in", "flow", "P", "flux", "P_th")
"""Roles que las constraints de F3-M2 necesitan para evaluar las 6 reglas."""

_COMPATIBLE_UNITS: dict[SensorType, frozenset[MeasurementUnit]] = {
    SensorType.THERMOCOUPLE: frozenset({MeasurementUnit.CELSIUS}),
    SensorType.PRESSURE: frozenset({MeasurementUnit.BAR}),
    SensorType.FLOW: frozenset({MeasurementUnit.KG_S}),
    SensorType.FLUX_DETECTOR: frozenset({MeasurementUnit.PERCENT}),
    SensorType.POSITION: frozenset({MeasurementUnit.PERCENT}),
    SensorType.NORMALIZED: frozenset({MeasurementUnit.PERCENT, MeasurementUnit.NORMALIZED}),
}

_ENTRY_KEYS = (
    "type",
    "location",
    "unit",
    "elevation_m",
    "source",
    "gain",
    "offset",
    "noise_sigma",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader que rechaza claves repetidas en un mapping.

    PyYAML se queda en silencio con la ultima aparicion de una clave: para un
    catalogo eso significaria perder un sensor sin aviso.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Build a mapping, raising on duplicated keys.

        Args:
            node: YAML mapping node.
            deep: Whether to construct nested objects immediately.

        Returns:
            The constructed dict.

        Raises:
            ValueError: If a key appears more than once.
        """
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f"sim_sensors.yaml: duplicated key '{key}'.")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class SensorDef:
    """Definicion de un sensor del catalogo.

    Attributes:
        tag: Etiqueta unica del instrumento (SensorInfo.id).
        sensor_type: Principio de medida.
        location: Zona de la planta.
        unit: Unidad de ingenieria.
        elevation_m: Elevacion respecto al datum de planta [m].
        source: Nombre de la magnitud verdadera que lee (OBSERVABLE_NAMES).
        gain: Ganancia sobre la magnitud.
        offset: Desvio constante en unidades de ingenieria.
        noise_sigma: Sigma inicial del ruido blanco, en unidades de ingenieria.
    """

    tag: str
    sensor_type: SensorType
    location: SensorLocation
    unit: MeasurementUnit
    elevation_m: float
    source: str
    gain: float
    offset: float
    noise_sigma: float

    def true_value(self, observables: Mapping[str, float]) -> float:
        """Return the noise-free value this sensor reads.

        Args:
            observables: Magnitudes verdaderas del modelo.

        Returns:
            gain * observables[source] + offset.

        Raises:
            KeyError: If the source is not in observables.
        """
        return self.gain * observables[self.source] + self.offset


@dataclass(frozen=True)
class SensorCatalog:
    """Conjunto ordenado de sensores y roles para las constraints.

    Attributes:
        sensors: Sensores en el orden del fichero.
        roles: Rol -> tags que lo desempenan.
    """

    sensors: tuple[SensorDef, ...]
    roles: dict[str, tuple[str, ...]]

    @property
    def tags(self) -> tuple[str, ...]:
        """Return the sensor tags in catalog order."""
        return tuple(sensor.tag for sensor in self.sensors)

    def by_tag(self, tag: str) -> SensorDef:
        """Return the sensor with a given tag.

        Args:
            tag: Instrument tag.

        Returns:
            The SensorDef.

        Raises:
            KeyError: If the tag is not in the catalog.
        """
        for sensor in self.sensors:
            if sensor.tag == tag:
                return sensor
        raise KeyError(f"sensor catalog: unknown tag '{tag}'.")

    def true_values(self, observables: Mapping[str, float]) -> dict[str, float]:
        """Return the noise-free value of every sensor.

        Args:
            observables: Magnitudes verdaderas del modelo.

        Returns:
            Mapping tag -> value, in catalog order.
        """
        return {sensor.tag: sensor.true_value(observables) for sensor in self.sensors}


def _parse_entry(tag: str, entry: Any, observable_names: frozenset[str]) -> SensorDef:
    """Build and validate one SensorDef.

    Raises:
        TypeError: If the entry is not a mapping.
        KeyError: If a required key is missing.
        ValueError: If a value is not valid for the contract or the model.
    """
    if not isinstance(entry, dict):
        raise TypeError(f"sim_sensors.yaml: entry for '{tag}' must be a mapping.")
    for key in _ENTRY_KEYS:
        if key not in entry:
            raise KeyError(f"sim_sensors.yaml: '{tag}' is missing key '{key}'.")
    try:
        sensor_type = SensorType(entry["type"])
        location = SensorLocation(entry["location"])
        unit = MeasurementUnit(entry["unit"])
    except ValueError as exc:
        raise ValueError(f"sim_sensors.yaml: '{tag}' has an invalid enum value: {exc}") from exc

    if unit not in _COMPATIBLE_UNITS[sensor_type]:
        raise ValueError(
            f"sim_sensors.yaml: '{tag}' pairs type '{sensor_type.value}' with unit "
            f"'{unit.value}', which is not a valid combination."
        )
    source = str(entry["source"])
    if source not in observable_names:
        raise ValueError(f"sim_sensors.yaml: '{tag}' reads unknown source '{source}'.")

    elevation = float(entry["elevation_m"])
    low, high = ELEVATION_RANGE_M
    if not low <= elevation <= high:
        raise ValueError(
            f"sim_sensors.yaml: '{tag}' elevation_m={elevation} is outside [{low}, {high}]."
        )
    gain = float(entry["gain"])
    offset = float(entry["offset"])
    sigma = float(entry["noise_sigma"])
    if not (math.isfinite(gain) and gain != 0.0 and math.isfinite(offset)):
        raise ValueError(f"sim_sensors.yaml: '{tag}' needs a finite, non-zero gain and offset.")
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sim_sensors.yaml: '{tag}' needs noise_sigma > 0.")
    return SensorDef(
        tag=tag,
        sensor_type=sensor_type,
        location=location,
        unit=unit,
        elevation_m=elevation,
        source=source,
        gain=gain,
        offset=offset,
        noise_sigma=sigma,
    )


def _parse_roles(raw: Any, tags: set[str]) -> dict[str, tuple[str, ...]]:
    """Validate the roles block against the known tags.

    Raises:
        TypeError: If roles is not a mapping of lists.
        KeyError: If a required role is missing.
        ValueError: If a role is empty or names an unknown tag.
    """
    if not isinstance(raw, dict):
        raise TypeError("sim_sensors.yaml: 'roles:' must be a mapping.")
    roles: dict[str, tuple[str, ...]] = {}
    for name, members in raw.items():
        if not isinstance(members, list):
            raise TypeError(f"sim_sensors.yaml: role '{name}' must be a list of tags.")
        resolved = tuple(str(tag) for tag in members)
        unknown = [tag for tag in resolved if tag not in tags]
        if unknown:
            raise ValueError(f"sim_sensors.yaml: role '{name}' names unknown tags {unknown}.")
        if not resolved:
            raise ValueError(f"sim_sensors.yaml: role '{name}' is empty.")
        roles[str(name)] = resolved
    for required in REQUIRED_ROLES:
        if required not in roles:
            raise KeyError(f"sim_sensors.yaml: missing required role '{required}'.")
    return roles


def load_sensor_catalog(
    path: str | Path = DEFAULT_SENSORS_PATH,
    observable_names: Iterable[str] = OBSERVABLE_NAMES,
) -> SensorCatalog:
    """Load and validate the simulation sensor catalog.

    Args:
        path: Path to the YAML file.
        observable_names: Magnitudes the model can deliver; a sensor `source`
            must be one of them.

    Returns:
        The validated SensorCatalog.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError: If a section, key or required role is missing.
        TypeError: If a block has the wrong YAML type.
        ValueError: If a key is duplicated, the catalog has fewer than
            MIN_SENSORS sensors, or an entry is invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Sensor catalog not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        document = yaml.load(fh, Loader=_UniqueKeyLoader) or {}  # noqa: S506 - SafeLoader
    if "sensors" not in document:
        raise KeyError(f"{file_path}: missing required section 'sensors:'.")
    raw_sensors = document["sensors"]
    if not isinstance(raw_sensors, dict):
        raise TypeError(f"{file_path}: 'sensors:' must be a mapping of tag to entry.")

    names = frozenset(observable_names)
    sensors = tuple(_parse_entry(str(tag), entry, names) for tag, entry in raw_sensors.items())
    if len(sensors) < MIN_SENSORS:
        raise ValueError(
            f"{file_path}: catalog has {len(sensors)} sensors, at least {MIN_SENSORS} required."
        )
    if "roles" not in document:
        raise KeyError(f"{file_path}: missing required section 'roles:'.")
    roles = _parse_roles(document["roles"], {sensor.tag for sensor in sensors})
    return SensorCatalog(sensors=sensors, roles=roles)
