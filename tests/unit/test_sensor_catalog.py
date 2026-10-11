"""Unit tests for data.simulation.sensor_catalog.

Dos garantias: el catalogo real cabe en el contrato `SensorReading` y en el
modelo (cada fuente existe), y un catalogo defectuoso falla al cargarse, no
cuando el validador de Fase 2 deja de ver un sensor (hallazgo H7 del plan).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.schemas.sensor_reading import (
    Measurement,
    MeasurementUnit,
    QualityFlag,
    SensorInfo,
    SensorLocation,
    SensorType,
)
from data.simulation.plant_params import load_neutronics, load_plant_params
from data.simulation.sensor_catalog import (
    DEFAULT_SENSORS_PATH,
    ELEVATION_RANGE_M,
    MIN_SENSORS,
    REQUIRED_ROLES,
    SensorCatalog,
    SensorDef,
    load_sensor_catalog,
)
from data.simulation.thermal_model import OBSERVABLE_NAMES, ModelInputs, ReactorModel
from data.simulation.water_properties import load_water_table

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned catalog."""
    return load_sensor_catalog(_REPO_ROOT / DEFAULT_SENSORS_PATH)


@pytest.fixture(scope="module")
def model() -> ReactorModel:
    """Return the model on the versioned configuration."""
    return ReactorModel(
        load_plant_params(_REPO_ROOT / "configs" / "plant_parameters.yaml"),
        load_neutronics(_REPO_ROOT / "configs" / "neutronics_coefficients.yaml"),
        load_water_table(_REPO_ROOT / "configs" / "water_table.yaml"),
    )


@pytest.fixture(scope="module")
def values(catalog: SensorCatalog, model: ReactorModel) -> dict[str, float]:
    """Return every sensor's noise-free value at the nominal equilibrium."""
    return catalog.true_values(model.observables(model.steady_state(), ModelInputs()))


def _entry(index: int) -> dict[str, Any]:
    return {
        "type": "thermocouple",
        "location": "core",
        "unit": "celsius",
        "elevation_m": 2.0,
        "source": "t_c",
        "gain": 1.0,
        "offset": float(index) * 0.1,
        "noise_sigma": 0.3,
    }


def _document(count: int = MIN_SENSORS) -> dict[str, Any]:
    tags = [f"TC-X-{i:02d}" for i in range(count)]
    roles = {role: tags[:2] for role in REQUIRED_ROLES}
    return {"sensors": {tag: _entry(i) for i, tag in enumerate(tags)}, "roles": roles}


def _write(path: Path, document: object) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


class TestRealCatalog:
    """configs/sim_sensors.yaml against the contract and the model."""

    def test_has_the_proposed_27_sensors(self, catalog: SensorCatalog) -> None:
        """The plan proposes 27 sensors and the minimum is 21."""
        assert len(catalog.sensors) == 27
        assert len(catalog.sensors) >= MIN_SENSORS

    def test_tags_are_unique(self, catalog: SensorCatalog) -> None:
        """No tag repeats."""
        assert len(set(catalog.tags)) == len(catalog.tags)

    def test_covers_every_type_location_and_unit(self, catalog: SensorCatalog) -> None:
        """All 6 sensing principles and all 4 zones appear."""
        assert {s.sensor_type for s in catalog.sensors} == set(SensorType)
        assert {s.location for s in catalog.sensors} == set(SensorLocation)
        assert {s.unit for s in catalog.sensors} <= set(MeasurementUnit)

    def test_sources_are_model_observables(self, catalog: SensorCatalog) -> None:
        """Every source is something ReactorModel.observables delivers."""
        assert {s.source for s in catalog.sensors} <= set(OBSERVABLE_NAMES)

    def test_every_entry_fits_the_reading_contract(self, catalog: SensorCatalog) -> None:
        """SensorInfo and Measurement accept every catalog entry."""
        for sensor in catalog.sensors:
            SensorInfo(
                id=sensor.tag,
                type=sensor.sensor_type,
                location=sensor.location,
                elevation_m=sensor.elevation_m,
            )
            Measurement(value=1.0, unit=sensor.unit, quality=QualityFlag.GOOD, raw_counts=0)

    def test_elevations_are_inside_the_contract_range(self, catalog: SensorCatalog) -> None:
        """elevation_m is in [-10, 100]."""
        low, high = ELEVATION_RANGE_M
        assert all(low <= s.elevation_m <= high for s in catalog.sensors)

    def test_noise_sigma_is_positive(self, catalog: SensorCatalog) -> None:
        """Every sensor has a positive initial sigma."""
        assert all(s.noise_sigma > 0.0 for s in catalog.sensors)

    def test_required_roles_present_and_resolved(self, catalog: SensorCatalog) -> None:
        """The six roles of F3-M2 exist and name catalog tags."""
        for role in REQUIRED_ROLES:
            assert catalog.roles[role]
            assert set(catalog.roles[role]) <= set(catalog.tags)

    def test_roles_have_the_right_physical_type(self, catalog: SensorCatalog) -> None:
        """T roles are thermocouples, flow is flow, P is pressure, flux is flux."""
        expected = {
            "T_out": SensorType.THERMOCOUPLE,
            "T_in": SensorType.THERMOCOUPLE,
            "flow": SensorType.FLOW,
            "flow_hot_leg": SensorType.FLOW,
            "flow_cold_leg": SensorType.FLOW,
            "P": SensorType.PRESSURE,
            "flux": SensorType.FLUX_DETECTOR,
        }
        for role, sensor_type in expected.items():
            assert all(catalog.by_tag(t).sensor_type is sensor_type for t in catalog.roles[role])

    def test_flow_role_is_the_union_of_hot_and_cold_legs(self, catalog: SensorCatalog) -> None:
        """The flow role contains exactly the hot- and cold-leg transmitters."""
        union = set(catalog.roles["flow_hot_leg"]) | set(catalog.roles["flow_cold_leg"])
        assert set(catalog.roles["flow"]) == union

    def test_core_thermocouples_have_a_small_fixed_spatial_offset(
        self, catalog: SensorCatalog
    ) -> None:
        """TC-CORE-xx read T_c shifted by 0,3 to 1,5 K in magnitude."""
        offsets = [abs(catalog.by_tag(f"TC-CORE-0{i}").offset) for i in range(1, 5)]
        assert min(offsets) == pytest.approx(0.3)
        assert max(offsets) == pytest.approx(1.5)


class TestTrueValuesAtNominal:
    """The catalog read on the nominal equilibrium gives plausible plant values."""

    def test_hot_and_cold_legs(self, values: dict[str, float], model: ReactorModel) -> None:
        """Hot leg reads T_out, cold leg reads T_in."""
        assert values["TC-HL-1"] == pytest.approx(model.nominal.t_out_c)
        assert values["TC-CL-1"] == pytest.approx(model.plant.t_in_nominal_c)

    def test_pressure_and_flow(self, values: dict[str, float]) -> None:
        """155 bar in the pressurizer; four loops of 4500 kg/s."""
        assert values["PT-PRZ-01"] == pytest.approx(155.0)
        assert values["FT-HL-1"] == pytest.approx(4500.0)
        assert values["FT-CL-2"] == pytest.approx(4500.0)

    def test_flux_and_calorimetric_power(self, values: dict[str, float]) -> None:
        """100 % flux and 100 % calorimetric power."""
        assert values["NI-FLUX-01"] == pytest.approx(100.0)
        assert values["PWR-TH-01"] == pytest.approx(100.0)

    def test_levels_sit_at_50_percent(self, values: dict[str, float]) -> None:
        """Pressurizer and steam generator levels are mid-range."""
        assert values["LT-PRZ-01"] == pytest.approx(50.0)
        assert values["LT-SG-1"] == pytest.approx(50.0)

    def test_containment_is_atmospheric(self, values: dict[str, float]) -> None:
        """No leak: 1,0 bar."""
        assert values["PT-CNT-01"] == pytest.approx(1.0)


class TestCatalogApi:
    """SensorCatalog helpers."""

    def test_by_tag_unknown_raises(self, catalog: SensorCatalog) -> None:
        """An unknown tag is a KeyError."""
        with pytest.raises(KeyError, match="NOPE"):
            catalog.by_tag("NOPE")

    def test_true_value_formula(self) -> None:
        """value = gain x observable + offset."""
        sensor = SensorDef(
            tag="X",
            sensor_type=SensorType.PRESSURE,
            location=SensorLocation.CORE,
            unit=MeasurementUnit.BAR,
            elevation_m=0.0,
            source="p_primary_bar",
            gain=2.0,
            offset=1.0,
            noise_sigma=0.1,
        )
        assert sensor.true_value({"p_primary_bar": 10.0}) == 21.0
        with pytest.raises(KeyError):
            sensor.true_value({})

    def test_tags_follow_file_order(self, catalog: SensorCatalog) -> None:
        """The first tags are the core thermocouples, as in the file."""
        assert catalog.tags[:2] == ("TC-CORE-01", "TC-CORE-02")


class TestValidation:
    """A defective catalog must fail at load time."""

    def test_minimal_valid_document_loads(self, tmp_path: Path) -> None:
        """The fixture builder produces a loadable catalog (guards the tests below)."""
        loaded = load_sensor_catalog(_write(tmp_path / "s.yaml", _document()))
        assert len(loaded.sensors) == MIN_SENSORS

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_sensor_catalog(tmp_path / "absent.yaml")

    def test_missing_sensors_section(self, tmp_path: Path) -> None:
        """The sensors: section is required."""
        with pytest.raises(KeyError, match="sensors"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", {"roles": {}}))

    def test_sensors_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A list of sensors is a TypeError."""
        with pytest.raises(TypeError, match="mapping"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", {"sensors": [1], "roles": {}}))

    def test_entry_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A scalar entry is a TypeError."""
        document = _document()
        document["sensors"]["TC-X-00"] = 5
        with pytest.raises(TypeError, match="TC-X-00"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    @pytest.mark.parametrize("key", ["type", "source", "noise_sigma", "elevation_m"])
    def test_missing_entry_key(self, tmp_path: Path, key: str) -> None:
        """A missing key names the sensor and the key."""
        document = _document()
        del document["sensors"]["TC-X-03"][key]
        with pytest.raises(KeyError, match=key):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    @pytest.mark.parametrize(
        ("key", "value"),
        [("type", "thermometer"), ("location", "roof"), ("unit", "kelvin")],
    )
    def test_invalid_enum_values(self, tmp_path: Path, key: str, value: str) -> None:
        """Enums must be members of the contract."""
        document = _document()
        document["sensors"]["TC-X-03"][key] = value
        with pytest.raises(ValueError, match="invalid enum"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_incompatible_type_and_unit(self, tmp_path: Path) -> None:
        """A thermocouple cannot read bar."""
        document = _document()
        document["sensors"]["TC-X-03"]["unit"] = "bar"
        with pytest.raises(ValueError, match="not a valid combination"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_unknown_source(self, tmp_path: Path) -> None:
        """The source must be a model observable."""
        document = _document()
        document["sensors"]["TC-X-03"]["source"] = "t_nowhere"
        with pytest.raises(ValueError, match="unknown source"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_custom_observable_names(self, tmp_path: Path) -> None:
        """The accepted sources can be narrowed by the caller."""
        path = _write(tmp_path / "s.yaml", _document())
        with pytest.raises(ValueError, match="unknown source"):
            load_sensor_catalog(path, observable_names=["only_this"])

    @pytest.mark.parametrize("elevation", [-10.5, 100.5])
    def test_elevation_outside_the_contract(self, tmp_path: Path, elevation: float) -> None:
        """elevation_m is bounded by SensorInfo."""
        document = _document()
        document["sensors"]["TC-X-03"]["elevation_m"] = elevation
        with pytest.raises(ValueError, match="elevation_m"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    @pytest.mark.parametrize(("key", "value"), [("gain", 0.0), ("gain", float("inf"))])
    def test_degenerate_gain(self, tmp_path: Path, key: str, value: float) -> None:
        """A zero or infinite gain is rejected."""
        document = _document()
        document["sensors"]["TC-X-03"][key] = value
        with pytest.raises(ValueError, match="gain"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    @pytest.mark.parametrize("sigma", [0.0, -1.0])
    def test_non_positive_noise(self, tmp_path: Path, sigma: float) -> None:
        """noise_sigma must be positive."""
        document = _document()
        document["sensors"]["TC-X-03"]["noise_sigma"] = sigma
        with pytest.raises(ValueError, match="noise_sigma"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_duplicated_tag_in_the_file(self, tmp_path: Path) -> None:
        """YAML would keep the last duplicate silently; the loader refuses it."""
        text = yaml.safe_dump(_document(), sort_keys=False)
        duplicated = text.replace("TC-X-01:", "TC-X-00:", 1)
        path = tmp_path / "s.yaml"
        path.write_text(duplicated, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicated key"):
            load_sensor_catalog(path)

    def test_too_few_sensors(self, tmp_path: Path) -> None:
        """Fewer than MIN_SENSORS is rejected."""
        with pytest.raises(ValueError, match="at least"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", _document(MIN_SENSORS - 1)))

    def test_missing_roles_section(self, tmp_path: Path) -> None:
        """roles: is required."""
        document = _document()
        del document["roles"]
        with pytest.raises(KeyError, match="roles"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_roles_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A list for roles is a TypeError."""
        document = _document()
        document["roles"] = ["T_out"]
        with pytest.raises(TypeError, match="roles"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_role_members_must_be_a_list(self, tmp_path: Path) -> None:
        """A role holding a scalar is a TypeError."""
        document = _document()
        document["roles"]["T_out"] = "TC-X-00"
        with pytest.raises(TypeError, match="T_out"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_role_with_unknown_tag(self, tmp_path: Path) -> None:
        """A role may only name catalog tags."""
        document = copy.deepcopy(_document())
        document["roles"]["T_out"] = ["TC-GHOST"]
        with pytest.raises(ValueError, match="unknown tags"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_empty_role(self, tmp_path: Path) -> None:
        """A role cannot be empty."""
        document = _document()
        document["roles"]["flux"] = []
        with pytest.raises(ValueError, match="empty"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))

    def test_missing_required_role(self, tmp_path: Path) -> None:
        """The six roles of F3-M2 are mandatory."""
        document = _document()
        del document["roles"]["P_th"]
        with pytest.raises(KeyError, match="P_th"):
            load_sensor_catalog(_write(tmp_path / "s.yaml", document))
