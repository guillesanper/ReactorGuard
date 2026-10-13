"""Unit tests for data.simulation.plant_params.

Mismo criterio que test_tep_params.py: la configuracion real debe resolverse
completa y una configuracion incompleta o fuera de rango debe fallar con un
mensaje que nombre la clave, no con un KeyError opaco ni con un modelo que
integra con parametros no fisicos.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.simulation.plant_params import (
    DEFAULT_NEUTRONICS_PATH,
    DEFAULT_PLANT_PARAMS_PATH,
    N_PRECURSOR_GROUPS,
    NeutronicsCoefficients,
    PlantParams,
    load_neutronics,
    load_plant_params,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _plant_document() -> dict[str, Any]:
    with (_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH).open(encoding="utf-8") as fh:
        document: dict[str, Any] = yaml.safe_load(fh)
    return document


def _neutronics_document() -> dict[str, Any]:
    with (_REPO_ROOT / DEFAULT_NEUTRONICS_PATH).open(encoding="utf-8") as fh:
        document: dict[str, Any] = yaml.safe_load(fh)
    return document


def _dump(path: Path, document: object) -> Path:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class TestRealPlantParameters:
    """configs/plant_parameters.yaml must resolve completely."""

    def test_returns_plant_params(self) -> None:
        """The loader returns the dataclass."""
        params = load_plant_params(_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH)
        assert isinstance(params, PlantParams)

    def test_nominal_point_matches_the_plan(self) -> None:
        """3000 MWt, 18000 kg/s, 155 bar and T_in 291 C."""
        params = load_plant_params(_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH)
        assert params.thermal_power_mw == 3000.0
        assert params.nominal_flow_kg_s == 18000.0
        assert params.nominal_pressure_bar == 155.0
        assert params.t_in_nominal_c == 291.0

    def test_limits_follow_the_tdd(self) -> None:
        """dT/dt 10 C/s (D9) and 5 % energy imbalance (TDD 5.4)."""
        limits = load_plant_params(_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH).limits
        assert limits.max_dt_c_per_s == 10.0
        assert limits.energy_imbalance_fraction == 0.05

    def test_constraint_scales_added_in_f3_m2(self) -> None:
        """The six keys F3-M2 added to limits: resolve with their documented units."""
        limits = load_plant_params(_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH).limits
        assert limits.n_loops == 4
        assert isinstance(limits.n_loops, int)
        assert limits.mass_balance_tolerance == 0.055
        assert limits.quasi_steady_dt_avg_k_s == 0.02
        assert limits.power_flux_k == 1.0
        assert limits.power_flux_threshold_percent == 4.5
        assert limits.conservation_tolerance == 1.0e-6

    def test_pump_max_is_derived_from_the_fraction(self) -> None:
        """pump_max_kg_s = fraction x nominal flow."""
        params = load_plant_params(_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH)
        assert params.pump_max_kg_s == pytest.approx(
            params.limits.pump_max_fraction * params.nominal_flow_kg_s
        )

    def test_fault_tiers_hold_the_three_process_faults(self) -> None:
        """F3-M4 fills the block: one entry per process fault, three levels each."""
        tiers = load_plant_params(_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH).fault_tiers
        assert set(tiers) == {
            "core_temp_excursion",
            "coolant_flow_reduction",
            "void_formation",
        }
        for block in tiers.values():
            assert set(block["levels"]) == {"low", "medium", "high"}

    def test_params_are_frozen(self) -> None:
        """PlantParams is immutable."""
        params = load_plant_params(_REPO_ROOT / DEFAULT_PLANT_PARAMS_PATH)
        with pytest.raises(AttributeError):
            params.thermal_power_mw = 1.0  # type: ignore[misc]


class TestPlantParameterErrors:
    """Incomplete or non-physical configurations must be rejected clearly."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_plant_params(tmp_path / "absent.yaml")

    def test_top_level_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A list at the top level is a TypeError."""
        with pytest.raises(TypeError, match="top level"):
            load_plant_params(_dump(tmp_path / "p.yaml", [1, 2]))

    @pytest.mark.parametrize("section", ["plant", "control", "limits"])
    def test_missing_section(self, tmp_path: Path, section: str) -> None:
        """Each section is required and named in the error."""
        document = _plant_document()
        del document[section]
        with pytest.raises(KeyError, match=f"'{section}:'"):
            load_plant_params(_dump(tmp_path / "p.yaml", document))

    def test_section_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A scalar section is a TypeError."""
        document = _plant_document()
        document["plant"] = 3
        with pytest.raises(TypeError, match="must be a mapping"):
            load_plant_params(_dump(tmp_path / "p.yaml", document))

    @pytest.mark.parametrize(
        ("section", "key"),
        [("plant", "thermal_power_mw"), ("control", "tau_rod_s"), ("limits", "max_dt_c_per_s")],
    )
    def test_missing_key_is_named(self, tmp_path: Path, section: str, key: str) -> None:
        """The error names the missing key."""
        document = _plant_document()
        del document[section][key]
        with pytest.raises(KeyError, match=key):
            load_plant_params(_dump(tmp_path / "p.yaml", document))

    def test_missing_fault_tiers(self, tmp_path: Path) -> None:
        """fault_tiers is required even when empty."""
        document = _plant_document()
        del document["fault_tiers"]
        with pytest.raises(KeyError, match="fault_tiers"):
            load_plant_params(_dump(tmp_path / "p.yaml", document))

    def test_fault_tiers_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A list for fault_tiers is a TypeError."""
        document = _plant_document()
        document["fault_tiers"] = [1]
        with pytest.raises(TypeError, match="fault_tiers"):
            load_plant_params(_dump(tmp_path / "p.yaml", document))

    @pytest.mark.parametrize(
        ("section", "key", "value"),
        [
            ("plant", "thermal_power_mw", 0.0),
            ("plant", "nominal_flow_kg_s", -5.0),
            ("plant", "capacity_core_mj_k", 0.0),
            ("plant", "tau_pump_s", 0.0),
            ("plant", "void_ramp_width_k", 0.0),
            ("control", "rod_ki_per_k_s", 0.0),
            ("limits", "pump_max_fraction", -1.0),
            ("limits", "mass_balance_tolerance", 0.0),
            ("limits", "quasi_steady_dt_avg_k_s", -0.01),
            ("limits", "power_flux_k", 0.0),
            ("limits", "power_flux_threshold_percent", 0.0),
            ("limits", "conservation_tolerance", -1.0e-6),
        ],
    )
    def test_non_positive_values_are_rejected(
        self, tmp_path: Path, section: str, key: str, value: float
    ) -> None:
        """Strictly positive quantities cannot be zero or negative."""
        document = _plant_document()
        document[section][key] = value
        with pytest.raises(ValueError, match="must be > 0"):
            load_plant_params(_dump(tmp_path / "p.yaml", document))

    @pytest.mark.parametrize("value", [0.0, 2.5, -1.0, float("nan"), float("inf")])
    def test_n_loops_must_be_a_whole_number(self, tmp_path: Path, value: float) -> None:
        """The loop count divides the flow: fractions, zero and non-finite are rejected."""
        document = _plant_document()
        document["limits"]["n_loops"] = value
        with pytest.raises(ValueError, match="n_loops"):
            load_plant_params(_dump(tmp_path / "p.yaml", document))

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("void_max_fraction", 1.0, "void_max_fraction"),
            ("t_secondary_nominal_c", 295.0, "t_secondary_nominal_c"),
            ("t_fuel_nominal_c", 290.0, "t_fuel_nominal_c"),
        ],
    )
    def test_inconsistent_temperatures_are_rejected(
        self, tmp_path: Path, key: str, value: float, message: str
    ) -> None:
        """The thermal ladder fuel > coolant > secondary must hold."""
        document = _plant_document()
        document["plant"][key] = value
        with pytest.raises(ValueError, match=message):
            load_plant_params(_dump(tmp_path / "p.yaml", document))


class TestRealNeutronics:
    """configs/neutronics_coefficients.yaml must resolve completely."""

    def test_returns_coefficients(self) -> None:
        """The loader returns the dataclass."""
        coefficients = load_neutronics(_REPO_ROOT / DEFAULT_NEUTRONICS_PATH)
        assert isinstance(coefficients, NeutronicsCoefficients)

    def test_provenance_is_the_placeholder(self) -> None:
        """Until F3-M8 the values are literature placeholders."""
        coefficients = load_neutronics(_REPO_ROOT / DEFAULT_NEUTRONICS_PATH)
        assert coefficients.provenance == "LITERATURE_PLACEHOLDER"

    def test_six_groups_and_standard_beta(self) -> None:
        """Keepin's six groups add up to 0,0065."""
        coefficients = load_neutronics(_REPO_ROOT / DEFAULT_NEUTRONICS_PATH)
        assert len(coefficients.beta) == N_PRECURSOR_GROUPS
        assert coefficients.beta_total == pytest.approx(0.0065, rel=1e-3)

    def test_fractions_sum_to_one(self) -> None:
        """Group shares of the delayed fraction are a partition."""
        coefficients = load_neutronics(_REPO_ROOT / DEFAULT_NEUTRONICS_PATH)
        assert sum(coefficients.beta_fractions) == pytest.approx(1.0)

    def test_feedback_is_negative(self) -> None:
        """Doppler, moderator and void coefficients are all negative."""
        coefficients = load_neutronics(_REPO_ROOT / DEFAULT_NEUTRONICS_PATH)
        assert coefficients.alpha_fuel_per_k < 0.0
        assert coefficients.alpha_moderator_per_k < 0.0
        assert coefficients.alpha_void_per_fraction < 0.0


class TestNeutronicsErrors:
    """Non-physical neutronics must be rejected."""

    def _edit(self, tmp_path: Path, **changes: Any) -> Path:
        document = copy.deepcopy(_neutronics_document())
        document["neutronics"].update(changes)
        return _dump(tmp_path / "n.yaml", document)

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_neutronics(tmp_path / "absent.yaml")

    def test_missing_provenance(self, tmp_path: Path) -> None:
        """provenance is mandatory."""
        document = _neutronics_document()
        del document["provenance"]
        with pytest.raises(KeyError, match="provenance"):
            load_neutronics(_dump(tmp_path / "n.yaml", document))

    def test_missing_section(self, tmp_path: Path) -> None:
        """The neutronics: section is mandatory."""
        with pytest.raises(KeyError, match="neutronics"):
            load_neutronics(_dump(tmp_path / "n.yaml", {"provenance": "X"}))

    def test_missing_key(self, tmp_path: Path) -> None:
        """A missing coefficient is named."""
        document = _neutronics_document()
        del document["neutronics"]["generation_time_s"]
        with pytest.raises(KeyError, match="generation_time_s"):
            load_neutronics(_dump(tmp_path / "n.yaml", document))

    def test_wrong_group_count(self, tmp_path: Path) -> None:
        """Exactly 6 precursor groups."""
        with pytest.raises(ValueError, match="6 groups"):
            load_neutronics(self._edit(tmp_path, beta=[0.001, 0.002]))

    def test_non_positive_group_values(self, tmp_path: Path) -> None:
        """beta and decay constants must be positive."""
        with pytest.raises(ValueError, match="must be > 0"):
            load_neutronics(
                self._edit(tmp_path, decay_constants_per_s=[0.01, 0.03, 0.1, 0.3, 1.1, 0.0])
            )

    def test_total_beta_must_be_below_one(self, tmp_path: Path) -> None:
        """The delayed fraction is a fraction."""
        with pytest.raises(ValueError, match="< 1"):
            load_neutronics(self._edit(tmp_path, beta=[0.2] * 6))

    @pytest.mark.parametrize(
        "key", ["alpha_fuel_per_k", "alpha_moderator_per_k", "alpha_void_per_fraction"]
    )
    def test_positive_feedback_is_rejected(self, tmp_path: Path, key: str) -> None:
        """The model requires negative reactivity feedback."""
        with pytest.raises(ValueError, match="negative"):
            load_neutronics(self._edit(tmp_path, **{key: 1.0e-5}))

    @pytest.mark.parametrize("key", ["generation_time_s", "rod_worth_per_percent"])
    def test_non_positive_scalars(self, tmp_path: Path, key: str) -> None:
        """Generation time and rod worth are strictly positive."""
        with pytest.raises(ValueError, match="must be > 0"):
            load_neutronics(self._edit(tmp_path, **{key: 0.0}))
