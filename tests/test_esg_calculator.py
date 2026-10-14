import pytest

from src.forecasting.esg_calculator import (
    EMISSION_FACTORS_KG_PER_TKM,
    calculate_carbon,
    calculate_emissions,
)


def test_truck_emissions_match_formula():
    # 300 km * 0.5 t * 0.105 kg/t-km
    assert calculate_carbon(300, 500, "truck") == 15.75


@pytest.mark.parametrize("mode", list(EMISSION_FACTORS_KG_PER_TKM))
def test_every_mode_is_supported(mode):
    assert calculate_carbon(1000, 1000, mode) == EMISSION_FACTORS_KG_PER_TKM[mode] * 1000


def test_mode_is_case_insensitive():
    assert calculate_carbon(100, 1000, " AIR ") == 50.0


def test_comparison_finds_lowest_mode():
    result = calculate_emissions(300, 500, "air")
    assert result.co2e_kg == 75.0
    assert result.lowest_emission_mode.value == "sea"
    assert set(result.alternatives_co2e_kg) == set(EMISSION_FACTORS_KG_PER_TKM)


@pytest.mark.parametrize(
    ("distance", "weight", "mode"), [(0, 10, "truck"), (10, -1, "truck"), (10, 10, "rocket")]
)
def test_invalid_input_raises(distance, weight, mode):
    with pytest.raises(ValueError):
        calculate_carbon(distance, weight, mode)
