import json

import pytest
from pydantic import ValidationError

from src.costing.quote import Mode, QuoteRequest, build_quote, compare_modes
from src.costing.rates import RateCard, get_rate_card
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def rates():
    return RateCard.load(ROOT / "data" / "reference" / "rates.json")


def _total(quote, category):
    return sum(i.amount_rm for i in quote.line_items if i.category == category)


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Penang", "penang-prai"),
        ("Prai", "penang-prai"),
        ("KL", "kuala-lumpur"),
        ("Johor", "pasir-gudang"),
        ("quote 2x40ft to Pasir Gudang please", "pasir-gudang"),
        ("Sabah", "kota-kinabalu"),
    ],
)
def test_destination_matching(rates, text, key):
    assert rates.resolve_destination(text) == key


def test_unknown_destination(rates):
    with pytest.raises(KeyError):
        rates.resolve_destination("Bangkok")
    assert rates.find_destination("no place here") is None


def test_fuel_cost_matches_rate_card(rates):
    q = build_quote(QuoteRequest(destination="Shah Alam", container_size="20"), rates)
    litres = 40 * (0.45 + 0.32)  # round trip: loaded out, empty back
    assert q.diesel_litres == pytest.approx(litres, abs=0.1)
    assert _total(q, "fuel") == pytest.approx(litres * 2.99, abs=0.01)
    assert q.co2e_kg == pytest.approx(litres * 2.68, abs=0.1)  # CO2 from the same litres


def test_price_is_cost_plus_margin(rates):
    q = build_quote(QuoteRequest(destination="Ipoh", containers=3), rates)
    assert q.cost_rm == pytest.approx(sum(i.amount_rm for i in q.line_items), abs=0.01)
    assert q.price_rm == pytest.approx(q.cost_rm * 1.15, abs=0.02)
    assert q.price_per_container_rm == pytest.approx(q.price_rm / 3, abs=0.01)
    assert q.cost_per_teu_rm == pytest.approx(q.cost_rm / 6, abs=0.01)  # 3 x 40ft = 6 TEU


def test_port_handling_uses_westports_tariff(rates):
    q20 = build_quote(QuoteRequest(destination="KL", container_size="20"), rates)
    q40 = build_quote(QuoteRequest(destination="KL", container_size="40"), rates)
    assert _total(q20, "port") == 375.0 and _total(q40, "port") == 562.5


def test_chemicals_cost_more_than_general_cargo(rates):
    general = build_quote(QuoteRequest(destination="Melaka"), rates)
    chem = build_quote(QuoteRequest(destination="Melaka", cargo_type="chemicals"), rates)
    assert chem.price_rm > general.price_rm
    assert any("surcharge 25%" in i.description for i in chem.line_items)


def test_dg_warehousing_rate_multiplier(rates):
    base = {"destination": "Shah Alam", "warehouse_pallets": 10, "warehouse_days": 10}
    general = build_quote(QuoteRequest(**base), rates)
    chem = build_quote(QuoteRequest(**base, cargo_type="chemicals"), rates)
    storage = [i for i in chem.line_items if i.description.startswith("Storage")][0]
    assert storage.amount_rm == pytest.approx(10 * 10 * 1.2 * 1.8)
    assert _total(general, "warehouse") == pytest.approx(10 * 10 * 1.2 + 10 * 8)


def test_northport_is_closer(rates):
    west = build_quote(QuoteRequest(destination="Shah Alam"), rates)
    north = build_quote(QuoteRequest(destination="Shah Alam", origin="Northport"), rates)
    assert north.distance_km < west.distance_km and north.cost_rm < west.cost_rm


def test_overweight_container_warns(rates):
    q = build_quote(QuoteRequest(destination="Nilai", cargo_weight_kg=24000), rates)
    assert q.warnings and "25,000 kg" in q.warnings[0]


def test_customs_can_be_excluded(rates):
    q = build_quote(QuoteRequest(destination="Nilai", customs_clearance=False), rates)
    assert _total(q, "customs") == 0


def test_rail_not_offered_everywhere(rates):
    with pytest.raises(ValueError):
        build_quote(QuoteRequest(destination="Shah Alam", mode=Mode.rail), rates)


def test_east_malaysia_goes_by_sea(rates):
    q = build_quote(QuoteRequest(destination="Kuching"), rates)
    assert q.mode is Mode.sea and q.diesel_litres == 0
    assert "Port-to-port" in q.warnings[0]


def test_compare_modes_long_haul(rates):
    cmp = compare_modes(QuoteRequest(destination="Penang", cargo_type="chemicals"), rates)
    assert [q.mode for q in cmp.quotes] == [Mode.truck, Mode.rail]
    rail = cmp.quotes[1]
    assert rail.co2e_kg < cmp.quotes[0].co2e_kg
    assert cmp.lowest_co2_mode is Mode.rail


def test_carbon_abatement_cost_when_greener_mode_is_dearer(tmp_path):
    data = json.loads((ROOT / "data" / "reference" / "rates.json").read_text())
    data["rail"]["rm_per_container_km"] = {"20": 20.0, "40": 30.0}  # make rail expensive
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(data))
    cmp = compare_modes(QuoteRequest(destination="Ipoh"), RateCard.load(path))
    assert cmp.cheapest_mode is Mode.truck and cmp.lowest_co2_mode is Mode.rail
    assert cmp.carbon_abatement_rm_per_tonne > 0


@pytest.mark.parametrize("size", ["20ft", "40'", 40])
def test_container_size_normalisation(size):
    assert QuoteRequest(destination="KL", container_size=size).container_size in {"20", "40"}


def test_invalid_requests_rejected():
    with pytest.raises(ValidationError):
        QuoteRequest(destination="KL", container_size="45")
    with pytest.raises(ValidationError):
        QuoteRequest(destination="KL", containers=0)


def test_default_rate_card_loads():
    assert get_rate_card().value("fuel", "diesel_rm_per_litre_peninsular") > 0
