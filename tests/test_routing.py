import pytest

from src.costing.quote import Mode, QuoteRequest, build_quote
from src.costing.rates import RateCard
from src.costing.routing import Priorities, recommend, transit_days
from tests.conftest import ROOT

ON_TIME = {
    "muara-fleet": 0.94,
    "cempaka": 0.84,
    "hijau": 0.96,
    "raillink": 0.87,
    "nusantara": 0.80,
    "borneo-express": 0.92,
}


@pytest.fixture(scope="module")
def rates():
    return RateCard.load(ROOT / "data" / "reference" / "rates.json")


def _rec(rates, dest="Penang", priorities=None, **kw):
    req = QuoteRequest(destination=dest, container_size="40", **kw)
    return recommend(req, priorities, rates, ON_TIME)


def test_options_cover_every_mode_and_carrier(rates):
    rec = _rec(rates)
    combos = {(o.mode, o.carrier) for o in rec.options}
    assert combos == {
        (Mode.truck, "muara-fleet"),
        (Mode.truck, "cempaka"),
        (Mode.truck, "hijau"),
        (Mode.rail, "raillink"),
    }
    assert [o.rank for o in rec.options] == [1, 2, 3, 4]
    assert rec.recommended == rec.options[0]


def test_carrier_factors_change_price_and_co2(rates):
    rec = {o.carrier: o for o in _rec(rates, "Shah Alam").options}
    base = build_quote(QuoteRequest(destination="Shah Alam", container_size="40"), rates)
    assert rec["muara-fleet"].price_rm == pytest.approx(base.price_rm, abs=0.02)
    assert rec["cempaka"].price_rm < base.price_rm < rec["hijau"].price_rm
    assert rec["hijau"].co2e_kg == pytest.approx(base.co2e_kg * 0.8, abs=0.1)


def test_transit_days(rates):
    carriers = rates.data["carriers"]
    assert transit_days(rates, "shah-alam", Mode.truck, carriers["muara-fleet"]) == 1
    assert transit_days(rates, "penang-prai", Mode.rail, carriers["raillink"]) == 2
    nusantara = transit_days(rates, "kuching", Mode.sea, carriers["nusantara"])
    assert transit_days(rates, "kuching", Mode.sea, carriers["borneo-express"]) == nusantara - 1


def test_priorities_change_the_recommendation(rates):
    cheap = _rec(rates, priorities=Priorities(cost=1, speed=0, co2=0, reliability=0))
    fast = _rec(rates, priorities=Priorities(cost=0, speed=1, co2=0, reliability=0))
    green = _rec(rates, "Shah Alam", priorities=Priorities(cost=0, speed=0, co2=1, reliability=0))
    assert cheap.recommended.mode is Mode.rail
    assert fast.recommended.mode is Mode.truck
    assert green.recommended.carrier == "hijau"


def test_explanation_states_the_trade_off(rates):
    rec = _rec(rates, cargo_type="chemicals")
    assert rec.recommended.carrier == "raillink"
    text = rec.explanation
    assert "1 day slower" in text and "cheaper" in text and "less CO2e" in text


def test_scores_are_normalised(rates):
    rec = _rec(rates)
    for c in ("cost", "speed", "co2", "reliability"):
        scores = [o.criterion_scores[c] for o in rec.options]
        assert min(scores) == 0.0 and max(scores) <= 1.0
    assert sum(rec.priorities.values()) == pytest.approx(1.0)
    assert all(0 <= o.fit <= 100 for o in rec.options)


def test_dominated_option_is_not_pareto_optimal(rates):
    on_time = dict(ON_TIME, cempaka=0.99)  # cheaper AND more reliable than Muara...
    rec = recommend(QuoteRequest(destination="Shah Alam"), None, rates, on_time)
    flags = {o.carrier: o.pareto_optimal for o in rec.options}
    assert flags["cempaka"] and flags["hijau"]  # ...but Muara still wins on CO2, so all survive
    data = rates.data | {"carriers": dict(rates.data["carriers"])}
    data["carriers"]["slowpoke"] = {
        "label": "Slowpoke", "modes": ["truck"], "price_factor": 1.3, "co2_factor": 1.3,
        "transit_days_delta": 1,
    }  # fmt: skip
    rec = recommend(
        QuoteRequest(destination="Shah Alam"), None, RateCard(data), dict(on_time, slowpoke=0.5)
    )
    slow = next(o for o in rec.options if o.carrier == "slowpoke")
    assert not slow.pareto_optimal and slow.rank == len(rec.options)


def test_zero_weights_fall_back_to_equal():
    assert Priorities(cost=0, speed=0, co2=0, reliability=0).normalised()["cost"] == 0.25
