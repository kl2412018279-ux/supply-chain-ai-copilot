"""Smoke test every dashboard page headlessly."""

import pytest

from tests.conftest import ROOT

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

APP = str(ROOT / "app.py")
MANAGER = {"username": "manager", "role": "manager", "display_name": "Demo", "demo": True}


@pytest.fixture
def app(trained_model, ingested_store, delay_model):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["user"] = MANAGER
    at.run()
    assert not at.exception
    return at


def test_control_tower_is_home(app):
    labels = [m.label for m in app.metric]
    for label in ("Past free time", "Charges at planned move", "Gross margin (90 d)",
                  "Margin leaked (90 d)", "On-time delivery (30 d)"):  # fmt: skip
        assert label in labels
    assert any("Biggest leak" in i.value for i in app.info)
    app.pills[0].set_value(["Customs hold"]).run()
    assert not app.exception


def test_copilot_page_suggestion_and_chat(app):
    app.switch_page("views/copilot.py").run()
    # Empty state shows one-click suggestions
    labels = [b.label for b in app.button]
    assert any(label.startswith("Quote 2 x 40ft") for label in labels)

    app.chat_input[0].set_value("Quote a 40ft container to Shah Alam").run()
    assert not app.exception
    assert any("Price: RM" in m.value for m in app.markdown)


def test_quote_page(app):
    app.switch_page("views/quote.py").run()
    assert not app.exception
    price = next(m for m in app.metric if m.label == "Quoted price")
    assert price.value.startswith("RM")
    labels = {m.label for m in app.metric}
    assert {"Cost", "Margin"} <= labels

    # What-if: higher diesel shows a delta against the rate card
    diesel = next(s for s in app.slider if s.label.startswith("Diesel"))
    diesel.set_value(3.5).run()
    price = next(m for m in app.metric if m.label == "Quoted price")
    assert "vs rate card" in (price.delta or "")


def test_quote_modes_view_shows_transit_days(app):
    app.switch_page("views/quote.py").run()
    app.selectbox[0].set_value("penang-prai").run()
    view = next(b for b in app.button_group if "Modes" in b.options)
    view.set_value("Modes").run()
    assert not app.exception
    assert any("per day saved" in i.value for i in app.info)


def test_forecast_and_documents_pages(app):
    app.switch_page("views/forecast.py").run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Forecast volume" in labels and "Prime movers at peak" in labels

    app.switch_page("views/documents.py").run()
    assert not app.exception
    app.text_input[0].set_value("K8 transshipment").run()
    assert any("K8" in m.value for m in app.markdown)


def test_new_pages_render(app):
    app.switch_page("views/planner.py").run()
    assert not app.exception
    assert any("Recommended:" in a.value for a in app.success)
    labels = {m.label for m in app.metric}
    assert {"On time", "Damage"} <= labels
    assert {s.label for s in app.slider} >= {"Low price", "Fast delivery", "Cargo safety"}

    app.switch_page("views/delays.py").run()
    assert not app.exception
    assert any(m.label == "Delay risk" for m in app.metric)

    app.switch_page("views/intake.py").run()
    assert not app.exception
    assert any("check digit" in e.value for e in app.error)  # default sample has a bad container

    app.switch_page("views/hauliers.py").run()
    assert not app.exception
    assert any(m.label == "Grade" for m in app.metric)
    assert any(m.label == "Compliance" for m in app.metric)


def test_quote_route_view_with_road_route(app, monkeypatch):
    from src.costing import road_routing
    from tests.test_live_data import ORS_SAMPLE

    def pick_route_view():
        app.switch_page("views/quote.py").run()
        view = next(b for b in app.button_group if "Route" in b.options)
        view.set_value("Route").run()
        assert not app.exception

    pick_route_view()  # no ORS key: straight line plus setup hint
    assert any("ORS_API_KEY" in c.value for c in app.caption)

    route = road_routing.parse_ors_geojson(ORS_SAMPLE)
    monkeypatch.setattr(road_routing, "road_route", lambda *a, **k: route)
    pick_route_view()
    assert any("OpenRouteService" in c.value for c in app.caption)
    assert any("Turn-by-turn" in e.label for e in app.expander)


def test_delay_page_shows_and_overrides_rain(app):
    app.switch_page("views/delays.py").run()
    assert any("Rain at" in c.value and "offline" in c.value for c in app.caption)
    next(t for t in app.toggle if t.label == "Set rainfall manually").set_value(True).run()
    assert not app.exception
    assert any("entered by user" in c.value for c in app.caption)


def test_login_required_and_demo_sign_in(trained_model, ingested_store, delay_model):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert not any(m.label == "Past free time" for m in at.metric)  # nothing before login
    assert any("muara-demo" in m.value for m in at.markdown)  # demo credentials shown

    at.text_input(key="login_username").set_value("sales")
    at.text_input(key="login_password").set_value("wrong")
    at.button[0].click().run()
    assert any("Wrong username or password" in e.value for e in at.error)

    at.text_input(key="login_username").set_value("sales")
    at.text_input(key="login_password").set_value("muara-demo")
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["user"]["role"] == "sales"
    assert any(m.label == "Past free time" for m in at.metric)  # landed on the control tower


def test_role_limits_pages(app):
    app.session_state["user"] = {**MANAGER, "username": "sales", "role": "sales"}
    app.switch_page("views/quote.py").run()
    assert not app.exception
    # Delay risk is an operations page: not registered for sales, so it can't be opened.
    with pytest.raises(ValueError, match="(?i)not find|unknown|page"):
        app.switch_page("views/delays.py").run()
