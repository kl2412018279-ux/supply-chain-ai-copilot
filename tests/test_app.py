"""Smoke test every dashboard page headlessly."""

import pytest

from tests.conftest import ROOT

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

APP = str(ROOT / "app.py")


@pytest.fixture
def app(trained_model, ingested_store):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def test_copilot_page_suggestion_and_chat(app):
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

    # What-if: higher diesel shows a delta against the rate card
    diesel = next(s for s in app.slider if s.label.startswith("Diesel"))
    diesel.set_value(3.5).run()
    price = next(m for m in app.metric if m.label == "Quoted price")
    assert "vs rate card" in (price.delta or "")


def test_forecast_and_documents_pages(app):
    app.switch_page("views/forecast.py").run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Forecast volume" in labels and "Prime movers at peak" in labels

    app.switch_page("views/documents.py").run()
    assert not app.exception
    app.text_input[0].set_value("K8 transshipment").run()
    assert any("K8" in m.value for m in app.markdown)
