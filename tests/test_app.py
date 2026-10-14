"""Smoke test the Streamlit dashboard headlessly."""

import pytest

from tests.conftest import ROOT

pytest.importorskip("streamlit")
pytest.importorskip("plotly")


def test_dashboard_renders_and_chats(trained_model, ingested_store):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4

    at.chat_input[0].set_value("Emissions for 500 kg shipped 300 km by truck").run()
    assert not at.exception
    assert any("15.75 kg CO2e" in m.value for m in at.markdown)
