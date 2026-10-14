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

    # Quote tab renders a price for the default selection
    assert any(m.label == "Quoted price" and m.value.startswith("RM") for m in at.metric)

    at.chat_input[0].set_value("Quote a 40ft container to Shah Alam").run()
    assert not at.exception
    assert any("Price: RM" in m.value for m in at.markdown)
