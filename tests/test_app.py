"""Smoke tests for the Streamlit UI (app.py) using Streamlit's AppTest."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(at: AppTest) -> AppTest:
    at.run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_with_defaults():
    at = _run(AppTest.from_file(APP))
    assert at.title[0].value.endswith("Campus Delivery Robot")
    assert at.sidebar.selectbox(key="policy").value == "bayes_mdp"
    assert len(at.tabs) == 5


def test_switch_map_and_policy():
    at = _run(AppTest.from_file(APP))
    at.sidebar.selectbox(key="map").set_value("Classic 5x5")
    at.sidebar.selectbox(key="policy").set_value("naive")
    _run(at)
    at.checkbox(key="compare_naive").check()
    _run(at)


def test_timeline_and_ground_truth():
    at = _run(AppTest.from_file(APP))
    at.toggle(key="truth").set_value(True)
    at.slider(key="step").set_value(2)
    _run(at)
    at.button(key="next").click()
    _run(at)
    assert at.slider(key="step").value == 3


def test_bayes_tab_handles_bad_input():
    at = _run(AppTest.from_file(APP))
    at.text_input(key="bayes_readings").input("A X Q")
    _run(at)
    assert any("Unknown readings" in e.value for e in at.error)
    at.text_input(key="bayes_readings").input("A A Q A")
    at.toggle(key="bayes_dyn").set_value(True)
    _run(at)


def test_mdp_tab_without_detour():
    at = _run(AppTest.from_file(APP))
    at.selectbox(key="mdp_extra").set_value(None)
    at.radio(key="mdp_cell").set_value("Ordinary cell")
    _run(at)


def test_small_experiment_run():
    at = _run(AppTest.from_file(APP))
    at.slider(key="episodes").set_value(50)
    _run(at)
    at.button(key="run_compare").click()
    _run(at)
    assert at.success, "comparison should report the best policy"
