"""Operations console — navigation, layer defaults and language discipline.

These parse web/ops.html as text rather than driving a browser. That is enough
to catch the regressions that actually happen: someone adds `checked` to a layer
so their own demo is quicker to set up, or writes "caused the spill" in a
template. Both are one-word changes that a screenshot review would miss.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

OPS = Path("web/ops.html")
pytestmark = pytest.mark.skipif(not OPS.exists(), reason="web/ops.html not present")


@pytest.fixture(scope="module")
def html() -> str:
    return OPS.read_text(encoding="utf-8")


# --------------------------------------------------------------- navigation

def test_top_nav_has_every_page(html):
    for page in ("dashboard", "satellite", "spill", "drift", "ais", "evidence", "help"):
        assert f'data-p="{page}"' in html, f"navbar is missing {page}"


def test_every_nav_target_has_a_page_container(html):
    for page in re.findall(r'data-p="([a-z]+)"', html):
        assert f'id="p_{page}"' in html, f"nav points at {page} with no page div"


def test_there_is_no_permanent_application_sidebar(html):
    """Part 1: contextual panels only. A `.ctx` belongs to one page at a time."""
    ctx = re.findall(r'class="ctx" id="ctx_([a-z]+)"', html)
    assert set(ctx) <= {"dashboard", "spill", "drift", "ais"}
    assert "ctx_satellite" not in html and "ctx_help" not in html


# ------------------------------------------------------------ layer defaults

def test_every_map_layer_is_off_by_default(html):
    """Part 3 / 15: the base map is visible and nothing else until asked."""
    inputs = re.findall(r'<input\b[^>]*\bdata-l="[a-z]+"[^>]*>', html)
    assert inputs, "no layer checkboxes found"
    assert len(inputs) >= 6
    for tag in inputs:
        assert "checked" not in tag, f"layer starts enabled: {tag}"


def test_layer_list_covers_the_required_set(html):
    found = set(re.findall(r'data-l="([a-z]+)"', html))
    for required in ("slick", "tracks", "ships", "forecast", "sat", "envelope"):
        assert required in found, f"missing layer control: {required}"


def test_layers_are_not_added_to_the_map_at_construction(html):
    """A layer group must be created detached; only a toggle may attach it."""
    m = re.search(r"LAYERS\[k\]\s*=\s*\{[^\n]*\}", html)
    assert m, "layer construction line not found"
    assert "addTo(map)" not in m.group(0), \
        "a layer attaches itself at construction, so it would be on at load"


# ------------------------------------------------------------- no Winds panel

def test_no_standalone_winds_navigation_item(html):
    """Part 2: wind stays in the backend and appears only as context."""
    assert 'data-p="wind"' not in html and 'data-p="winds"' not in html
    nav = html.split('<div id="tabs">')[1].split("</div>")[0]
    assert "wind" not in nav.lower()


def test_wind_still_appears_as_contextual_information(html):
    """Removing the panel must not remove the variable from the picture."""
    assert "mean_wind_speed_ms" in html
    assert "wind gate" in html.lower()


# --------------------------------------------------------- language discipline

def test_no_causal_attribution_language(html):
    """Part 13. A causal phrase is allowed ONLY inside a denial.

    The naive "this substring must not appear" test fails on the very sentence
    that does the right thing — "no vessel is asserted to have caused the
    spill" — so it checks the context instead: every occurrence must sit inside
    a negation, and phrases that cannot be rescued by one are banned outright.
    """
    low = html.lower()
    for phrase in ("guilty", "the polluter is", "proves that the vessel"):
        assert phrase not in low, f"causal language in the UI: {phrase}"

    negations = ("no vessel", "not asserted", "never", "is not", "rather than",
                 "correlation is not causation", "without")
    for phrase in ("caused the spill", "responsible for the spill",
                   "caused this discharge"):
        for m in re.finditer(re.escape(phrase), low):
            window = low[max(0, m.start() - 140):m.end()]
            assert any(n in window for n in negations), (
                f"'{phrase}' appears without a negation near it: "
                f"...{low[max(0, m.start() - 90):m.end() + 20]}..."
            )


def test_candidate_language_is_present(html):
    low = html.lower()
    assert "high-correlation candidate" in low or "candidate vessels associated" in low
    assert "correlation is not causation" in low


def test_detections_are_called_anomalies_not_confirmed_oil(html):
    assert "OIL-LIKE SURFACE ANOMALY" in html
    assert "not a confirmed oil detection" in html


def test_synthetic_mode_is_labelled(html):
    assert "SYNTHETIC DEMO" in html
    assert "not a satellite observation" in html.lower()


def test_live_unavailable_message_exists(html):
    assert "LIVE/NRT UNAVAILABLE" in html


# ------------------------------------------------------------- provenance UI

def test_provenance_panel_asks_the_judge_question(html):
    assert "where did this image come from" in html.lower()
    for field in ("Satellite", "Sensor", "Product", "Acquisition", "Source", "Checksum"):
        assert f"['{field}'" in html or f'"{field}"' in html or f">{field}<" in html \
            or f"['{field}'," in html, f"provenance is missing {field}"


def test_missing_acquisition_time_is_surfaced_not_hidden(html):
    assert "NOT PUBLISHED BY SOURCE" in html


def test_operator_anchor_is_declared_as_operator_input(html):
    low = html.lower()
    assert "operator input" in low
    assert "no coordinate reference system" in low


# ------------------------------------------------------------ other pages

def test_help_page_states_what_is_real_and_what_is_not(html):
    assert "What is real and what is not" in html
    assert "BHOONIDHI_USER" in html


def test_ops_page_refuses_to_run_from_the_filesystem(html):
    assert "must be served by the API" in html
