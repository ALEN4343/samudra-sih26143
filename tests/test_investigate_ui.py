"""Investigator view (/investigate) — a redesigned copy of the / page.

Text-level checks on the page source plus the three additive API endpoints it
relies on. The point of most of these is the same as tests/test_ops_ui.py: the
redesign must not quietly start hardcoding results, claiming guilt, or
inventing imagery.
"""

import asyncio
import re
from pathlib import Path

import pytest
from fastapi import HTTPException

from samudra import api

WEB = Path("web/investigate")
pytestmark = pytest.mark.skipif(not WEB.exists(), reason="web/investigate not present")

# Endpoints are called as plain functions: starlette's TestClient needs an extra
# package (httpx2) that this project does not otherwise depend on.


@pytest.fixture(scope="module")
def src() -> str:
    return "\n".join((WEB / f).read_text(encoding="utf-8")
                     for f in ("index.html", "app.js", "style.css"))


def _ready():
    return [i for i in api.list_incidents() if i.get("ready")]


def _incident_with_suspects():
    for i in _ready():
        if i.get("suspects"):
            return i["incident_id"]
    pytest.skip("no analysed incident with suspects on disk")


def test_original_page_is_untouched_and_still_served():
    assert Path(api.index().path).name == "index.html"
    assert "Ranked suspects" in Path("web/index.html").read_text(encoding="utf-8")


def test_investigate_route_serves_the_copy():
    f = Path(api.investigate().path)
    assert f == WEB / "index.html" and "SAMUDRA" in f.read_text(encoding="utf-8")


def test_no_guilt_language(src):
    low = src.lower()
    for bad in ("culprit", "guilty", "confirmed polluter", "ranked suspects"):
        assert bad not in low, bad


def test_responsible_language_present(src):
    assert "Potential source vessel" in src
    assert "does not establish legal responsibility" in src
    assert "not a probability of responsibility" in src


def test_score_weights_are_not_hardcoded(src):
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "/api/config/scoring" in js
    # The old pages printed the weights as literals next to each term.
    assert not re.search(r"×\s*\.?(40|25|10)\b", js)


def test_forecast_is_never_interpolated(src):
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "never" in js and "interpolat" in js
    assert "S.horizons.filter(x => x <= h" in js   # snap to the computed horizon


def test_missing_imagery_is_stated_not_invented(src):
    assert "Satellite evidence layer unavailable for this scenario" in src
    assert "No imagery is substituted" in src


def test_synthetic_mode_is_labelled(src):
    assert "Simulation mode" in src and "Synthetic exercise" in src


def test_age_crosscheck_is_not_called_independent(src):
    # CLAUDE.md 10.3: the Fay estimate depends on an assumed volume.
    assert "independent age validation" not in src.lower()
    assert "not independent confirmation" in src


def test_scoring_config_endpoint_matches_weights_file():
    import yaml
    cfg = yaml.safe_load(Path("config/weights.yaml").read_text())
    got = api.get_scoring_config()
    assert got["score_weights"] == cfg["score_weights"]
    assert got["temperature"] == cfg["priors"]["temperature"]


def test_replay_accepts_any_ranked_candidate():
    iid = _incident_with_suspects()
    inc = api.get_incident(iid)
    default = api.get_replay(iid, n_frames=4)
    assert default["mmsi"] == inc["suspects"][0]["mmsi"]
    last = inc["suspects"][-1]
    r = api.get_replay(iid, n_frames=4, mmsi=last["mmsi"])
    assert r["mmsi"] == last["mmsi"]
    assert r["release_at"] == last["best_hypothesis"]["release_at"]
    with pytest.raises(HTTPException) as e:
        api.get_replay(iid, mmsi=1)
    assert e.value.status_code == 404


async def _body(resp) -> bytes:
    return b"".join([c async for c in resp.body_iterator])


def test_scene_endpoint_reports_availability_honestly():
    for i in _ready():
        iid = i["incident_id"]
        m = api.get_scene_meta(iid)
        if not (Path("artifacts") / iid / "scene.tif").exists():
            assert m["available"] is False and m["reason"]
            with pytest.raises(HTTPException) as e:
                api.get_scene_png(iid)
            assert e.value.status_code == 404
        elif m["available"]:
            png = asyncio.run(_body(api.get_scene_png(iid, max_px=256)))
            assert png[:4] == bytes([0x89]) + b"PNG"
            (s, w), (n, e) = m["bounds"]
            assert s < n and w < e
            if m["synthetic"]:
                assert "not a satellite acquisition" in m["source"]
