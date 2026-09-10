"""Forecast and coastline impact — CLAUDE.md layer 8."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from samudra.attribution import drift
from samudra.geo import Projector, load_polygon
from samudra.impact.coastline import coastline_impacts, load_coastline
from samudra.impact.forecast import forecast

ART = Path("artifacts/demo-001")

pytestmark = pytest.mark.skipif(
    not (ART / "observed_slick.geojson").exists(),
    reason="run: python -m samudra.synth.generate --scenario demo-001",
)


@pytest.fixture(scope="module")
def fcs():
    gj = json.loads((ART / "observed_slick.geojson").read_text())
    slick = load_polygon(gj)
    acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])
    env = drift.EnvField.load(ART / "env.npz")
    return slick, acq, forecast(slick, acq, env, slick_age_hours=8.5, n_particles=400, seed=2)


def test_area_grows_with_horizon(fcs):
    slick, _, f = fcs
    areas = [x["area_km2"] for x in f]
    print("\nareas km2:", [round(a, 1) for a in areas])
    assert areas == sorted(areas), f"forecast area must not shrink: {areas}"
    proj = Projector(slick.centroid.y, slick.centroid.x)
    assert areas[0] > proj.polygon_to_m(slick).area / 1e6


def test_cone_encloses_the_forecast(fcs):
    _, _, f = fcs
    for x in f:
        assert x["cone_area_km2"] > x["area_km2"], (
            f"+{x['horizon_hours']}h cone {x['cone_area_km2']:.1f} does not exceed "
            f"forecast {x['area_km2']:.1f} km2"
        )


def test_slick_actually_moves(fcs):
    slick, _, f = fcs
    proj = Projector(slick.centroid.y, slick.centroid.x)
    ax, ay = proj.to_m(slick.centroid.x, slick.centroid.y)
    bx, by = proj.to_m(f[-1]["centroid_lon"], f[-1]["centroid_lat"])
    moved_km = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / 1000.0
    print(f"\n+{f[-1]['horizon_hours']}h displacement: {moved_km:.1f} km")
    assert moved_km > 5.0, "a 72 h forecast that has not moved means the env field is dead"


def test_coastline_impact_is_reported_either_way(fcs):
    slick, acq, f = fcs
    coast = load_coastline()
    imps = coastline_impacts(f, coast, acq)
    assert len(imps) == len(f)
    for i in imps:
        assert "coastline_intersects" in i
        assert i["coastline_source"]
        if i["coastline_intersects"]:
            assert i["affected_shoreline_km"] > 0
            assert i["coastline_eta"] is not None
        else:
            assert i["distance_to_coast_km"] > 0


def test_coastline_loads_and_covers_the_aoi():
    c = load_coastline()
    minx, miny, maxx, maxy = c["geometry"].bounds
    assert minx < 73.0 < maxx or minx < 70.0 < maxx
    assert miny < 17.5 < maxy
