"""Layer 11 — alerting. CLAUDE.md 5.8.

The thing worth testing here is not the message wording, it is that the tier
decides the distribution list and that the distribution list is not simply the
N nearest pins: a Tier II alert must reach a regional HQ even when six local
stations are closer, and a Tier I alert must never reach Coast Guard HQ.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from samudra.contracts import Alert
from samudra.dissemination import alert as A


@pytest.fixture
def stations():
    return A.load_stations()["stations"]


def _incident(area_km2: float, incident_id: str = "t-001") -> dict:
    acq = datetime(2026, 1, 14, 6, 30, tzinfo=timezone.utc)
    return {
        "incident_id": incident_id,
        "acquisition_at": acq.isoformat(),
        "observed_slick": {
            "area_km2": area_km2,
            "centroid_lat": 16.12,
            "centroid_lon": 69.72,
            "major_axis_deg": 41.9,
            "eccentricity": 0.985,
        },
        "env_summary": {
            "mean_wind_speed_ms": 6.7,
            "mean_wind_dir_deg": 91.6,
            "mean_current_speed_ms": 0.23,
            "mean_current_dir_deg": 131.3,
            "wind_gate_pass": True,
        },
        "slick_age": {"best_hours": 8.5, "low_hours": 8.0, "high_hours": 9.0},
        "suspects": [{
            "mmsi": 461279535, "vessel_name": "PACIFIC PIONEER",
            "vessel_type": "Bulk Carrier", "flag": "OM",
            "posterior": 0.715, "classification": "MATCHED",
        }],
    }


def _forecast(shore_in_hours: float | None) -> dict:
    acq = datetime(2026, 1, 14, 6, 30, tzinfo=timezone.utc)
    out = {}
    for h in (24, 48, 72):
        hit = shore_in_hours is not None and shore_in_hours <= h
        props = {
            "kind": "forecast", "horizon_hours": h,
            "area_km2": 150.0 + h, "coastline_intersects": hit,
            "affected_shoreline_km": 12.0 if hit else 0.0,
            "distance_to_coast_km": 0.0 if hit else 300.0,
            "coastline_eta": (acq + timedelta(hours=shore_in_hours)).isoformat()
                             if hit else None,
        }
        out[f"forecast_{h}h"] = {"type": "FeatureCollection",
                                 "features": [{"type": "Feature", "properties": props,
                                               "geometry": None}]}
    return out


# ------------------------------------------------------------------ tiering

def test_small_offshore_slick_is_tier_1():
    cls = A.classify(_incident(8.0), _forecast(None), A._tiering())
    assert cls["nosdcp_tier"] == 1
    assert cls["severity"] == "ADVISORY"


def test_area_alone_escalates_to_tier_2():
    cls = A.classify(_incident(89.8), _forecast(None), A._tiering())
    assert cls["nosdcp_tier"] == 2
    assert "89.8" in cls["reasons"][0]


def test_large_slick_is_tier_3():
    cls = A.classify(_incident(400.0), _forecast(None), A._tiering())
    assert cls["nosdcp_tier"] == 3
    assert cls["severity"] == "EMERGENCY"


def test_imminent_shore_contact_escalates_a_small_slick():
    """A 10 km2 slick 12 hours off a beach is a bigger problem than a 100 km2
    slick 300 km offshore. Tier has to reflect that, not just area."""
    small_offshore = A.classify(_incident(10.0), _forecast(None), A._tiering())
    small_inshore = A.classify(_incident(10.0), _forecast(12.0), A._tiering())
    assert small_offshore["nosdcp_tier"] == 1
    assert small_inshore["nosdcp_tier"] == 3
    assert small_inshore["shore_contact"] is True


def test_distant_shore_contact_is_tier_2_not_3():
    cls = A.classify(_incident(10.0), _forecast(60.0), A._tiering())
    assert cls["nosdcp_tier"] == 2


# ------------------------------------------------------- distribution list

def test_tier_1_never_reaches_national_hq(stations):
    r = A.nearest_stations(16.12, 69.72, 1, 6, stations)
    assert all(s["role"] != "NATIONAL_HQ" for s in r)
    assert r, "a Tier I alert still has to reach someone"


def test_tier_2_reserves_slots_for_command_a_distance_sort_would_lose(stations):
    """Gulf of Kutch: five ICG units sit within ~100 km, the regional HQ is ~400.

    A plain nearest-six would be six local stations and no command unit, so the
    alert would never reach anyone who can mobilise a response. The two-list
    merge is what prevents that, and this is the position that proves it.
    """
    lat, lon, limit = 22.5, 69.0, 4
    eligible = [s for s in stations if s.get("nosdcp_tier_min", 1) <= 2]
    by_distance = sorted(
        eligible, key=lambda s: A.haversine_km(lat, lon, s["lat"], s["lon"])
    )[:limit]
    assert all(s["role"] != "REGION_HQ" for s in by_distance), "premise changed"

    r = A.nearest_stations(lat, lon, 2, limit, stations)
    assert any(s["role"] == "REGION_HQ" for s in r)
    assert any(s["role"] in ("STATION", "DISTRICT_HQ") for s in r)


def test_command_unit_for_the_owning_region_comes_first(stations):
    """Kutch is North West water: Gandhinagar outranks Mumbai there even though
    both are on the list."""
    r = A.nearest_stations(22.5, 69.0, 2, 6, stations)
    hqs = [s for s in r if s["role"] == "REGION_HQ"]
    assert hqs and hqs[0]["region"] == "North West"


def test_tier_3_reaches_coast_guard_headquarters(stations):
    r = A.nearest_stations(16.12, 69.72, 3, 6, stations)
    assert any(s["role"] == "NATIONAL_HQ" for s in r)


def test_bay_of_bengal_incident_is_addressed_to_the_east_coast(stations):
    """A spill off Visakhapatnam must not be handed to Gujarat."""
    r = A.nearest_stations(17.2, 84.0, 2, 6, stations)
    assert r[0]["region"] in ("East", "North East")
    assert all(s["region"] != "North West" for s in r)


def test_recipient_count_is_capped(stations):
    assert len(A.nearest_stations(16.12, 69.72, 3, 4, stations)) <= 4


# --------------------------------------------------------------- end to end

def test_build_writes_a_contract_valid_alert(tmp_path):
    d = tmp_path / "t-001"
    d.mkdir()
    (d / "incident.json").write_text(json.dumps(_incident(89.8)))
    for k, v in _forecast(None).items():
        (d / f"{k}.geojson").write_text(json.dumps(v))

    a = A.build("t-001", tmp_path)

    assert (d / "alert.json").exists()
    Alert.model_validate({k: v for k, v in a.items()})   # Part B contract
    assert a["transmitted"] is False
    assert a["severity"] == "ALERT"
    assert "PACIFIC PIONEER" in a["message"]
    assert "NOT AN ACCUSATION" in a["message"]
    assert a["attachment"] == "dossier_t-001.pdf"


def test_message_states_no_shore_contact_when_there_is_none(tmp_path):
    d = tmp_path / "t-002"
    d.mkdir()
    (d / "incident.json").write_text(json.dumps(_incident(30.0, "t-002")))
    for k, v in _forecast(None).items():
        (d / f"{k}.geojson").write_text(json.dumps(v))
    a = A.build("t-002", tmp_path)
    assert "No shoreline contact within 72 h" in a["message"]


def test_stations_geojson_marks_only_the_addressed_units(tmp_path):
    d = tmp_path / "t-003"
    d.mkdir()
    (d / "incident.json").write_text(json.dumps(_incident(89.8, "t-003")))
    for k, v in _forecast(None).items():
        (d / f"{k}.geojson").write_text(json.dumps(v))
    a = A.build("t-003", tmp_path)

    fc = A.stations_geojson(a)
    marked = [f for f in fc["features"] if f["properties"]["addressed"]]
    assert len(marked) == len(a["recipients"])
    assert all(f["properties"]["distance_km"] is not None for f in marked)
    # and an un-addressed station carries no range/bearing to a slick it is not on
    other = [f for f in fc["features"] if not f["properties"]["addressed"]][0]
    assert other["properties"]["distance_km"] is None


# ------------------------------------------------------------------- geometry

def test_bearing_and_range_are_reciprocal_sane():
    # Ratnagiri to the demo-001 slick: roughly west-north-west of the slick.
    d = A.haversine_km(16.988, 73.300, 16.125, 69.724)
    assert 350 < d < 450
    b = A.bearing_deg(16.988, 73.300, 16.125, 69.724)
    assert 230 < b < 280
    assert A.compass(b) in ("WSW", "SW", "W")


def test_dms_formats_a_maritime_position():
    assert A.dms(16.125027, True) == "16°07.50'N"
    assert A.dms(69.724239, False) == "69°43.45'E"
