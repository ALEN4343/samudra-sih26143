"""Instantiate every contract model with dummy values.

This is a smoke test: it proves the models load, validate, and round-trip through
JSON. It does not test behaviour.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import yaml

from samudra import contracts as c

T0 = datetime(2025, 3, 14, 6, 30, tzinfo=timezone.utc)

SQUARE = {
    "type": "Polygon",
    "coordinates": [[[70.0, 17.0], [70.1, 17.0], [70.1, 17.1], [70.0, 17.1], [70.0, 17.0]]],
}


def _point(**kw):
    base = dict(mmsi=419001234, t=T0, lat=17.0, lon=70.0, sog=11.4, cog=95.0, heading=96.0)
    base.update(kw)
    return c.AisPoint(**base)


def _slick():
    return c.SlickPolygon(
        slick_id="slick-0",
        geometry=SQUARE,
        area_km2=42.5,
        perimeter_km=31.2,
        centroid_lat=17.05,
        centroid_lon=70.05,
        major_axis_deg=112.0,
        minor_axis_m=1800.0,
        major_axis_m=14200.0,
        eccentricity=0.99,
        shape_complexity=2.4,
        mean_sigma0_db=-24.1,
        cnn_oil_prob=0.91,
        anomaly_z=3.8,
        wind_gate_pass=True,
        confidence=0.83,
    )


def _hypothesis(mmsi=419001234):
    return c.Hypothesis(
        hypothesis_id=f"hyp-{mmsi}-0",
        mmsi=mmsi,
        release_at=T0 - timedelta(hours=9),
        release_lat=16.9,
        release_lon=69.8,
        simulated_geometry=SQUARE,
        iou=0.61,
        centroid_offset_km=2.3,
        orientation_delta_deg=8.0,
        area_ratio=0.88,
        score=0.74,
    )


def test_ais_point_and_track():
    pts = [_point(t=T0 + timedelta(minutes=i)) for i in range(3)]
    gap = c.AisGap(
        start_t=T0,
        end_t=T0 + timedelta(minutes=40),
        duration_min=40.0,
        start_lat=17.0,
        start_lon=70.0,
        end_lat=17.1,
        end_lon=70.2,
    )
    track = c.AisTrack(
        mmsi=419001234,
        points=pts,
        vessel_name="MV DUMMY",
        imo=9123456,
        vessel_type="Tanker",
        length_m=183.0,
        flag="IN",
        gaps=[gap],
    )
    assert len(track.points) == 3
    assert track.gaps[0].duration_min == 40.0


def test_naive_datetime_becomes_utc():
    p = c.AisPoint(mmsi=1, t=datetime(2025, 3, 14, 6, 30), lat=0.0, lon=0.0, sog=0.0)
    assert p.t.tzinfo is timezone.utc


def test_vessel_detection():
    d = c.VesselDetection(det_id="cfar-7", lat=17.0, lon=70.0, length_m=190.0, peak_intensity=8.4)
    assert d.matched_mmsi is None


def test_slick_polygon():
    assert _slick().confidence == pytest.approx(0.83)


def test_env_models():
    field = c.EnvField(
        lats=[17.0, 17.1],
        lons=[70.0, 70.1],
        times=[T0],
        wind_u=[[5.0, 5.1], [5.2, 5.3]],
        wind_v=[[1.0, 1.1], [1.2, 1.3]],
        curr_u=[[0.2, 0.2], [0.2, 0.2]],
        curr_v=[[0.1, 0.1], [0.1, 0.1]],
    )
    summary = c.EnvSummary(
        mean_wind_speed_ms=6.2,
        mean_wind_dir_deg=240.0,
        mean_current_speed_ms=0.25,
        mean_current_dir_deg=190.0,
        wind_gate_pass=True,
    )
    assert len(field.lats) == 2
    assert summary.wind_gate_pass


def test_trust_and_flags():
    ts = c.TrustScore(
        mmsi=419001234,
        score=0.35,
        classification="IDENTITY_MISMATCH",
        trust_flags=[c.Flag(code="POSITION_TELEPORT", severity="CRITICAL", detail="41 km in 62 s", at=T0)],
        behaviour_flags=[c.Flag(code="LOITERING", severity="WARN", detail="0.4 kn for 2.1 h")],
    )
    assert ts.classification == "IDENTITY_MISMATCH"
    assert ts.trust_flags[0].severity == "CRITICAL"


def test_all_classifications_accepted():
    for cls in ("MATCHED", "DARK", "PHANTOM", "IDENTITY_MISMATCH"):
        assert c.TrustScore(mmsi=1, score=1.0, classification=cls).classification == cls


def test_attribution_models():
    suspect = c.Suspect(
        mmsi=419001234,
        vessel_name="MV DUMMY",
        rank=1,
        posterior=0.72,
        best_hypothesis=_hypothesis(),
        top_hypotheses=[_hypothesis()],
        trust_prior=1.3,
        behaviour_prior=1.25,
        rationale=["Inside origin envelope for 4.5 h before acquisition."],
    )
    funnel = c.FunnelCounts(total_in_scene=40, in_envelope=6, scored=6, ranked=6)
    age = c.SlickAge(
        best_hours=9.0, low_hours=8.0, high_hours=11.5, fay_estimate_hours=10.2, agrees_with_fay=True
    )
    assert suspect.rank == 1
    assert funnel.total_in_scene > funnel.in_envelope
    assert age.agrees_with_fay


def test_incident_and_forecast():
    incident = c.Incident(
        incident_id="demo-001",
        acquisition_at=T0,
        aoi_bounds=(68.0, 15.0, 73.0, 20.0),
        observed_slick=_slick(),
        origin_envelope=SQUARE,
        env_summary=c.EnvSummary(
            mean_wind_speed_ms=6.2,
            mean_wind_dir_deg=240.0,
            mean_current_speed_ms=0.25,
            mean_current_dir_deg=190.0,
            wind_gate_pass=True,
        ),
        funnel=c.FunnelCounts(total_in_scene=40, in_envelope=6, scored=6, ranked=6),
    )
    fc = c.ForecastPolygon(
        horizon_hours=48,
        geometry=SQUARE,
        uncertainty_cone=SQUARE,
        area_km2=120.0,
        coastline_intersects=True,
        coastline_eta=T0 + timedelta(hours=41),
        affected_shoreline_km=18.3,
    )
    assert incident.suspects == []
    assert fc.horizon_hours == 48


def test_scenario_roundtrips_through_json():
    s = c.Scenario(
        scenario_id="demo-001",
        culprit_mmsi=419001234,
        true_release_at=T0 - timedelta(hours=9),
        true_release_lat=16.9,
        true_release_lon=69.8,
        acquisition_at=T0,
        aoi_bounds=(68.0, 15.0, 73.0, 20.0),
    )
    assert c.Scenario.model_validate_json(s.model_dump_json()) == s


def test_extra_fields_rejected():
    with pytest.raises(Exception):
        c.AisPoint(mmsi=1, t=T0, lat=0.0, lon=0.0, sog=0.0, bogus_field=1)


def test_weights_yaml_matches_spec():
    with open("config/weights.yaml") as fh:
        cfg = yaml.safe_load(fh)
    w = cfg["score_weights"]
    assert (w["iou"], w["centroid"], w["orientation"], w["area"]) == (0.40, 0.25, 0.25, 0.10)
    assert sum(w.values()) == pytest.approx(1.0)
    assert cfg["drift"]["wind_factor"] == 0.03
    assert cfg["drift"]["wind_deflection_deg"] == 15.0
    assert cfg["drift"]["diffusion_k_m2s"] == 5.0
