"""Every contract model instantiates, and Part A matches CLAUDE.md section 4.

The second half is the one that matters: it parses the spec out of CLAUDE.md and
compares it to the code, so the contract and its implementation cannot drift
apart silently.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from samudra import contracts as c

T0 = datetime(2026, 1, 14, 6, 30, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 14, 7, 30, tzinfo=timezone.utc)
POLY = {"type": "Polygon", "coordinates": [[[70.0, 17.0], [70.1, 17.0], [70.1, 17.1], [70.0, 17.0]]]}


# --------------------------------------------------------------------------
# Part A — the contract
# --------------------------------------------------------------------------


def test_scene():
    s = c.Scene(scene_id="S1A_001", sensor="S1_GRD", acquired_at=T0,
                footprint=POLY, raster_path=Path("artifacts/demo-001/scene.tif"),
                incidence_angle_deg=38.4)
    assert s.sensor == "S1_GRD"


def test_scene_rejects_unknown_sensor():
    with pytest.raises(Exception):
        c.Scene(scene_id="x", sensor="LANDSAT", acquired_at=T0, footprint=POLY,
                raster_path=Path("x.tif"))


def test_env_sample():
    e = c.EnvSample(lat=17.0, lon=70.0, t=T0, wind_speed_ms=6.7,
                    wind_dir_deg=272.0, current_u_ms=0.17, current_v_ms=-0.15)
    assert e.wind_speed_ms == 6.7


def test_slick_detection():
    d = c.SlickDetection(
        slick_id="slick-1", scene_id="S1A_001", polygon=POLY,
        area_km2=89.8, perimeter_km=64.2, shape_complexity=1.91,
        eccentricity=0.985, major_axis_deg=42.0, mean_sigma0_db=-24.1,
        contrast_db=10.2, edge_gradient=0.42, cnn_oil_prob=0.93,
        cnn_lookalike_prob=0.04, baseline_anomaly_z=3.8,
        wind_gate_pass=True, confidence=0.88,
    )
    assert d.area_km2 == 89.8


def test_slick_detection_bounds_probabilities():
    for bad in ({"cnn_oil_prob": 1.4}, {"confidence": -0.1}, {"major_axis_deg": 200.0}):
        kw = dict(
            slick_id="s", scene_id="sc", polygon=POLY, area_km2=1.0, perimeter_km=1.0,
            shape_complexity=1.0, eccentricity=0.5, major_axis_deg=10.0,
            mean_sigma0_db=-20.0, contrast_db=8.0, edge_gradient=0.3,
            cnn_oil_prob=0.5, cnn_lookalike_prob=0.1, baseline_anomaly_z=2.0,
            wind_gate_pass=True, confidence=0.5,
        )
        kw.update(bad)
        with pytest.raises(Exception):
            c.SlickDetection(**kw)


def test_ais_point_and_track():
    p = c.AisPoint(mmsi=461279535, t=T0, lat=17.0, lon=70.0, sog=12.4, cog=220.0, heading=221.0)
    t = c.AisTrack(mmsi=461279535, imo=9123456, name="PACIFIC PIONEER", ship_type=70,
                   flag_mid=461, declared_length_m=210.0, points=[p], gaps=[(T0, T1)])
    assert t.points[0].sog == 12.4
    assert t.gaps[0][1] == T1


def test_ais_point_rejects_impossible_position():
    with pytest.raises(Exception):
        c.AisPoint(mmsi=1, t=T0, lat=95.0, lon=70.0, sog=1.0)


def test_sar_ship():
    s = c.SarShip(target_id="cfar-12", scene_id="S1A_001", lat=17.0, lon=70.0,
                  est_length_m=214.0, peak_sigma0_db=4.2)
    assert s.est_length_m == 214.0


def test_trust_score_uses_string_flags():
    """Section 4 types the flag lists as list[str], not structured objects."""
    t = c.TrustScore(mmsi=1, scene_id="S1A_001", score=0.53,
                     trust_flags=["SOG_MISMATCH", "POSITION_TELEPORT"],
                     behaviour_flags=["LOITERING"], sar_corroborated=False,
                     classification="IDENTITY_MISMATCH")
    assert t.trust_flags == ["SOG_MISMATCH", "POSITION_TELEPORT"]


def test_release_hypothesis_and_suspect():
    h = c.ReleaseHypothesis(
        mmsi=461279535, release_at=T0, release_lat=16.2, release_lon=69.7,
        simulated_polygon=POLY, iou=0.798, centroid_offset_km=0.8,
        orientation_delta_deg=0.6, area_ratio=0.99, geometric_score=0.898,
    )
    s = c.Suspect(mmsi=461279535, name="PACIFIC PIONEER", best_hypothesis=h,
                  likelihood=0.72, trust_prior=1.0, behaviour_prior=1.0,
                  type_risk_prior=1.25, gap_coincidence=False, prior_offences=0,
                  posterior=0.715, rationale=["overlaps 80% of the observed slick"])
    assert s.best_hypothesis.geometric_score == 0.898
    assert s.posterior == 0.715


def test_slick_age():
    a = c.SlickAge(incident_id="demo-001", estimated_range_hours=(8.0, 9.0),
                   fay_spreading_estimate_hours=8.5, agreement=True)
    assert a.estimated_range_hours == (8.0, 9.0)


def test_scenario():
    s = c.Scenario(scenario_id="demo-001", culprit_mmsi=461279535,
                   true_release_at=T0, true_release_lat=16.2, true_release_lon=69.7,
                   acquisition_at=T1, aoi_bounds=(68.0, 15.0, 73.0, 20.0))
    assert s.culprit_mmsi == 461279535


# --------------------------------------------------------------------------
# Part B — this build's additions
# --------------------------------------------------------------------------


def test_part_b_models_instantiate():
    flag = c.Flag(code="AIS_GAP", severity="WARN", detail="silent for 42 minutes",
                  at=T0, lat=17.0, lon=70.0)
    gap = c.AisGap(start_t=T0, end_t=T1, duration_min=60.0, start_lat=17.0,
                   start_lon=70.0, end_lat=17.1, end_lon=70.1)
    env = c.EnvField(lats=[15.0, 16.0], lons=[69.0, 70.0], times=[T0],
                     wind_u=[[1.0, 2.0]], wind_v=[[0.5, 0.6]],
                     curr_u=[[0.1, 0.2]], curr_v=[[0.0, 0.1]])
    summ = c.EnvSummary(mean_wind_speed_ms=6.7, mean_wind_dir_deg=92.0,
                        mean_current_speed_ms=0.23, mean_current_dir_deg=131.0,
                        wind_gate_pass=True)
    fun = c.FunnelCounts(total_in_scene=24, in_envelope=3, scored=3, ranked=3)
    fc = c.ForecastPolygon(horizon_hours=24, geometry=POLY, uncertainty_cone=POLY,
                           area_km2=158.8, coastline_intersects=False)
    assert flag.severity == "WARN"
    assert gap.duration_min == 60.0
    assert len(env.lats) == 2 and summ.wind_gate_pass
    assert fun.in_envelope == 3 and fc.horizon_hours == 24


def test_incident_envelope():
    inc = c.Incident(
        incident_id="demo-001", acquisition_at=T0, aoi_bounds=(68.0, 15.0, 73.0, 20.0),
        synthetic=True,
        observed_slick=c.SlickDetection(
            slick_id="s", scene_id="sc", polygon=POLY, area_km2=89.8, perimeter_km=64.2,
            shape_complexity=1.91, eccentricity=0.985, major_axis_deg=42.0,
            mean_sigma0_db=-24.1, contrast_db=10.2, edge_gradient=0.42,
            cnn_oil_prob=0.93, cnn_lookalike_prob=0.04, baseline_anomaly_z=3.8,
            wind_gate_pass=True, confidence=0.88),
        origin_envelope=POLY,
        env_summary=c.EnvSummary(mean_wind_speed_ms=6.7, mean_wind_dir_deg=92.0,
                                 mean_current_speed_ms=0.23, mean_current_dir_deg=131.0,
                                 wind_gate_pass=True),
        funnel=c.FunnelCounts(total_in_scene=24, in_envelope=3, scored=3, ranked=3),
        suspects=[],
        slick_age=c.SlickAge(incident_id="demo-001", estimated_range_hours=(8.0, 9.0),
                             fay_spreading_estimate_hours=8.5, agreement=True),
    )
    assert inc.synthetic is True


def test_extra_fields_are_rejected():
    """Typos must fail loudly rather than being silently absorbed."""
    with pytest.raises(Exception):
        c.Scenario(scenario_id="x", culprit_mmsi=1, true_release_at=T0,
                   true_release_lat=1.0, true_release_lon=1.0, acquisition_at=T1,
                   aoi_bounds=(0, 0, 1, 1), typo_field=True)


# --------------------------------------------------------------------------
# The contract and its implementation must not drift apart
# --------------------------------------------------------------------------


def _spec_models() -> dict[str, set[str]]:
    """Parse the model names and field names out of CLAUDE.md section 4."""
    text = Path("CLAUDE.md").read_text(encoding="utf-8")
    block = text.split("## 4. Data contracts")[1].split("\n## ")[0]
    out: dict[str, set[str]] = {}
    current = None
    for line in block.splitlines():
        m = re.match(r"\s*class (\w+)\(BaseModel\):", line)
        if m:
            current = m.group(1)
            out[current] = set()
            continue
        if current is None or not line.startswith("    "):
            continue
        body = line.split("#", 1)[0]
        # Handles both "lat: float" and "lat: float; lon: float" on one line.
        for part in body.split(";"):
            fm = re.match(r"\s*([a-z_][a-z0-9_]*)\s*:", part)
            if fm:
                out[current].add(fm.group(1))
    return out


def test_part_a_matches_claude_md_section_4():
    spec = _spec_models()
    assert spec, "could not parse section 4 out of CLAUDE.md"

    missing_models = [n for n in spec if not hasattr(c, n)]
    assert not missing_models, f"contracts.py is missing spec models: {missing_models}"

    problems = []
    for name, fields in spec.items():
        actual = set(getattr(c, name).model_fields)
        absent = fields - actual
        if absent:
            problems.append(f"{name} is missing fields {sorted(absent)}")
        extra = actual - fields
        if extra:
            problems.append(f"{name} has fields not in the contract: {sorted(extra)}")
    assert not problems, "contracts.py has drifted from CLAUDE.md section 4:\n  " + "\n  ".join(problems)
