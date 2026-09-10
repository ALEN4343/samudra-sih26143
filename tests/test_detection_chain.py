"""Preprocess, polygonize, fuse, baseline, and the detection fallback.

The fallback is the one that matters operationally: CLAUDE.md build order step 11
requires run_demo.sh to work whether or not a checkpoint exists, and a pipeline
that silently changes its input source depending on what is on disk is a demo
that breaks on stage.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from samudra.attribution import drift
from samudra.baseline.anomaly import anomaly_z, wind_bin
from samudra.detection.fuse import fuse, fuse_detection
from samudra.detection.polygonize import mask_to_polygons
from samudra.detection.preprocess import land_mask, scene_db, stitch, tiles, wind_gate
from samudra.geo import Projector

ART = Path("artifacts/demo-001")

needs_scene = pytest.mark.skipif(
    not (ART / "scene.tif").exists(),
    reason="run: python -m samudra.synth.generate --scenario demo-001",
)


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


def test_tiles_cover_every_pixel():
    shape = (1000, 1300)
    seen = np.zeros(shape, dtype=bool)
    for y0, y1, x0, x1 in tiles(shape, 512, 64):
        seen[y0:y1, x0:x1] = True
    assert seen.all(), "tiling left gaps — those pixels would never be scored"


def test_tiles_are_full_size_and_overlap():
    got = list(tiles((1000, 1300), 512, 64))
    assert all((y1 - y0) == 512 and (x1 - x0) == 512 for y0, y1, x0, x1 in got)
    assert len(got) > 1


def test_tile_overlap_must_be_smaller_than_the_tile():
    with pytest.raises(ValueError):
        list(tiles((100, 100), 64, 64))


def test_stitch_averages_overlaps():
    shape = (10, 10)
    pieces = [((0, 6, 0, 10), np.ones((6, 10))), ((4, 10, 0, 10), np.full((6, 10), 3.0))]
    out = stitch(shape, pieces)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[9, 0] == pytest.approx(3.0)
    assert out[5, 0] == pytest.approx(2.0), "overlap must average, not overwrite"


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


@needs_scene
def test_wind_gate_matches_the_spec_band():
    db, transform, bounds = scene_db(ART / "scene.tif")
    gj = json.loads((ART / "observed_slick.geojson").read_text())
    acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])
    env = drift.EnvField.load(ART / "env.npz")

    gate, summ = wind_gate(env, acq.timestamp(), bounds, db.shape)
    assert gate.shape == db.shape
    assert summ["gate_lo"] == 3.0 and summ["gate_hi"] == 10.0
    # The generator holds wind inside 4-9 m/s, so the whole scene should pass.
    assert summ["fraction_in_band"] > 0.99


@needs_scene
def test_land_mask_is_small_but_present_for_this_aoi():
    db, transform, bounds = scene_db(ART / "scene.tif")
    land = land_mask(bounds, db.shape, transform)
    assert land.shape == db.shape
    # The AOI's eastern edge clips the Indian coast; it must not mask the sea.
    assert 0.0 <= land.mean() < 0.2


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_wind_bins_match_the_spec():
    assert wind_bin(1.0) == "0-3"
    assert wind_bin(4.0) == "3-5"
    assert wind_bin(6.9) == "5-7"
    assert wind_bin(9.0) == "7-10"
    assert wind_bin(14.0) == "10+"


@needs_scene
def test_anomaly_falls_back_loudly_without_a_baseline():
    """CLAUDE.md 5.1: below 15 scenes, fall back globally and say so."""
    db, transform, bounds = scene_db(ART / "scene.tif")
    z, info = anomaly_z(db, transform, bounds, 6.7, store=Path("does/not/exist.parquet"))
    assert info["mode"] == "global_fallback"
    assert "fewer than" in info["reason"]
    assert z.shape == db.shape
    # The planted slick is genuinely darker than background.
    assert z.min() < -2.0


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def test_fusion_requires_all_three_signals():
    # Strongly dark, confident CNN, gate open -> high confidence.
    assert fuse(0.9, -4.0, True) > 0.8
    # Same, but the wind gate is shut -> zero, regardless of the other two.
    assert fuse(0.9, -4.0, False) == 0.0
    # Confident CNN but not anomalous (a chronic seep) -> suppressed.
    assert fuse(0.9, 0.0, True) < 0.25
    # Anomalous but the CNN disagrees -> suppressed.
    assert fuse(0.05, -4.0, True) < 0.1


def test_fusion_uses_darkness_not_raw_z():
    """Oil is dark. A bright anomaly must not score as oil."""
    dark = fuse(0.9, -4.0, True)
    bright = fuse(0.9, +4.0, True)
    assert dark > bright * 10, "sign error: bright anomalies are scoring as oil"


def test_fuse_detection_reports_its_terms():
    rec = fuse_detection({"slick_id": "s0", "cnn_oil_prob": 0.8, "area_km2": 10.0}, -3.0, True)
    assert rec["confidence"] == pytest.approx(fuse(0.8, -3.0, True))
    assert rec["wind_gate_pass"] is True
    assert rec["fusion"]["cnn_oil_prob"] == 0.8
    # No look-alike class exists in either training set; claiming one would be a lie.
    assert rec["cnn_lookalike_prob"] == 0.0


# --------------------------------------------------------------------------
# Polygonize
# --------------------------------------------------------------------------


def test_polygonize_extracts_geometry_features():
    import rasterio.transform

    mask = np.zeros((200, 200), dtype=bool)
    mask[80:120, 40:160] = True  # a wide, elongated rectangle
    transform = rasterio.transform.from_bounds(70.0, 17.0, 70.5, 17.5, 200, 200)
    proj = Projector(17.25, 70.25)

    polys = mask_to_polygons(mask, transform, min_area_km2=0.1, proj=proj)
    assert polys, "an obvious rectangle must polygonise"

    from samudra.geo import polygon_metrics

    m = polygon_metrics(proj, polys[0])
    assert m["area_km2"] > 1.0
    assert m["eccentricity"] > 0.8, "an elongated shape must read as eccentric"
    assert 0.0 <= m["major_axis_deg"] < 180.0


def test_polygonize_drops_specks():
    import rasterio.transform

    mask = np.zeros((200, 200), dtype=bool)
    mask[100:102, 100:102] = True
    transform = rasterio.transform.from_bounds(70.0, 17.0, 70.5, 17.5, 200, 200)
    proj = Projector(17.25, 70.25)
    assert mask_to_polygons(mask, transform, min_area_km2=5.0, proj=proj) == []


# --------------------------------------------------------------------------
# The fallback
# --------------------------------------------------------------------------


@needs_scene
def test_pipeline_falls_back_when_no_detection_output(tmp_path):
    from samudra.attribution.__main__ import resolve_observed_slick

    d = tmp_path / "inc"
    d.mkdir()
    (d / "observed_slick.geojson").write_text(
        (ART / "observed_slick.geojson").read_text()
    )
    poly, acq, src = resolve_observed_slick(d)
    assert src["source"] == "synthetic"
    assert poly.area > 0


@needs_scene
def test_pipeline_refuses_a_non_representative_checkpoint(tmp_path):
    """A smoke-test checkpoint must not silently replace a known-good input."""
    from samudra.attribution.__main__ import resolve_observed_slick

    d = tmp_path / "inc"
    d.mkdir()
    (d / "observed_slick.geojson").write_text((ART / "observed_slick.geojson").read_text())
    gj = json.loads((ART / "observed_slick.geojson").read_text())
    (d / "detected_slicks.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": gj["features"][0]["geometry"],
                        "properties": {"slick_id": "slick-000", "confidence": 0.9},
                    }
                ],
            }
        )
    )
    (d / "detection_meta.json").write_text(json.dumps({"representative": False}))

    _, _, auto = resolve_observed_slick(d, "auto")
    assert auto["source"] == "synthetic"
    assert "NOT representative" in auto["reason"]

    _, _, forced = resolve_observed_slick(d, "always")
    assert forced["source"] == "detected"

    (d / "detection_meta.json").write_text(json.dumps({"representative": True}))
    _, _, good = resolve_observed_slick(d, "auto")
    assert good["source"] == "detected", "a real checkpoint must actually be used"
