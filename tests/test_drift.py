"""Drift round-trip test — the silent-bug catcher.

The generator advects the culprit's discharge forward to the acquisition time
using one implementation of the physics. drift.py implements the same physics
independently. If the two agree, the coordinate handling is right in both. If the
IoU comes back near zero, some coordinate conversion is wrong — which is exactly
the failure this test exists to catch, because nothing else downstream would
notice.

This module is one of the only places allowed to read ground_truth.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from samudra.attribution import drift
from samudra.geo import Projector, angular_delta_deg, iou, load_polygon, major_axis_deg
from samudra.timeutil import epoch_seconds

ART = Path("artifacts/demo-001")

pytestmark = pytest.mark.skipif(
    not (ART / "ground_truth.json").exists(),
    reason="run: python -m samudra.synth.generate --scenario demo-001",
)


@pytest.fixture(scope="module")
def gt() -> dict:
    return json.loads((ART / "ground_truth.json").read_text())


@pytest.fixture(scope="module")
def env() -> drift.EnvField:
    return drift.EnvField.load(ART / "env.npz")


@pytest.fixture(scope="module")
def observed():
    return load_polygon(json.loads((ART / "observed_slick.geojson").read_text()))


@pytest.fixture(scope="module")
def culprit_track(gt) -> np.ndarray:
    """(epoch, lat, lon) for the culprit, from ais.parquet — a pipeline input."""
    df = pd.read_parquet(ART / "ais.parquet")
    d = df[df.mmsi == gt["culprit_mmsi"]].sort_values("t")
    return np.column_stack(
        [epoch_seconds(d.t), d.lat.to_numpy(), d.lon.to_numpy()]
    )


def test_forward_roundtrip_matches_observed_slick(gt, env, observed, culprit_track):
    release_at = datetime.fromisoformat(gt["true_release_at"])
    acq = datetime.fromisoformat(gt["acquisition_at"])

    _, _, poly = drift.advect_forward(
        gt["true_release_lat"],
        gt["true_release_lon"],
        release_at,
        acq,
        env,
        n_particles=900,
        track=culprit_track,
        release_duration_min=gt["release_duration_min"],
        seed=7,
    )

    proj = Projector(gt["true_release_lat"], gt["true_release_lon"])
    score = iou(proj, poly, observed)

    om = major_axis_deg(proj.polygon_to_m(observed))
    sm = major_axis_deg(proj.polygon_to_m(poly))
    print(
        f"\nIoU                 : {score:.4f}"
        f"\nsimulated area km2  : {proj.polygon_to_m(poly).area / 1e6:.2f}"
        f"\nobserved  area km2  : {proj.polygon_to_m(observed).area / 1e6:.2f}"
        f"\norientation delta   : {angular_delta_deg(om, sm):.1f} deg"
    )
    assert score > 0.5, f"IoU {score:.3f} — coordinate handling is probably wrong"


def test_reverse_envelope_contains_the_true_release_point(gt, env, observed):
    from shapely.geometry import Point

    acq = datetime.fromisoformat(gt["acquisition_at"])
    hours_back = gt["slick_age_hours"] + 4.0

    steps = drift.advect_reverse(observed, acq, hours_back, env, n_particles=600, seed=3)
    envelope = drift.origin_envelope(steps)

    truth = Point(gt["true_release_lon"], gt["true_release_lat"])
    proj = Projector(gt["true_release_lat"], gt["true_release_lon"])
    print(
        f"\nenvelope area km2   : {proj.polygon_to_m(envelope).area / 1e6:.1f}"
        f"\nsteps               : {len(steps)}"
        f"\ncontains truth      : {envelope.contains(truth)}"
    )
    assert envelope.contains(truth), "origin envelope missed the true release point"


def test_reverse_steps_grow_monotonically(gt, env, observed):
    acq = datetime.fromisoformat(gt["acquisition_at"])
    steps = drift.advect_reverse(observed, acq, 8.0, env, n_particles=400, seed=1)
    proj = Projector(observed.centroid.y, observed.centroid.x)
    areas = [proj.polygon_to_m(s["polygon"]).area for s in steps]
    assert areas[-1] > areas[0], "uncertainty must grow going backwards in time"


def test_forward_rejects_backwards_time(gt, env):
    t = datetime.fromisoformat(gt["acquisition_at"])
    with pytest.raises(ValueError):
        drift.advect_forward(17.0, 70.0, t, t - timedelta(hours=1), env)


def test_wind_deflection_is_clockwise():
    """Northward wind must deflect east. A sign error here ruins attribution."""
    u, v = drift._rotate_cw(0.0, 10.0, 15.0)
    assert u > 0, "clockwise rotation of a northward vector must gain an eastward component"
    assert v == pytest.approx(10.0 * np.cos(np.radians(15.0)), abs=1e-6)
