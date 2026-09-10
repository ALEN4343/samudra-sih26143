"""CFAR ship detection against planted targets — CLAUDE.md 5.3, build step 11.

This module is one of the few allowed to read ground_truth.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from samudra.vessels.cfar import cfar_mask, detect, dn_to_linear, match_to_truth

ART = Path("artifacts/demo-001")

pytestmark = pytest.mark.skipif(
    not (ART / "scene.tif").exists(),
    reason="run: python -m samudra.synth.generate --scenario demo-001",
)


@pytest.fixture(scope="module")
def result():
    gt = json.loads((ART / "ground_truth.json").read_text())
    targets, summary = detect(ART / "scene.tif")
    truth = gt["scene"]["planted_vessel_px"]
    return targets, summary, match_to_truth(targets, truth), truth


def test_recovers_the_planted_vessels(result):
    _, _, m, truth = result
    print(
        f"\nplanted {m['planted']}, recovered {m['recovered']} "
        f"(recall {m['recall'] * 100:.0f}%), {m['false_positives']} other detections"
    )
    assert m["planted"] > 0, "scenario has no planted vessel targets"
    assert m["recall"] >= 0.8, (
        f"CFAR recovered only {m['recovered']}/{m['planted']} planted vessels"
    )


def test_false_alarms_stay_bounded(result):
    targets, summary, m, _ = result
    print(f"\nfalse alarm rate {summary['false_alarm_rate']:.2e}, "
          f"{m['false_positives']} spurious targets")
    # Spec fixes k=4.5; the grouping threshold is what keeps speckle out.
    assert m["false_positives"] <= 25, (
        f"{m['false_positives']} spurious targets — grouping is too permissive"
    )


def test_detections_land_close_to_truth(result):
    _, _, m, _ = result
    if m["matches"]:
        worst = max(x["dist_px"] for x in m["matches"])
        assert worst <= 6.0, f"worst matched detection is {worst:.1f} px away"


def test_cfar_runs_in_linear_intensity_not_db():
    """Running CFAR on dB inverts the speckle model and the FAR stops meaning anything."""
    rng = np.random.default_rng(0)
    lin = rng.gamma(shape=3.0, scale=1 / 3.0, size=(256, 256)).astype(np.float32)
    lin[128, 128] = 60.0  # one bright target

    mask, z = cfar_mask(lin, k=4.5)
    assert mask[128, 128], "an obvious bright target must be detected"
    # Background false alarms must stay rare on pure speckle.
    bg = mask.copy()
    bg[120:136, 120:136] = False
    assert bg.mean() < 0.02, f"false alarm rate {bg.mean():.4f} is too high on pure speckle"


def test_guard_band_is_validated():
    lin = np.ones((32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        cfar_mask(lin, guard=12, background=8)


def test_dn_to_linear_round_trips_the_documented_scaling():
    dn = np.array([0, 255], dtype=np.uint8)
    lin = dn_to_linear(dn, -30.0, 0.0)
    assert 10 * np.log10(lin[0]) == pytest.approx(-30.0, abs=1e-4)
    assert 10 * np.log10(lin[1]) == pytest.approx(0.0, abs=1e-4)
