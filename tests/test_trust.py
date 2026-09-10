"""Trust and behaviour layer against planted anomalies.

The generator plants a known set: three AIS gaps, one spoofed segment, two
loiterers, one sharp slowdown. Layer 5 must find all of them, and must not
invent flags on the other forty vessels.

Precision matters as much as recall here. A trust layer that flags everything
is useless to an investigator, and false flags on the synthetic data usually
mean the *generator* produced physically incoherent tracks rather than that the
detector is clever.

This module is one of the only places allowed to read ground_truth.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from samudra.attribution.prune import load_tracks
from samudra.geo import Projector
from samudra.trust.score import priors_from, score_tracks

ART = Path("artifacts/demo-001")

pytestmark = pytest.mark.skipif(
    not (ART / "ground_truth.json").exists(),
    reason="run: python -m samudra.synth.generate --scenario demo-001",
)


@pytest.fixture(scope="module")
def gt() -> dict:
    return json.loads((ART / "ground_truth.json").read_text())


@pytest.fixture(scope="module")
def scored(gt):
    cfg = yaml.safe_load(open("config/weights.yaml"))
    tracks = load_tracks(ART / "ais.parquet")
    proj = Projector.for_bounds(gt["aoi_bounds"])
    return score_tracks(tracks, cfg, proj), cfg


def _codes(s: dict) -> set[str]:
    return {f["code"] for f in s["trust_flags"]} | {f["code"] for f in s["behaviour_flags"]}


def test_spoofed_vessel_is_identity_mismatch(scored, gt):
    s, _ = scored
    for mmsi in gt["planted_anomalies"]["spoofed"]:
        assert "POSITION_TELEPORT" in _codes(s[mmsi]), f"{mmsi}: {_codes(s[mmsi])}"
        assert s[mmsi]["classification"] == "IDENTITY_MISMATCH"


def test_loiterers_are_flagged(scored, gt):
    s, _ = scored
    for mmsi in gt["planted_anomalies"]["loiter"]:
        assert "LOITERING" in _codes(s[mmsi]), f"{mmsi}: {_codes(s[mmsi])}"


def test_ais_gaps_are_flagged(scored, gt):
    s, _ = scored
    for mmsi in gt["planted_anomalies"]["ais_gap"]:
        assert "AIS_GAP" in _codes(s[mmsi]), f"{mmsi}: {_codes(s[mmsi])}"


def test_slowdown_is_flagged(scored, gt):
    s, _ = scored
    for mmsi in gt["planted_anomalies"]["slowdown"]:
        assert "SPEED_REDUCTION" in _codes(s[mmsi]), f"{mmsi}: {_codes(s[mmsi])}"


def test_no_false_positives(scored, gt):
    """Only the planted vessels carry flags."""
    s, _ = scored
    planted = set(sum(gt["planted_anomalies"].values(), []))
    flagged = {m for m, v in s.items() if v["trust_flags"] or v["behaviour_flags"]}
    spurious = flagged - planted
    assert not spurious, (
        f"{len(spurious)} unplanted vessel(s) flagged: "
        + "; ".join(f"{m}={sorted(_codes(s[m]))}" for m in sorted(spurious)[:5])
    )


def test_culprit_track_is_clean(scored, gt):
    """The culprit must win on physics, not because priors handed it the answer."""
    s, _ = scored
    c = s[gt["culprit_mmsi"]]
    assert not c["trust_flags"] and not c["behaviour_flags"], _codes(c)
    assert c["classification"] == "MATCHED"


def test_identity_mismatch_raises_not_lowers_the_prior(scored):
    """Spoofing is evidence of intent. It must not protect a vessel."""
    s, cfg = scored
    priors = priors_from(s, cfg)
    for mmsi, v in s.items():
        if v["classification"] == "IDENTITY_MISMATCH":
            assert priors[mmsi]["trust_prior"] > 1.0


def test_culprit_still_ranked_first_with_priors_applied(gt):
    from samudra.attribution.__main__ import run

    out = run("demo-001", Path("artifacts"), quiet=True, write_outputs=False)
    top = out["suspects"][0]
    assert top["mmsi"] == gt["culprit_mmsi"], (
        f"priors demoted the culprit: "
        + "; ".join(
            f"#{x['rank']} {x['mmsi']} post={x['posterior']:.3f} "
            f"score={x['best_hypothesis']['score']:.3f} "
            f"t={x['trust_prior']:.2f} b={x['behaviour_prior']:.2f}"
            for x in out["suspects"]
        )
    )
