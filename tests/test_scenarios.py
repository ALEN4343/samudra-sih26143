"""Generalisation across every generated scenario.

test_attribution.py proves the engine works on demo-001, the scenario it was
developed against. That is necessary but weak evidence: code can be tuned, even
unintentionally, until one case passes.

This module runs the same assertions over *every* scenario present in
artifacts/. demo-002 uses a different seed, a different culprit, roughly twice
the traffic and a higher wind band, and the engine never saw it during
development. Passing here is the only real evidence the physics generalises
rather than the constants having been fitted.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from samudra.attribution.__main__ import run
from samudra.geo import Projector

ROOT = Path("artifacts")


def _scenarios() -> list[str]:
    if not ROOT.is_dir():
        return []
    return sorted(
        d.name for d in ROOT.iterdir()
        if d.is_dir() and (d / "ground_truth.json").exists()
    )


SCENARIOS = _scenarios()


def _is_real(scenario: str) -> bool:
    """True when the traffic in this scenario is a real AIS feed."""
    gt = json.loads((ROOT / scenario / "ground_truth.json").read_text())
    return gt.get("traffic") == "real"


SYNTHETIC = [s for s in SCENARIOS if not _is_real(s)]
REAL = [s for s in SCENARIOS if _is_real(s)]

pytestmark = pytest.mark.skipif(not SCENARIOS, reason="no scenarios generated")


@pytest.fixture(scope="module")
def results() -> dict[str, tuple[dict, dict]]:
    """Run each scenario once and share across tests."""
    out = {}
    for s in SCENARIOS:
        gt = json.loads((ROOT / s / "ground_truth.json").read_text())
        out[s] = (run(s, ROOT, quiet=True), gt)
    return out


@pytest.mark.parametrize("scenario", SYNTHETIC)
def test_culprit_ranked_first(results, scenario):
    inc, gt = results[scenario]
    top = inc["suspects"][0]
    if top["mmsi"] != gt["culprit_mmsi"]:
        rows = "; ".join(
            f"#{s['rank']} {s['mmsi']} post={s['posterior']:.3f} "
            f"score={s['best_hypothesis']['score']:.3f} "
            f"iou={s['best_hypothesis']['iou']:.3f} "
            f"dori={s['best_hypothesis']['orientation_delta_deg']:.0f}"
            + (" <-- TRUTH" if s["mmsi"] == gt["culprit_mmsi"] else "")
            for s in inc["suspects"]
        )
        pytest.fail(f"{scenario}: ranked {top['mmsi']}, truth {gt['culprit_mmsi']}. {rows}")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_release_recovered_accurately(results, scenario):
    """Whoever is ranked first must at least describe the right event."""
    inc, gt = results[scenario]
    b = inc["suspects"][0]["best_hypothesis"]

    dt_h = abs(
        (datetime.fromisoformat(b["release_at"])
         - datetime.fromisoformat(gt["true_release_at"])).total_seconds()
    ) / 3600.0

    proj = Projector(gt["true_release_lat"], gt["true_release_lon"])
    ax, ay = proj.to_m(b["release_lon"], b["release_lat"])
    bx, by = proj.to_m(gt["true_release_lon"], gt["true_release_lat"])
    km = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / 1000.0

    age_err = abs(inc["slick_age"]["best_hours"] - gt["slick_age_hours"])
    print(f"\n{scenario}: time {dt_h:.2f} h, position {km:.2f} km, age {age_err:.2f} h")

    assert dt_h <= 1.5, f"{scenario}: release time off by {dt_h:.2f} h"
    assert km <= 25.0, f"{scenario}: release position off by {km:.1f} km"
    assert age_err <= 2.0, f"{scenario}: slick age off by {age_err:.2f} h"


@pytest.mark.parametrize("scenario", SYNTHETIC)
def test_funnel_discriminates(results, scenario):
    inc, _ = results[scenario]
    f = inc["funnel"]
    assert f["total_in_scene"] > f["in_envelope"], f"{scenario}: pruning removed nobody"
    assert f["in_envelope"] >= 2, (
        f"{scenario}: only {f['in_envelope']} candidate(s) survived pruning, so ranking "
        f"never had to discriminate"
    )
    top, second = inc["suspects"][0], inc["suspects"][1]
    margin = top["best_hypothesis"]["score"] - second["best_hypothesis"]["score"]
    assert margin > 0.05, f"{scenario}: winner beats runner-up by only {margin:.3f}"


@pytest.mark.parametrize("scenario", SYNTHETIC)
def test_planted_anomalies_are_found(results, scenario):
    """Trust layer recall, per scenario."""
    import yaml

    from samudra.attribution.prune import load_tracks
    from samudra.trust.score import score_tracks

    inc, gt = results[scenario]
    cfg = yaml.safe_load(open("config/weights.yaml"))
    tracks = load_tracks(ROOT / scenario / "ais.parquet")
    scores = score_tracks(tracks, cfg, Projector.for_bounds(gt["aoi_bounds"]))

    def codes(m):
        s = scores[m]
        return {f["code"] for f in s["trust_flags"]} | {f["code"] for f in s["behaviour_flags"]}

    expect = {
        "spoofed": "POSITION_TELEPORT",
        "loiter": "LOITERING",
        "ais_gap": "AIS_GAP",
        "slowdown": "SPEED_REDUCTION",
    }
    for kind, code in expect.items():
        for mmsi in gt["planted_anomalies"].get(kind, []):
            assert code in codes(mmsi), f"{scenario}: {mmsi} planted {kind}, got {codes(mmsi)}"

    planted = set(sum(gt["planted_anomalies"].values(), []))
    flagged = {m for m, v in scores.items() if v["trust_flags"] or v["behaviour_flags"]}
    assert not (flagged - planted), (
        f"{scenario}: false positives on {sorted(flagged - planted)}"
    )


@pytest.mark.parametrize("scenario", SYNTHETIC)
def test_culprit_wins_without_prior_help(results, scenario):
    """The culprit's own track must be clean, so physics does the work."""
    inc, gt = results[scenario]
    top = inc["suspects"][0]
    assert top["mmsi"] == gt["culprit_mmsi"]
    assert top["trust_prior"] == pytest.approx(1.0)
    assert top["behaviour_prior"] == pytest.approx(1.0)


def test_more_than_one_scenario_exists():
    """A single passing scenario is not evidence of generalisation."""
    assert len(SCENARIOS) >= 2, (
        f"only {SCENARIOS} generated. Run: "
        f"bash scripts/run_demo.sh demo-002 --regen --no-serve --seed 77341 "
        f"--vessels 80 --decoys 9 --wind 7.0 9.0"
    )


# --------------------------------------------------------------------------
# Real traffic
#
# A weaker claim, and deliberately so. In the Houston Ship Channel every
# oil-capable transit runs the same fairway - measured maximum separation between
# any such vessel and the nearest other moving vessel is 72 m across all 97
# candidates. Several vessels therefore pass the same point within the hour, and
# their simulated slicks are geometrically near-identical (every ranked
# candidate's orientation delta is 1-13 degrees, against 45-86 for the synthetic
# decoys).
#
# Geometry cannot single one out there, and asserting that it does would be
# asserting a falsehood. What the engine CAN do on real data is narrow hundreds
# of vessels to a short list containing the culprit, and recover WHEN the release
# happened. Those are what is tested.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", REAL)
def test_real_culprit_survives_pruning_and_is_scored(results, scenario):
    inc, gt = results[scenario]
    ranked = [s["mmsi"] for s in inc["suspects"]]
    assert gt["culprit_mmsi"] in ranked, (
        f"{scenario}: the culprit was dropped before scoring. It must always be "
        f"scored even when geometry cannot single it out."
    )


@pytest.mark.parametrize("scenario", REAL)
def test_real_culprit_is_in_the_shortlist(results, scenario):
    """Containment, not first place."""
    inc, gt = results[scenario]
    ranked = [s["mmsi"] for s in inc["suspects"]]
    pos = ranked.index(gt["culprit_mmsi"]) + 1
    in_scene = inc["funnel"]["total_in_scene"]
    print()
    print(f"{scenario}: culprit ranked {pos} of {len(ranked)} scored, from {in_scene} in scene (top {100 * pos / in_scene:.1f}% of the fleet)")
    assert pos <= max(20, len(ranked) // 2), (
        f"{scenario}: culprit ranked {pos} of {len(ranked)} — outside the shortlist"
    )
    assert pos / in_scene < 0.10, (
        f"{scenario}: culprit at {pos}/{in_scene} is not a useful narrowing"
    )


@pytest.mark.parametrize("scenario", REAL)
def test_real_release_time_is_recovered(results, scenario):
    """The culprit's own best hypothesis must still find the right moment."""
    inc, gt = results[scenario]
    me = next(s for s in inc["suspects"] if s["mmsi"] == gt["culprit_mmsi"])
    b = me["best_hypothesis"]
    dt_h = abs(
        (datetime.fromisoformat(b["release_at"])
         - datetime.fromisoformat(gt["true_release_at"])).total_seconds()
    ) / 3600.0

    proj = Projector(gt["true_release_lat"], gt["true_release_lon"])
    ax, ay = proj.to_m(b["release_lon"], b["release_lat"])
    bx, by = proj.to_m(gt["true_release_lon"], gt["true_release_lat"])
    km = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / 1000.0
    print()
    print(f"{scenario}: culprit own hypothesis - time {dt_h:.2f} h, position {km:.2f} km")
    assert dt_h <= 1.5, f"{scenario}: release time off by {dt_h:.2f} h"
    assert km <= 6.0, f"{scenario}: release position off by {km:.1f} km"


@pytest.mark.parametrize("scenario", REAL)
def test_real_funnel_narrows_hard(results, scenario):
    inc, _ = results[scenario]
    f = inc["funnel"]
    assert f["in_envelope"] < f["total_in_scene"] * 0.25, (
        f"{scenario}: pruning kept {f['in_envelope']} of {f['total_in_scene']}"
    )
    assert f.get("in_envelope_underway", 0) < f["in_envelope"], (
        "in a port most envelope survivors are moored; if all are under way the "
        "underway test is not doing anything"
    )
