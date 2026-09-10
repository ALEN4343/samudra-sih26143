"""Full attribution chain against planted ground truth.

The most important test in the build. It runs layers 6 and 7 end to end on
demo-001 and asserts the planted culprit comes out on top.

If this fails, do NOT adjust the scoring weights to force the right answer —
that hides the bug and will fail on any other scenario. Print the per-term
breakdown and find which term is wrong.

This module is one of the only places allowed to read ground_truth.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from samudra.attribution.__main__ import run

ART = Path("artifacts/demo-001")

pytestmark = pytest.mark.skipif(
    not (ART / "ground_truth.json").exists(),
    reason="run: python -m samudra.synth.generate --scenario demo-001",
)


@pytest.fixture(scope="module")
def gt() -> dict:
    return json.loads((ART / "ground_truth.json").read_text())


@pytest.fixture(scope="module")
def result() -> dict:
    return run("demo-001", Path("artifacts"), quiet=True, write_outputs=False)


def test_culprit_is_ranked_first(result, gt):
    top = result["suspects"][0]
    if top["mmsi"] != gt["culprit_mmsi"]:
        lines = [f"\nRanked {top['mmsi']} first, truth is {gt['culprit_mmsi']}.", ""]
        for s in result["suspects"]:
            b = s["best_hypothesis"]
            mark = " <-- TRUTH" if s["mmsi"] == gt["culprit_mmsi"] else ""
            lines.append(
                f"  #{s['rank']} {s['mmsi']} post={s['posterior']:.4f} "
                f"score={b['score']:.4f} iou={b['iou']:.4f} "
                f"dcen={b['centroid_offset_km']:.2f}km dori={b['orientation_delta_deg']:.1f}deg "
                f"area={b['area_ratio']:.3f} trust={s['trust_prior']:.2f}{mark}"
            )
        pytest.fail("\n".join(lines))


def test_funnel_actually_narrows(result):
    f = result["funnel"]
    assert f["total_in_scene"] > f["in_envelope"], "pruning removed nobody"
    assert f["in_envelope"] >= 2, (
        f"only {f['in_envelope']} candidate(s) in the envelope — ranking never had "
        f"to discriminate, so a correct answer proves nothing"
    )
    assert f["ranked"] == f["scored"]


def test_top_suspect_is_clearly_separated(result):
    """The winner must beat the runner-up, not tie it."""
    s = result["suspects"]
    assert len(s) >= 2
    assert s[0]["best_hypothesis"]["score"] > s[1]["best_hypothesis"]["score"] + 0.05


def test_release_time_and_position_are_close_to_truth(result, gt):
    from samudra.geo import Projector

    b = result["suspects"][0]["best_hypothesis"]
    truth_t = datetime.fromisoformat(gt["true_release_at"])
    est_t = datetime.fromisoformat(b["release_at"])
    dt_h = abs((est_t - truth_t).total_seconds()) / 3600.0

    proj = Projector(gt["true_release_lat"], gt["true_release_lon"])
    ax, ay = proj.to_m(b["release_lon"], b["release_lat"])
    bx, by = proj.to_m(gt["true_release_lon"], gt["true_release_lat"])
    dist_km = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / 1000.0

    print(f"\nrelease time error : {dt_h:.2f} h\nrelease pos error  : {dist_km:.2f} km")
    assert dt_h <= 1.0, f"release time off by {dt_h:.2f} h"
    assert dist_km <= 20.0, f"release position off by {dist_km:.1f} km"


def test_slick_age_matches_truth(result, gt):
    a = result["slick_age"]
    err = abs(a["best_hours"] - gt["slick_age_hours"])
    print(
        f"\nage estimate {a['best_hours']:.2f} h vs truth {gt['slick_age_hours']:.2f} h "
        f"(Fay {a['fay_estimate_hours']:.2f} h)"
    )
    assert err <= 1.5, f"slick age off by {err:.2f} h"
    assert a["low_hours"] <= a["best_hours"] <= a["high_hours"]


def test_posteriors_form_a_distribution(result):
    total = sum(s["posterior"] for s in result["suspects"])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_every_suspect_has_readable_rationale(result):
    for s in result["suspects"]:
        assert s["rationale"], f"suspect {s['mmsi']} has no rationale"
        assert all(isinstance(x, str) and x.strip() for x in s["rationale"])


def test_pipeline_never_reads_ground_truth():
    """Structural guarantee, not a convention.

    Checks for an actual *read*, not a mere mention: integrity.py legitimately
    names the file in order to exclude it from the audit manifest, and a naive
    substring check would flag that as a violation and train us to ignore it.

    synth/generate.py is exempt: it is the simulator and authors the file.
    """
    READ_VERBS = ("read_text", "read_bytes", "json.load", "open(", "loads(")
    author = Path("src/samudra/synth/generate.py").resolve()

    offenders = []
    for f in Path("src/samudra").rglob("*.py"):
        if f.resolve() == author:
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if "ground_truth" not in line:
                continue
            code = line.split("#", 1)[0]
            if "ground_truth" in code and any(v in code for v in READ_VERBS):
                offenders.append(f"{f}:{n}: {line.strip()}")
    joined = chr(10).join(offenders)
    assert not offenders, "pipeline code reading ground truth: " + joined

    # And the simulator must only write it, never read it back.
    text = author.read_text()
    assert 'ground_truth.json").write_text' in text, "simulator no longer writes ground truth"
    assert 'ground_truth.json").read_text' not in text, "simulator reads back its own answer"
