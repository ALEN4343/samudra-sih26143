"""Evidence dossier and chain of custody — CLAUDE.md section 4.6."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from samudra.evidence import integrity

ART = Path("artifacts/demo-001")

pytestmark = pytest.mark.skipif(
    not (ART / "incident.json").exists(),
    reason="run: python -m samudra.attribution --incident demo-001",
)


# --------------------------------------------------------------------------
# Hash chain
# --------------------------------------------------------------------------


@pytest.fixture
def chain(tmp_path):
    """An isolated artifacts tree with two incidents recorded."""
    root = tmp_path / "artifacts"
    d = root / "demo-001"
    d.mkdir(parents=True)
    for n in ("incident.json", "observed_slick.geojson"):
        shutil.copy(ART / n, d / n)
    log = root / "audit.log"
    e1 = integrity.append("demo-001", d, log)
    (d / "extra.json").write_text(json.dumps({"b": 2, "a": 1}))
    e2 = integrity.append("demo-001", d, log)
    return root, log, e1, e2


def test_chain_links_and_verifies(chain):
    root, log, e1, e2 = chain
    assert e1["prev_hash"] == integrity.GENESIS
    assert e2["prev_hash"] == e1["chain_hash"]
    r = integrity.verify(log, root)
    assert r["ok"], r["errors"]
    assert r["entries"] == 2
    assert r["head"] == e2["chain_hash"]


def test_tampering_with_an_artifact_is_detected(chain):
    """The point of the whole layer."""
    root, log, _, _ = chain
    target = root / "demo-001" / "observed_slick.geojson"
    gj = json.loads(target.read_text())
    gj["features"][0]["properties"]["area_km2"] = 1.0  # quietly change the finding
    target.write_text(json.dumps(gj))

    r = integrity.verify(log, root)
    assert not r["ok"]
    assert any("ON-DISK CONTENT DIFFERS" in e for e in r["errors"]), r["errors"]


def test_tampering_with_the_log_breaks_every_later_link(chain):
    root, log, _, _ = chain
    lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    lines[0]["incident_id"] = "demo-999"  # rewrite history
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    r = integrity.verify(log, root)
    assert not r["ok"]
    assert any("chain_hash does not match its body" in e for e in r["errors"]), r["errors"]


def test_removing_an_artifact_is_detected(chain):
    root, log, _, _ = chain
    (root / "demo-001" / "extra.json").unlink()
    r = integrity.verify(log, root)
    assert not r["ok"]
    assert any("missing from disk" in e for e in r["errors"]), r["errors"]


def test_canonical_json_is_order_independent():
    """Key order and whitespace must not change the digest."""
    a = integrity.canonical_json({"b": 1, "a": [1, 2], "c": {"y": 1, "x": 2}})
    b = integrity.canonical_json({"c": {"x": 2, "y": 1}, "a": [1, 2], "b": 1})
    assert a == b


def test_ground_truth_is_never_hashed(chain):
    """Ground truth is not a pipeline artifact and must stay out of the chain."""
    root, log, _, e2 = chain
    (root / "demo-001" / "ground_truth.json").write_text('{"culprit_mmsi": 1}')
    e3 = integrity.append("demo-001", root / "demo-001", log)
    assert "ground_truth.json" not in e3["files"]
    assert "audit.log" not in e3["files"]


# --------------------------------------------------------------------------
# Dossier
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pdf():
    from samudra.evidence.report import build

    return build("demo-001", Path("artifacts"))


def test_dossier_has_real_embedded_images(pdf):
    """A blank box where the map should be is the classic silent failure."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open(pdf)
    imgs = [
        doc.extract_image(x[0])
        for page in doc
        for x in page.get_images(full=True)
    ]
    assert len(imgs) >= 2, f"expected the SAR chip and the hypothesis map, got {len(imgs)}"
    for im in imgs:
        assert im["width"] > 300 and im["height"] > 300
        assert len(im["image"]) > 20_000, "image is too small to contain a real figure"


def test_dossier_states_the_finding_and_the_chain_hash(pdf):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open(pdf)
    text = "".join(p.get_text() for p in doc)

    inc = json.loads((ART / "incident.json").read_text())
    top = inc["suspects"][0]
    assert str(top["mmsi"]) in text
    assert "MARPOL" in text
    assert "Chain hash" in text
    assert "Regulation 15" in text and "Regulation 34" in text

    head = integrity.verify()["head"]
    assert head in text.replace("\n", ""), "the dossier must print the current chain head"


def test_synthetic_dossier_is_labelled_synthetic(pdf):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open(pdf)
    text = "".join(p.get_text() for p in doc)
    inc = json.loads((ART / "incident.json").read_text())
    if inc.get("synthetic"):
        assert "SYNTHETIC" in text
        # On every page, not just the first.
        for i, page in enumerate(doc):
            assert "SYNTHETIC" in page.get_text(), f"page {i + 1} is missing the banner"
