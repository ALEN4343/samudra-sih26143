"""Dataset loaders and model plumbing — CLAUDE.md 5.2.

These guard the label conventions measured during inventory. If a future change
silently alters how masks are thresholded or how the two datasets are combined,
training would still run and the metrics would still look plausible — which is
exactly the kind of failure that is expensive to notice late.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from samudra.detection.datasets import (  # noqa: E402
    MASK_THRESHOLD,
    BinaryClassification,
    SOSSegmentation,
)
from samudra.detection.train import confusion, metrics_from_confusion  # noqa: E402

SOS_ROOT = Path("data/raw/oilspill/sos")
BIN_ROOT = Path("data/raw/oilspill/binary/data")

needs_sos = pytest.mark.skipif(
    not (SOS_ROOT / "train" / "palsar" / "image").is_dir(),
    reason="SOS dataset not present; run the inventory step",
)
needs_bin = pytest.mark.skipif(
    not (BIN_ROOT / "Class_0").is_dir(),
    reason="binary dataset not present; run the inventory step",
)


@needs_sos
def test_sos_pairs_images_with_labels():
    ds = SOSSegmentation(SOS_ROOT, "train", augment=False, limit=8)
    assert len(ds) == 8
    s = ds[0]
    assert s["image"].shape == (3, 256, 256)
    assert s["mask"].shape == (256, 256)
    assert s["image"].dtype == torch.float32
    assert s["mask"].dtype == torch.int64
    assert s["sensor"] in ("palsar", "sentinel")


@needs_sos
def test_masks_are_binary_after_thresholding():
    """Sentinel masks are anti-aliased; both sensors must end up 0/1."""
    ds = SOSSegmentation(SOS_ROOT, "train", augment=False, limit=24)
    seen = set()
    for i in range(len(ds)):
        seen |= set(np.unique(ds[i]["mask"].numpy()).tolist())
    assert seen <= {0, 1}, f"masks must be binary after thresholding, got {sorted(seen)}"


@needs_sos
def test_subset_keeps_both_sensors():
    """A subset drawn from one sensor only would hide cross-sensor failure."""
    ds = SOSSegmentation(SOS_ROOT, "train", augment=False, limit=20)
    assert len({ds.sensor_of(i) for i in range(len(ds))}) == 2


@needs_sos
def test_augmentation_keeps_image_and_mask_aligned():
    """A flip applied to one and not the other trains on wrong labels silently."""
    # Same seed so both pick the same underlying files; only augmentation differs.
    plain = SOSSegmentation(SOS_ROOT, "train", augment=False, limit=4, seed=0)
    aug = SOSSegmentation(SOS_ROOT, "train", augment=True, limit=4, seed=0)
    assert [p for p, _, _ in plain.items] == [p for p, _, _ in aug.items]

    for i in range(len(plain)):
        # Flips and 90-degree rotations permute pixels, so the oil pixel count is
        # invariant. If image and mask were transformed independently this still
        # holds, so also check the mask actually moved with a real transform.
        assert int(plain[i]["mask"].sum()) == int(aug[i]["mask"].sum())


@needs_sos
def test_augmentation_transforms_image_and_mask_identically():
    """The real risk: image flipped, mask not. Verified against a known transform."""


    from samudra.detection.datasets import _augment

    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = (img > 7).astype(np.uint8)

    class _R:
        """Forces one horizontal flip and no rotation."""

        def __init__(self):
            self.calls = 0

        def random(self):
            self.calls += 1
            return 0.0 if self.calls == 1 else 1.0

        def randint(self, a, b):
            return 0

    gi, gm = _augment(img, mask, _R())
    assert np.array_equal(gi, np.fliplr(img))
    assert np.array_equal(gm, np.fliplr(mask))


@needs_sos
def test_missing_dataset_fails_loudly():
    with pytest.raises(FileNotFoundError):
        SOSSegmentation("data/raw/oilspill/does-not-exist", "train")


@needs_bin
def test_binary_set_is_image_level_and_resized():
    tr = BinaryClassification(BIN_ROOT, split="train", augment=False, limit=6)
    s = tr[0]
    assert s["image"].shape == (3, 256, 256), "must match the segmentation input size"
    assert s["label"].item() in (0, 1)


@needs_bin
def test_binary_train_val_split_is_disjoint_and_deterministic():
    tr = BinaryClassification(BIN_ROOT, split="train", seed=7)
    va = BinaryClassification(BIN_ROOT, split="val", seed=7)
    assert not ({p for p, _ in tr.items} & {p for p, _ in va.items}), "split leaks"
    assert [p for p, _ in va.items] == [
        p for p, _ in BinaryClassification(BIN_ROOT, split="val", seed=7).items
    ]


def test_mask_threshold_is_the_documented_value():
    assert MASK_THRESHOLD == 128


def test_confusion_and_iou_are_correct():
    # 2 background right, 1 background called oil, 3 oil right, 1 oil missed.
    pred = np.array([0, 0, 1, 1, 1, 1, 0])
    true = np.array([0, 0, 0, 1, 1, 1, 1])
    cm = confusion(pred, true)
    assert cm.tolist() == [[2, 1], [1, 3]]
    m = metrics_from_confusion(cm)
    assert m["per_class"]["oil"]["iou"] == pytest.approx(3 / (3 + 1 + 1))
    assert m["per_class"]["oil"]["recall"] == pytest.approx(3 / 4)
    assert m["pixel_accuracy"] == pytest.approx(5 / 7)


def test_model_shapes():
    from samudra.detection.train import DeepLabV3Plus

    m = DeepLabV3Plus(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        assert m(x).shape == (2, 2, 256, 256), "segmentation output must match input size"
        assert m(x, aux_only=True).shape == (2, 2)


def test_smoke_metrics_are_labelled_not_representative():
    """A CPU smoke run must never be quotable by accident."""
    import json

    p = Path("artifacts/model_metrics.json")
    if not p.exists():
        pytest.skip("no metrics yet; run scripts/train_segmenter.py --smoke")
    m = json.loads(p.read_text())
    assert "representative" in m
    if not m["representative"]:
        assert "NOT REPRESENTATIVE" in m["note"]
    assert "oil-vs-lookalike" in m["classes_note"] or "lookalike" in m["classes_note"]
