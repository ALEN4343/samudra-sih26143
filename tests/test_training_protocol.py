"""Training protocol — the split, the loss, and the leakage rule.

The previous revision selected its checkpoint on the test set. Nothing in the
test suite caught that, because everything it asserted was about shapes. These
assert the protocol itself, so the same mistake cannot come back quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from samudra.detection.datasets import SOSSegmentation, _val_bucket
from samudra.detection.train import SegLoss, metrics_from_confusion, soft_dice_loss

SOS = Path("data/raw/oilspill/sos")
needs_sos = pytest.mark.skipif(not SOS.is_dir(), reason="Deep-SAR SOS not present")


# ------------------------------------------------------------------- splits

@needs_sos
def test_train_val_test_are_disjoint():
    tr = {str(p) for p, _, _ in SOSSegmentation(split="train", augment=False).items}
    va = {str(p) for p, _, _ in SOSSegmentation(split="val", augment=False).items}
    te = {str(p) for p, _, _ in SOSSegmentation(split="test", augment=False).items}
    assert tr and va and te
    assert not (tr & va), "train and val overlap"
    assert not (tr & te), "train and test overlap"
    assert not (va & te), "val and test overlap"


@needs_sos
def test_val_comes_out_of_train_not_test():
    """Test is the dataset's published split and must stay whole."""
    te_dir = SOSSegmentation(split="test", augment=False)
    assert all("/test/" in str(p).replace("\\", "/") for p, _, _ in te_dir.items)
    va = SOSSegmentation(split="val", augment=False)
    assert all("/train/" in str(p).replace("\\", "/") for p, _, _ in va.items)


@needs_sos
def test_split_is_deterministic_across_instances():
    a = {str(p) for p, _, _ in SOSSegmentation(split="val", augment=False).items}
    b = {str(p) for p, _, _ in SOSSegmentation(split="val", augment=False).items}
    assert a == b


@needs_sos
def test_split_does_not_move_when_limit_changes():
    """A `limit` must subset the split, never redefine it."""
    full = {str(p) for p, _, _ in SOSSegmentation(split="val", augment=False).items}
    small = {str(p) for p, _, _ in
             SOSSegmentation(split="val", augment=False, limit=50).items}
    assert small <= full


@needs_sos
def test_both_sensors_appear_in_every_split():
    """A split that lost a sensor would make the per-sensor report meaningless."""
    for split in ("train", "val", "test"):
        sensors = {s for _, _, s in SOSSegmentation(split=split, augment=False).items}
        assert sensors == {"palsar", "sentinel"}, f"{split} has {sensors}"


@needs_sos
def test_val_fraction_is_close_to_requested():
    tr = len(SOSSegmentation(split="train", augment=False, val_frac=0.15).items)
    va = len(SOSSegmentation(split="val", augment=False, val_frac=0.15).items)
    assert 0.12 < va / (tr + va) < 0.18


def test_val_bucket_is_stable_and_spread():
    buckets = [_val_bucket("sentinel", f"{i}.png") for i in range(2000)]
    assert all(0 <= b < 1000 for b in buckets)
    assert len(set(buckets)) > 500, "hash is not spreading chips across buckets"
    assert _val_bucket("sentinel", "7.png") == _val_bucket("sentinel", "7.png")
    assert _val_bucket("sentinel", "7.png") != _val_bucket("palsar", "7.png")


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError):
        SOSSegmentation(split="holdout")


# -------------------------------------------------------------------- loss

def test_dice_loss_is_zero_for_a_perfect_prediction():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, 2:6, 2:6] = 1
    logits = torch.full((1, 2, 8, 8), -12.0)
    logits[0, 0][target[0] == 0] = 12.0
    logits[0, 1][target[0] == 1] = 12.0
    assert soft_dice_loss(logits, target).item() < 0.02


def test_dice_loss_punishes_predicting_everything():
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[0, 6:10, 6:10] = 1                      # 6.25% oil
    all_oil = torch.stack([torch.full((16, 16), -8.0),
                           torch.full((16, 16), 8.0)]).unsqueeze(0)
    perfect = torch.full((1, 2, 16, 16), -8.0)
    perfect[0, 0][target[0] == 0] = 8.0
    perfect[0, 1][target[0] == 1] = 8.0
    assert soft_dice_loss(all_oil, target) > soft_dice_loss(perfect, target) + 0.5


def test_seg_loss_reduces_to_cross_entropy_when_dice_is_zero():
    w = torch.tensor([1.0, 1.0])
    logits = torch.randn(2, 2, 8, 8)
    target = (torch.rand(2, 8, 8) > 0.5).long()
    ce_only = SegLoss(w, dice_weight=0.0)(logits, target)
    ce = torch.nn.CrossEntropyLoss(weight=w)(logits, target)
    assert torch.allclose(ce_only, ce)


def test_seg_loss_with_dice_differs_from_ce_alone():
    w = torch.tensor([1.0, 2.0])
    logits = torch.randn(2, 2, 8, 8)
    target = (torch.rand(2, 8, 8) > 0.7).long()
    assert not torch.allclose(SegLoss(w, 0.0)(logits, target),
                              SegLoss(w, 0.5)(logits, target))


# ----------------------------------------------------------------- metrics

def test_metrics_report_iou_dice_precision_recall():
    cm = np.array([[80, 20], [10, 90]])
    m = metrics_from_confusion(cm)
    for cls in ("background", "oil"):
        for k in ("iou", "dice", "precision", "recall", "f1"):
            assert k in m["per_class"][cls]
    assert "mean_dice" in m and "mean_iou" in m


def test_dice_is_never_below_iou():
    """Algebraic identity — a good guard against a transcription error."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        cm = rng.integers(1, 5000, size=(2, 2))
        m = metrics_from_confusion(cm)
        for cls in ("background", "oil"):
            assert m["per_class"][cls]["dice"] + 1e-9 >= m["per_class"][cls]["iou"]


def test_perfect_confusion_scores_one():
    m = metrics_from_confusion(np.array([[100, 0], [0, 50]]))
    assert m["mean_iou"] == pytest.approx(1.0)
    assert m["mean_dice"] == pytest.approx(1.0)


# -------------------------------------------------- recorded run provenance

def test_recorded_metrics_declare_their_protocol():
    p = Path("artifacts/model_metrics.json")
    if not p.exists():
        pytest.skip("no metrics recorded yet")
    m = json.loads(p.read_text())
    if "protocol" not in m:
        pytest.skip("metrics predate the protocol block")
    assert m["protocol"]["threshold_selected_on"] == "validation"
    assert "test" in m["protocol"] and "once" in m["protocol"]["test"]
    assert "known_limitation" in m["protocol"]


def test_recorded_metrics_flag_a_non_representative_run():
    p = Path("artifacts/model_metrics.json")
    if not p.exists():
        pytest.skip("no metrics recorded yet")
    m = json.loads(p.read_text())
    if not m.get("representative", False):
        assert "NOT REPRESENTATIVE" in m.get("note", "")


# -------------------------------------------------- small-slick boosting

def test_small_slick_weights_hit_their_band_targets():
    """The whole point is a controlled distribution, so check it is controlled.

    Weights are per-chip, so a band holding 10x more chips must give each of
    them 10x less weight to land on the same share of draws. Getting that
    backwards would silently boost the bands that are already dominant.
    """
    import numpy as np

    from samudra.detection.datasets import SMALL_SLICK_BANDS, small_slick_weights

    # Deliberately lopsided: one tiny chip, hundreds of huge ones.
    fr = np.array([0.0] * 20 + [0.005] * 1 + [0.03] * 40
                  + [0.10] * 300 + [0.25] * 400 + [0.60] * 500)
    w, report = small_slick_weights(fr)

    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()

    targets = {name: t for _, _, name, t in SMALL_SLICK_BANDS}
    for r in report:
        # Each band's total sampling probability equals its target share.
        assert r["sampled_share"] == pytest.approx(targets[r["band"]], abs=1e-3)

    # The single 'tiny' chip must be sampled far more often than a 'huge' one.
    assert w[20] > w[-1] * 100


def test_small_slick_weights_renormalise_when_a_band_is_absent():
    """A --train-limit subset can contain no chips of some size."""
    import numpy as np

    from samudra.detection.datasets import small_slick_weights

    fr = np.array([0.20] * 10 + [0.60] * 10)      # only large and huge
    w, report = small_slick_weights(fr)
    assert {r["band"] for r in report} == {"large", "huge"}
    assert w.sum() == pytest.approx(1.0)
    assert sum(r["sampled_share"] for r in report) == pytest.approx(1.0, abs=1e-6)


def test_small_slick_weights_fall_back_to_uniform_not_zeros():
    """Zero weights would make WeightedRandomSampler raise, losing the run."""
    import numpy as np

    from samudra.detection.datasets import small_slick_weights

    w, _ = small_slick_weights(np.array([]))
    assert len(w) == 0 or w.sum() == pytest.approx(1.0)


def test_boosting_lowers_the_oil_fraction_the_loss_is_weighted_against():
    """Sampling more small slicks makes oil rarer; the class weight must follow.

    If `oil_pixel_fraction` ignored the sampler, the loss would keep the
    unboosted weight while oil became scarcer — under-weighting the class
    exactly when boosting was meant to help it.
    """
    import numpy as np

    from samudra.detection.datasets import small_slick_weights

    fr = np.array([0.0] * 10 + [0.005] * 10 + [0.03] * 10
                  + [0.10] * 10 + [0.25] * 10 + [0.60] * 50)
    w, _ = small_slick_weights(fr)
    uniform = float(fr.mean())
    weighted = float((fr * w).sum() / w.sum())
    assert weighted < uniform, (weighted, uniform)
