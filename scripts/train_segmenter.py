"""Train the slick segmenter — CLAUDE.md 5.2, build order step 10.

    python scripts/train_segmenter.py --smoke       # 2 epochs, tiny subset
    python scripts/train_segmenter.py --epochs 25   # full run, wants a GPU

Primary supervision is Deep-SAR SOS (pixel masks). The Sentinel-1 binary set
feeds an auxiliary image-level head; see detection/datasets.py for why it is not
mixed into the segmentation loss.

PROTOCOL. This is the part a reviewer should check first.

  train  deterministic 85% of the published SOS train directory
  val    the other 15%. Drives early stopping, checkpoint selection and the
         decision threshold. Never trained on.
  test   the published SOS test directory. Loaded ONCE, at the very end, by the
         best checkpoint, at the threshold already fixed on val.

The previous revision evaluated `test` after every epoch and kept the checkpoint
with the best test oil-IoU. That is selection on the test set: the reported
number was the maximum of ~N noisy draws, not an estimate of unseen
performance. Fixed here, and called out because the old numbers are in git.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samudra.detection.datasets import (  # noqa: E402
    BinaryClassification, SOSSegmentation, small_slick_weights)
from samudra.detection.train import (  # noqa: E402
    DeepLabV3Plus,
    evaluate,
    evaluate_aux,
    pick_device,
    qualitative_panel,
    save_metrics,
    sweep_threshold,
    train_one_epoch,
)


def main() -> None:
    # A long run must not die on a console encoding. Windows picks cp1252 the
    # moment stdout is a pipe rather than a terminal, and this script prints
    # set-intersection and degree signs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Train DeepLabv3+ on the SAR oil-spill sets.")
    ap.add_argument("--sos", default="data/raw/oilspill/sos")
    ap.add_argument("--binary", default="data/raw/oilspill/binary/data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--aux-weight", type=float, default=0.4)
    ap.add_argument("--dice-weight", type=float, default=0.5,
                    help="0 reproduces the old weighted-CE-only objective")
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="oil class weight = ((1-p)/p)**power. 1.0 is the old "
                         "full inverse-frequency weight that drove precision to 0.35")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-aux", action="store_true", help="segmentation only")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--checkpoint", default="data/models/seg.pt")
    ap.add_argument("--metrics", default="artifacts/model_metrics.json")
    ap.add_argument("--plot", default="artifacts/model_metrics.png")
    ap.add_argument("--qualitative", default="artifacts/model_qualitative.png")
    ap.add_argument("--small-slick-boost", action="store_true",
                    help="re-weight sampling toward chips with little oil; "
                         "off by default so the previous run reproduces")
    ap.add_argument("--tag", default="", help="free-text label stored in the metrics")
    ap.add_argument(
        "--smoke", action="store_true",
        help="2 epochs on a small subset, to prove the pipeline runs end to end",
    )
    a = ap.parse_args()

    if a.smoke:
        a.epochs = 2
        a.train_limit = a.train_limit or 96
        a.val_limit = a.val_limit or 48
        a.test_limit = a.test_limit or 64
        a.batch_size = min(a.batch_size, 4)

    device = pick_device(a.cpu)
    representative = device.type == "cuda" and not a.smoke and a.train_limit is None

    print(f"mode              : {'SMOKE TEST' if a.smoke else 'full training'}")
    print(f"epochs            : {a.epochs}   patience {a.patience}")

    train_ds = SOSSegmentation(a.sos, "train", augment=True,
                               limit=a.train_limit, val_frac=a.val_frac)
    val_ds = SOSSegmentation(a.sos, "val", augment=False,
                             limit=a.val_limit, val_frac=a.val_frac)
    test_ds = SOSSegmentation(a.sos, "test", augment=False, limit=a.test_limit)
    print(f"SOS train/val/test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)} pairs")

    # A split that leaks is worse than no split, so prove it does not.
    overlap = {p.name for p, _, _ in train_ds.items} & {p.name for p, _, _ in val_ds.items}
    tr_paths = {str(p) for p, _, _ in train_ds.items}
    te_paths = {str(p) for p, _, _ in test_ds.items}
    assert not (tr_paths & te_paths), "train/test path overlap"
    print(f"leakage check     : train∩test {len(tr_paths & te_paths)} paths, "
          f"train∩val {len(set(tr_paths) & {str(p) for p, _, _ in val_ds.items})} paths"
          f"  (filename collisions across sensors: {len(overlap)}, expected)")

    aux_train = aux_val = None
    if not a.no_aux:
        lim = a.train_limit * 2 if a.train_limit else None
        aux_train = BinaryClassification(a.binary, split="train", augment=True, limit=lim)
        aux_val = BinaryClassification(
            a.binary, split="val", augment=False, limit=a.val_limit
        )
        print(f"binary aux train  : {len(aux_train)} images {aux_train.class_counts}")

    # Small-slick boosting. Measured on the SOS train split, mean oil fraction
    # is 24.7% and only 11.4% of chips hold less than 5% oil, so a uniformly
    # sampled model learns "large dark region = oil". The detection-limit
    # measurement sits at ~150 px, which is 0.2% of a chip — a size this set
    # barely contains. The sampler re-weights toward those chips; it does not
    # add data, and it cannot invent a size the set does not hold at all.
    sampler = None
    boost_report = None
    if a.small_slick_boost:
        fracs = train_ds.oil_fractions()
        w_chip, boost_report = small_slick_weights(fracs)
        sampler = WeightedRandomSampler(
            torch.as_tensor(w_chip, dtype=torch.double),
            num_samples=len(train_ds), replacement=True,
        )
        print("small-slick boost : ON")
        print(f"  {'band':<8}{'chips':>7}{'natural':>10}{'sampled':>10}{'boost':>8}")
        for r in boost_report:
            print(f"  {r['band']:<8}{r['chips']:>7}{r['natural_share']*100:>9.1f}%"
                  f"{r['sampled_share']*100:>9.1f}%{r['boost']:>7.1f}x")

    # Class weighting from the measured oil fraction, not a guessed constant.
    # weight_power damps it: the full inverse-frequency weight (power 1.0) is
    # what produced oil recall 0.98 at precision 0.35 in the previous run.
    # Under boosting the fraction must be computed against the sampler, because
    # drawing more small slicks lowers the oil fraction the network actually
    # sees — using the uniform figure would under-weight oil exactly when it
    # has become rarer.
    oil_frac = train_ds.oil_pixel_fraction(
        sample=min(300, len(train_ds)),
        weights=(w_chip if a.small_slick_boost else None),
    )
    oil_frac = float(min(max(oil_frac, 1e-3), 0.9))
    w_oil = ((1.0 - oil_frac) / oil_frac) ** a.weight_power
    weights = torch.tensor([1.0, w_oil], dtype=torch.float32)
    weights = weights / weights.mean()
    print(f"oil pixel fraction: {oil_frac * 100:.2f}%   class weights "
          f"[bg {weights[0]:.3f}, oil {weights[1]:.3f}]  (power {a.weight_power})")
    print(f"objective         : weighted CE + {a.dice_weight} x soft Dice")

    dl = lambda ds, sh, smp=None: DataLoader(  # noqa: E731
        ds, batch_size=a.batch_size, shuffle=(sh and smp is None), sampler=smp,
        num_workers=a.workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
        persistent_workers=bool(a.workers),
    )
    # Only the TRAINING loader is re-weighted. Validation and test stay at their
    # natural distribution — boosting them would change what the metric means
    # and make runs incomparable.
    seg_train = dl(train_ds, True, sampler)
    seg_val, seg_test = dl(val_ds, False), dl(test_ds, False)
    aux_train_dl = dl(aux_train, True) if aux_train else None
    aux_val_dl = dl(aux_val, False) if aux_val else None

    model = DeepLabV3Plus(num_classes=2, pretrained=not a.no_pretrained).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model             : DeepLabv3+ / ResNet-50, {n_par:.1f} M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(1, len(seg_train)) * max(a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps, pct_start=0.25, div_factor=10.0,
        final_div_factor=100.0,
    )
    use_amp = device.type == "cuda" and not a.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"schedule          : OneCycle, max_lr {a.lr}, {steps} steps"
          f"{'  | AMP fp16' if use_amp else ''}")

    history = []
    best_val = -1.0
    best_epoch = 0
    since_best = 0
    ckpt = Path(a.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        print(f"\n-- epoch {ep}/{a.epochs} " + "-" * 50)
        tr = train_one_epoch(
            model, seg_train, opt, device, weights,
            aux_loader=aux_train_dl, aux_weight=a.aux_weight,
            log_every=10 if a.smoke else 80, dice_weight=a.dice_weight,
            scaler=scaler, sched_per_batch=sched,
        )
        ev = evaluate(model, seg_val, device, per_sensor=False)   # VAL, never test
        oil = ev["per_class"]["oil"]
        print(f"   seg loss {tr['seg_loss']:.4f}   aux loss {tr['aux_loss']:.4f}   "
              f"{tr['seconds']:.0f}s")
        print(f"   VAL mean IoU {ev['mean_iou']:.4f}   oil IoU {oil['iou']:.4f}   "
              f"dice {oil['dice']:.4f}   P {oil['precision']:.4f}   R {oil['recall']:.4f}")
        history.append({
            "epoch": ep, **tr,
            "val_mean_iou": ev["mean_iou"], "val_oil_iou": oil["iou"],
            "val_oil_dice": oil["dice"], "val_oil_precision": oil["precision"],
            "val_oil_recall": oil["recall"],
            "lr": opt.param_groups[0]["lr"],
        })

        if oil["iou"] > best_val:
            best_val, best_epoch, since_best = oil["iou"], ep, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "deeplabv3plus_resnet50",
                    "num_classes": 2,
                    "classes": ["background", "oil"],
                    "input_size": 256,
                    "normalisation": "imagenet",
                    "mask_threshold": 128,
                    "epoch": ep,
                    "val_oil_iou": oil["iou"],
                    "selected_on": "validation split (never test)",
                    "representative": representative,
                },
                ckpt,
            )
            print(f"   checkpoint saved ({ckpt}, val oil IoU {oil['iou']:.4f})")
        else:
            since_best += 1
            print(f"   no val improvement ({since_best}/{a.patience})")
            if since_best >= a.patience:
                print(f"   early stop at epoch {ep}; best was epoch {best_epoch}")
                break

    # ---- restore the selected checkpoint, then touch test exactly once -----
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"\nrestored epoch {state['epoch']} (val oil IoU {state['val_oil_iou']:.4f})")

    sweep = sweep_threshold(model, seg_val, device)
    thr = sweep["best_by_oil_iou"]
    print(f"threshold chosen on VAL: {thr:.2f}")

    val_metrics = evaluate(model, seg_val, device, threshold=thr)
    test_metrics = evaluate(model, seg_test, device, threshold=thr)
    aux_metrics = evaluate_aux(model, aux_val_dl, device) if aux_val_dl else None

    qual = None
    try:
        qual = str(qualitative_panel(model, test_ds, device, Path(a.qualitative),
                                     n=6, threshold=thr))
    except Exception as exc:  # noqa: BLE001 - a missing figure must not lose a run
        print(f"qualitative panel skipped: {exc}")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tag": a.tag,
        "representative": representative,
        "note": (
            "Full GPU training run. Checkpoint and decision threshold selected on "
            "the validation split; the test split was evaluated once, afterwards."
            if representative
            else "NOT REPRESENTATIVE: reduced subset and/or CPU smoke test. These "
                 "numbers prove the training pipeline runs end to end and nothing more."
        ),
        "protocol": {
            "train": "85% of the published SOS train directory (hash partition)",
            "val": "15% of the published SOS train directory; drives early "
                   "stopping, checkpoint selection and the decision threshold",
            "test": "the published SOS test directory; loaded once, at the end",
            "selection_metric": "validation oil IoU",
            "threshold_selected_on": "validation",
            "known_limitation": "SOS chips carry no parent-scene id, so chips cut "
                                "from one scene can fall on both sides of the "
                                "train/val partition. Validation is therefore "
                                "optimistic; the published test split is the "
                                "number to quote.",
        },
        "device": str(device),
        "platform": platform.platform(),
        "epochs_run": len(history),
        "epochs_requested": a.epochs,
        "best_epoch": best_epoch,
        "batch_size": a.batch_size,
        "lr": a.lr,
        "scheduler": "OneCycleLR",
        "amp": bool(use_amp),
        "dice_weight": a.dice_weight,
        "weight_power": a.weight_power,
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
        "test_pairs": len(test_ds),
        "oil_pixel_fraction": oil_frac,
        "small_slick_boost": {
            "enabled": bool(a.small_slick_boost),
            "why": ("SOS train has mean oil fraction 24.7% and only 11.4% of "
                    "chips below 5% oil, so uniform sampling teaches 'large "
                    "dark region = oil'. The measured detection floor is ~150 "
                    "px, i.e. 0.2% of a chip."),
            "bands": boost_report,
            "note": ("Training loader only. Val and test keep their natural "
                     "distribution, so metrics stay comparable across runs. "
                     "oil_pixel_fraction above is computed against the sampler "
                     "when boosting is on."),
        },
        "class_weights": weights.tolist(),
        "aux_head": {
            "used": aux_train is not None,
            "why": (
                "Per-image labels only, and its intensity statistics differ sharply "
                "from SOS (std 8-12 vs 31-49). Mixing it into the segmentation loss "
                "would let the network discriminate dataset rather than oil, so it "
                "trains a separate image-level head sharing the backbone."
            ),
            "train_images": len(aux_train) if aux_train else 0,
        },
        "classes_note": (
            "oil vs background only. Neither SOS sensor encodes an oil-vs-lookalike "
            "distinction — checked during inventory, masks are binary."
        ),
        "decision_threshold": thr,
        "threshold_sweep_val": sweep["per_threshold"],
        "validation": val_metrics,
        "segmentation": test_metrics,     # key kept: downstream readers expect it
        "classification": aux_metrics,
        "history": history,
        "checkpoint": str(ckpt),
        "qualitative_panel": qual,
    }
    save_metrics(metrics, Path(a.metrics), Path(a.plot))

    print("\n" + "=" * 78)
    print(f"TEST SPLIT (held out, threshold {thr:.2f}, evaluated once)")
    cm = np.array(test_metrics["confusion_matrix"])
    print(f"  confusion matrix   [[{cm[0,0]:>12,} {cm[0,1]:>12,}]   true background")
    print(f"                      [{cm[1,0]:>12,} {cm[1,1]:>12,}]]  true oil")
    for name, m in test_metrics["per_class"].items():
        print(f"  {name:<12} IoU {m['iou']:.4f}  Dice {m['dice']:.4f}  "
              f"P {m['precision']:.4f}  R {m['recall']:.4f}")
    print(f"  mean IoU {test_metrics['mean_iou']:.4f}   "
          f"mean Dice {test_metrics['mean_dice']:.4f}   "
          f"pixel accuracy {test_metrics['pixel_accuracy']:.4f}")
    for s, m in sorted(test_metrics.get("per_sensor", {}).items()):
        o = m["per_class"]["oil"]
        print(f"  {s:<10} oil IoU {o['iou']:.4f}  Dice {o['dice']:.4f}  "
              f"P {o['precision']:.4f}  R {o['recall']:.4f}")
    if aux_metrics:
        print("\nAUX IMAGE-LEVEL HEAD (binary set, val split)")
        for name, m in aux_metrics["per_class"].items():
            print(f"  {name:<12} precision {m['precision']:.4f}  recall {m['recall']:.4f}")
    if not representative:
        print("\n!! NOT REPRESENTATIVE — reduced run. Do not quote these numbers.")
    print(f"\nmetrics  : {a.metrics}")
    print(f"plot     : {a.plot}")
    if qual:
        print(f"examples : {qual}")


if __name__ == "__main__":
    main()
