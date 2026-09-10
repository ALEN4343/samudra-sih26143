"""Train the slick segmenter — CLAUDE.md 5.2, build order step 10.

    python scripts/train_segmenter.py --smoke      # 2 epochs, small subset, CPU
    python scripts/train_segmenter.py --epochs 30  # full run, needs a GPU

Primary supervision is Deep-SAR SOS (pixel masks). The Sentinel-1 binary set
feeds an auxiliary image-level head; see detection/datasets.py for why it is not
mixed into the segmentation loss.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samudra.detection.datasets import BinaryClassification, SOSSegmentation  # noqa: E402
from samudra.detection.train import (  # noqa: E402
    DeepLabV3Plus,
    evaluate,
    evaluate_aux,
    pick_device,
    save_metrics,
    train_one_epoch,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DeepLabv3+ on the SAR oil-spill sets.")
    ap.add_argument("--sos", default="data/raw/oilspill/sos")
    ap.add_argument("--binary", default="data/raw/oilspill/binary/data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--aux-weight", type=float, default=0.4)
    ap.add_argument("--no-aux", action="store_true", help="segmentation only")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--checkpoint", default="data/models/seg.pt")
    ap.add_argument("--metrics", default="artifacts/model_metrics.json")
    ap.add_argument("--plot", default="artifacts/model_metrics.png")
    ap.add_argument(
        "--smoke", action="store_true",
        help="2 epochs on a small subset, to prove the pipeline runs end to end",
    )
    a = ap.parse_args()

    if a.smoke:
        a.epochs = 2
        a.train_limit = a.train_limit or 96
        a.test_limit = a.test_limit or 64
        a.batch_size = min(a.batch_size, 4)

    device = pick_device(a.cpu)
    representative = device.type == "cuda" and not a.smoke

    print(f"mode              : {'SMOKE TEST' if a.smoke else 'full training'}")
    print(f"epochs            : {a.epochs}")

    train_ds = SOSSegmentation(a.sos, "train", augment=True, limit=a.train_limit)
    test_ds = SOSSegmentation(a.sos, "test", augment=False, limit=a.test_limit)
    print(f"SOS train / test  : {len(train_ds)} / {len(test_ds)} image-mask pairs")

    aux_train = aux_test = None
    if not a.no_aux:
        lim = a.train_limit * 2 if a.train_limit else None
        aux_train = BinaryClassification(a.binary, split="train", augment=True, limit=lim)
        aux_test = BinaryClassification(
            a.binary, split="val", augment=False, limit=a.test_limit
        )
        print(f"binary aux train  : {len(aux_train)} images {aux_train.class_counts}")

    # Class weighting from the measured oil fraction, not a guessed constant.
    oil_frac = train_ds.oil_pixel_fraction(sample=min(300, len(train_ds)))
    oil_frac = float(min(max(oil_frac, 1e-3), 0.9))
    weights = torch.tensor([1.0, (1.0 - oil_frac) / oil_frac], dtype=torch.float32)
    weights = weights / weights.mean()
    print(f"oil pixel fraction: {oil_frac * 100:.2f}%   class weights "
          f"[bg {weights[0]:.3f}, oil {weights[1]:.3f}]")

    dl = lambda ds, sh: DataLoader(  # noqa: E731
        ds, batch_size=a.batch_size, shuffle=sh, num_workers=a.workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    seg_train, seg_test = dl(train_ds, True), dl(test_ds, False)
    aux_train_dl = dl(aux_train, True) if aux_train else None
    aux_test_dl = dl(aux_test, False) if aux_test else None

    model = DeepLabV3Plus(num_classes=2, pretrained=not a.no_pretrained).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model             : DeepLabv3+ / ResNet-50, {n_par:.1f} M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))

    history = []
    best_iou = -1.0
    ckpt = Path(a.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        print(f"\n-- epoch {ep}/{a.epochs} " + "-" * 50)
        tr = train_one_epoch(
            model, seg_train, opt, device, weights,
            aux_loader=aux_train_dl, aux_weight=a.aux_weight,
            log_every=10 if a.smoke else 50,
        )
        sched.step()
        ev = evaluate(model, seg_test, device)
        oil_iou = ev["per_class"]["oil"]["iou"]
        print(f"   seg loss {tr['seg_loss']:.4f}   aux loss {tr['aux_loss']:.4f}   "
              f"{tr['seconds']:.0f}s")
        print(f"   mean IoU {ev['mean_iou']:.4f}   oil IoU {oil_iou:.4f}   "
              f"oil recall {ev['per_class']['oil']['recall']:.4f}")
        history.append({"epoch": ep, **tr, "mean_iou": ev["mean_iou"], "oil_iou": oil_iou})

        if oil_iou > best_iou:
            best_iou = oil_iou
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
                    "oil_iou": oil_iou,
                    "representative": representative,
                },
                ckpt,
            )
            print(f"   checkpoint saved ({ckpt}, oil IoU {oil_iou:.4f})")

    seg_metrics = evaluate(model, seg_test, device)
    aux_metrics = evaluate_aux(model, aux_test_dl, device) if aux_test_dl else None

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "representative": representative,
        "note": (
            "Full GPU training run."
            if representative
            else "NOT REPRESENTATIVE: reduced subset and/or CPU smoke test. These "
                 "numbers prove the training pipeline runs end to end and nothing more."
        ),
        "device": str(device),
        "platform": platform.platform(),
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "lr": a.lr,
        "train_pairs": len(train_ds),
        "test_pairs": len(test_ds),
        "oil_pixel_fraction": oil_frac,
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
        "segmentation": seg_metrics,
        "classification": aux_metrics,
        "history": history,
        "checkpoint": str(ckpt),
    }
    save_metrics(metrics, Path(a.metrics), Path(a.plot))

    print("\n" + "=" * 78)
    print("SEGMENTATION (oil vs background)")
    cm = np.array(seg_metrics["confusion_matrix"])
    print(f"  confusion matrix   [[{cm[0,0]:>12,} {cm[0,1]:>12,}]   true background")
    print(f"                      [{cm[1,0]:>12,} {cm[1,1]:>12,}]]  true oil")
    for name, m in seg_metrics["per_class"].items():
        print(f"  {name:<12} IoU {m['iou']:.4f}   precision {m['precision']:.4f}   "
              f"recall {m['recall']:.4f}")
    print(f"  mean IoU {seg_metrics['mean_iou']:.4f}   "
          f"pixel accuracy {seg_metrics['pixel_accuracy']:.4f}")
    for s, m in sorted(seg_metrics.get("per_sensor", {}).items()):
        print(f"  [{s:<8}] oil IoU {m['per_class']['oil']['iou']:.4f}   "
              f"recall {m['per_class']['oil']['recall']:.4f}")
    if aux_metrics:
        print("\nCLASSIFICATION (auxiliary head, binary dataset)")
        print(f"  accuracy {aux_metrics['pixel_accuracy']:.4f}   "
              f"oil recall {aux_metrics['per_class']['oil']['recall']:.4f}")
    print(f"\n  metrics -> {a.metrics}")
    print(f"  plot    -> {a.plot}")
    print(f"  weights -> {ckpt}")
    if not representative:
        print("\n  REMINDER: these numbers are not representative. Run the full")
        print("  training on a GPU before quoting any of them.")
    print("=" * 78)


if __name__ == "__main__":
    main()
