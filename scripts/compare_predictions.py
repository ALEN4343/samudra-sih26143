"""Visual before/after: one row per chip, two models, on the held-out test split.

    python scripts/compare_predictions.py \
        --old training_runs/seg_baseline.pt \
        --old-metrics training_runs/metrics_baseline.json \
        --new data/models/seg.pt \
        --new-metrics artifacts/model_metrics.json \
        --out training_runs/prediction_comparison.png

Columns: SAR chip | ground truth | old prediction | new prediction | overlay of
the new prediction (green correct, red false positive, blue missed).

Chip selection is deliberately not "the ones where we improved". It takes the
biggest gains, the biggest REGRESSIONS, and a couple of typical cases, so the
figure can show the new model losing where it loses. A comparison that only
contains wins is marketing, not evidence.

Each model is evaluated at its OWN validation-selected threshold, because that
is the operating point each one would actually ship with. Forcing both to 0.5
would compare neither.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samudra.detection.datasets import SOSSegmentation  # noqa: E402
from samudra.detection.train import DeepLabV3Plus  # noqa: E402


def load(ckpt: Path, device):
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = DeepLabV3Plus(num_classes=blob.get("num_classes", 2), pretrained=False)
    m.load_state_dict(blob["model"])
    return m.to(device).eval(), blob


def threshold_of(metrics: Path) -> float:
    try:
        return float(json.loads(metrics.read_text()).get("decision_threshold", 0.5))
    except Exception:  # noqa: BLE001
        return 0.5


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(((pred == 1) & (gt == 1)).sum())
    union = float(((pred == 1) | (gt == 1)).sum())
    return inter / union if union else 1.0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Side-by-side prediction comparison.")
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--old-metrics", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--new-metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("training_runs/prediction_comparison.png"))
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old, old_blob = load(a.old, device)
    new, new_blob = load(a.new, device)
    t_old, t_new = threshold_of(a.old_metrics), threshold_of(a.new_metrics)
    print(f"old  {a.old}  epoch {old_blob.get('epoch')}  threshold {t_old:.2f}")
    print(f"new  {a.new}  epoch {new_blob.get('epoch')}  threshold {t_new:.2f}")

    ds = SOSSegmentation(split="test", augment=False)
    rng = np.random.default_rng(a.seed)
    pool = rng.choice(len(ds), size=min(len(ds), a.sample), replace=False)

    scored = []
    with torch.no_grad():
        for i in pool:
            s = ds[int(i)]
            gt = s["mask"].numpy()
            frac = float(gt.mean())
            if not (0.01 <= frac <= 0.99):     # same rule as the panel
                continue
            x = s["image"].unsqueeze(0).to(device)
            po = torch.softmax(old(x), 1)[0, 1].cpu().numpy()
            pn = torch.softmax(new(x), 1)[0, 1].cpu().numpy()
            mo = (po >= t_old).astype(np.uint8)
            mn = (pn >= t_new).astype(np.uint8)
            scored.append({
                "i": int(i), "gt": gt, "old": mo, "new": mn,
                "iou_old": iou(mo, gt), "iou_new": iou(mn, gt),
                "sensor": s["sensor"],
            })
    if not scored:
        raise SystemExit("no chips with a visible slick in this sample")

    for r in scored:
        r["delta"] = r["iou_new"] - r["iou_old"]
    scored.sort(key=lambda r: r["delta"])

    n = min(a.rows, len(scored))
    n_reg = min(2, max(1, n // 3))                     # worst regressions
    chosen = scored[:n_reg] + scored[-(n - n_reg):]    # then the biggest gains
    chosen.sort(key=lambda r: r["delta"])

    mean_old = float(np.mean([r["iou_old"] for r in scored]))
    mean_new = float(np.mean([r["iou_new"] for r in scored]))
    n_better = sum(1 for r in scored if r["delta"] > 0.01)
    n_worse = sum(1 for r in scored if r["delta"] < -0.01)
    print(f"\nover {len(scored)} sampled chips containing a visible slick:")
    print(f"  mean per-chip IoU   old {mean_old:.4f}   new {mean_new:.4f}")
    print(f"  improved {n_better}   regressed {n_worse}   "
          f"unchanged {len(scored) - n_better - n_worse}")

    titles = ["SAR chip", "ground truth", f"old (t={t_old:.2f})",
              f"new (t={t_new:.2f})", "overlay — new"]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(13.5, 2.75 * len(chosen)), dpi=140)
    if len(chosen) == 1:
        axes = axes[None, :]

    for r, rec in enumerate(chosen):
        img_p, _, _ = ds.items[rec["i"]]
        raw = np.asarray(Image.open(img_p).convert("L"), dtype=np.uint8)
        axes[r, 0].imshow(raw, cmap="gray", vmin=0, vmax=255)
        axes[r, 1].imshow(rec["gt"], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(rec["old"], cmap="gray", vmin=0, vmax=1)
        axes[r, 3].imshow(rec["new"], cmap="gray", vmin=0, vmax=1)

        rgb = np.stack([raw] * 3, -1).astype(np.float32) / 255.0
        gt, pn = rec["gt"], rec["new"]
        tp, fp, fn = (pn == 1) & (gt == 1), (pn == 1) & (gt == 0), (pn == 0) & (gt == 1)
        rgb[tp] = 0.45 * rgb[tp] + 0.55 * np.array([0.25, 0.85, 0.35])
        rgb[fp] = 0.45 * rgb[fp] + 0.55 * np.array([0.95, 0.25, 0.20])
        rgb[fn] = 0.45 * rgb[fn] + 0.55 * np.array([0.30, 0.55, 1.00])
        axes[r, 4].imshow(np.clip(rgb, 0, 1))

        d = rec["delta"]
        axes[r, 0].set_ylabel(
            f"{rec['sensor']}\nold {rec['iou_old']:.3f}\nnew {rec['iou_new']:.3f}\n"
            f"{'+' if d >= 0 else ''}{d:.3f}",
            fontsize=8,
            color=("#1a7f37" if d > 0.01 else ("#b3261e" if d < -0.01 else "#555")),
        )
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)

    fig.suptitle(
        f"Held-out test split — {n_reg} largest regressions (top) and "
        f"{len(chosen) - n_reg} largest gains (bottom)\n"
        f"over {len(scored)} sampled chips: mean per-chip IoU {mean_old:.3f} → "
        f"{mean_new:.3f}, {n_better} improved / {n_worse} regressed\n"
        "overlay: green = correct oil, red = false positive, blue = missed oil",
        fontsize=9, y=1.0,
    )
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
