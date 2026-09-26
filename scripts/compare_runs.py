"""Compare two training runs on the held-out test split.

    python scripts/compare_runs.py training_runs/metrics_baseline.json \
                                   artifacts/model_metrics.json

Exists so "the model improved" is a claim someone can check rather than a
sentence in a slide. It refuses to compare runs that were not evaluated the same
way: different test-set sizes or a different selection protocol means the two
numbers are not measuring the same thing, and printing them side by side would
imply they are.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(p: Path) -> dict:
    if not p.exists():
        raise SystemExit(f"{p} not found — run scripts/train_segmenter.py first")
    return json.loads(p.read_text())


def row(name: str, a, b, better: str = "up", width: int = 34) -> str:
    if a is None or b is None:
        return f"{name:<{width}} {'—':>10} {'—':>10}"
    d = b - a
    arrow = "" if abs(d) < 5e-4 else ("▲" if (d > 0) == (better == "up") else "▼")
    return f"{name:<{width}} {a:>10.4f} {b:>10.4f}   {d:>+8.4f} {arrow}"


def main() -> None:
    # Windows falls back to cp1252 the moment stdout is a pipe, and this prints
    # arrows and en dashes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Before/after on the held-out test split.")
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    a, b = load(ap.parse_args().before), load(ap.parse_args().after)

    print("=" * 74)
    print("HELD-OUT TEST SPLIT — evaluated once per run, after checkpoint selection")
    print("=" * 74)
    for label, m in (("before", a), ("after", b)):
        print(f"  {label:<7} {m.get('tag') or '(untagged)'}")
        print(f"          {m.get('epochs_run')} epochs, best epoch "
              f"{m.get('best_epoch')}, dice_weight {m.get('dice_weight')}, "
              f"weight_power {m.get('weight_power')}, threshold "
              f"{m.get('decision_threshold')}")

    sa, sb = a.get("segmentation", {}), b.get("segmentation", {})
    if sa.get("confusion_matrix") and sb.get("confusion_matrix"):
        na = sum(sum(r) for r in sa["confusion_matrix"])
        nb = sum(sum(r) for r in sb["confusion_matrix"])
        if na != nb:
            print(f"\n!! test pixel counts differ ({na:,} vs {nb:,}) — the two runs "
                  f"did not see the same test set, so these are not comparable.")
            sys.exit(2)

    print(f"\n{'metric':<34} {'before':>10} {'after':>10}   {'delta':>8}")
    print("-" * 74)
    oa, ob = sa.get("per_class", {}).get("oil", {}), sb.get("per_class", {}).get("oil", {})
    print(row("oil IoU", oa.get("iou"), ob.get("iou")))
    print(row("oil Dice", oa.get("dice"), ob.get("dice")))
    print(row("oil precision", oa.get("precision"), ob.get("precision")))
    print(row("oil recall", oa.get("recall"), ob.get("recall")))
    print(row("oil F1", oa.get("f1"), ob.get("f1")))
    ba, bb = sa.get("per_class", {}).get("background", {}), sb.get("per_class", {}).get("background", {})
    print(row("background IoU", ba.get("iou"), bb.get("iou")))
    print(row("mean IoU", sa.get("mean_iou"), sb.get("mean_iou")))
    print(row("mean Dice", sa.get("mean_dice"), sb.get("mean_dice")))
    print(row("pixel accuracy", sa.get("pixel_accuracy"), sb.get("pixel_accuracy")))

    print("\nper sensor (oil IoU)")
    for s in sorted(set(sa.get("per_sensor", {})) | set(sb.get("per_sensor", {}))):
        pa = sa.get("per_sensor", {}).get(s, {}).get("per_class", {}).get("oil", {})
        pb = sb.get("per_sensor", {}).get(s, {}).get("per_class", {}).get("oil", {})
        print(row(f"  {s}", pa.get("iou"), pb.get("iou")))

    va, vb = a.get("validation", {}), b.get("validation", {})
    if va and vb:
        print("\nvalidation (selection split — NOT a held-out estimate)")
        print(row("  oil IoU", va.get("per_class", {}).get("oil", {}).get("iou"),
                  vb.get("per_class", {}).get("oil", {}).get("iou")))

    print("\nA higher oil recall at a much lower precision is not an improvement:")
    print("it means the model is calling more of the scene oil. Read the pair.")


if __name__ == "__main__":
    main()
