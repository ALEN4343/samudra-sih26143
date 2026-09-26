"""Run the installed checkpoint over an incident's scene.tif. READ ONLY.

    python scripts/eval_scene.py --incident demo-001

Writes nothing into artifacts/<incident>/. The figure and the JSON summary go to
--out, which defaults to training_runs/, so a demo incident that is serving
as a regression baseline stays byte-identical.

The planted slick is read AFTER inference, only to report how close the model
came. It is not on the inference path: `segmenter.segment()` is given the raster
and the environment field, and the only thing it takes from
observed_slick.geojson is the acquisition timestamp for the wind gate. The IoU
printed at the end is post-hoc evaluation, which is the honest way to answer
"is this credible" — it is not, and must never become, an input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samudra.detection import polygonize, segmenter  # noqa: E402
from samudra.geo import Projector, load_polygon  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Read-only inference over a scene.")
    ap.add_argument("--incident", default="demo-001")
    ap.add_argument("--root", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("training_runs"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-gates", action="store_true")
    a = ap.parse_args()

    d = a.root / a.incident
    if not (d / "scene.tif").exists():
        raise SystemExit(f"{d / 'scene.tif'} not found")

    thr = a.threshold
    if thr is None:
        m = Path("artifacts/model_metrics.json")
        thr = float(json.loads(m.read_text()).get("decision_threshold", 0.5)) \
            if m.exists() else 0.5

    print(f"incident      : {a.incident}   (READ ONLY — nothing is written here)")
    print(f"threshold     : {thr:.2f}  (selected on validation during training)")

    res = segmenter.segment(d, threshold=thr, apply_gates=not a.no_gates)
    meta = res["checkpoint_meta"]
    print(f"checkpoint    : epoch {meta.get('epoch')}, "
          f"val oil IoU {meta.get('val_oil_iou', meta.get('oil_iou'))}, "
          f"representative {meta.get('representative')}")

    prob, mask, db = res["prob"], res["mask"], res["db"]
    print(f"scene         : {db.shape[1]}x{db.shape[0]} px")
    print(f"anomaly px    : {mask.sum():,} of {mask.size:,} "
          f"= {mask.mean() * 100:.3f}%")
    print(f"probability   : max {prob.max():.4f}  mean-in-mask "
          f"{(prob[mask].mean() if mask.any() else 0):.4f}")
    for k, v in (res.get("gates") or {}).items():
        if isinstance(v, (int, float)):
            print(f"  gate {k:<18} {v:.4f}")

    proj = Projector.for_bounds(res["bounds"])
    polys = polygonize.mask_to_polygons(mask, res["transform"], proj=proj)
    recs = polygonize.describe(polys, proj, db=db, prob=prob,
                               transform=res["transform"], scene_id=a.incident)
    recs.sort(key=lambda r: r["area_km2"], reverse=True)
    print(f"\npolygons      : {len(recs)}")
    for r in recs[:5]:
        print(f"  {r['slick_id']}  {r['area_km2']:8.2f} km2  "
              f"axis {r['major_axis_deg']:5.1f} deg  ecc {r['eccentricity']:.3f}  "
              f"cnn {r.get('cnn_oil_prob', float('nan')):.3f}  "
              f"contrast {r.get('contrast_db', float('nan')):.2f} dB")

    # ---- post-hoc credibility check against the planted slick --------------
    truth_p = d / "observed_slick.geojson"
    comparison = None
    if truth_p.exists() and recs:
        truth = load_polygon(json.loads(truth_p.read_text()))
        best, best_iou = None, -1.0
        for r in recs:
            from shapely.geometry import shape

            g = shape(r["polygon"])
            i = proj.polygon_to_m(g).intersection(proj.polygon_to_m(truth)).area
            u = proj.polygon_to_m(g).union(proj.polygon_to_m(truth)).area
            v = i / u if u else 0.0
            if v > best_iou:
                best, best_iou = r, v
        tm = __import__("samudra.geo", fromlist=["polygon_metrics"]).polygon_metrics(proj, truth)
        comparison = {
            "planted_area_km2": tm["area_km2"],
            "planted_axis_deg": tm["major_axis_deg"],
            "best_detected_area_km2": best["area_km2"],
            "best_detected_axis_deg": best["major_axis_deg"],
            "iou_vs_planted": best_iou,
            "centroid_offset_km": float(
                np.hypot(*np.subtract(
                    proj.to_m(best["centroid_lon"], best["centroid_lat"]),
                    proj.to_m(tm["centroid_lon"], tm["centroid_lat"]))) / 1000.0),
        }
        print("\nPOST-HOC CHECK vs the planted synthetic slick")
        print("  (evaluation only — this was NOT available to the model)")
        print(f"  planted     {tm['area_km2']:8.2f} km2  axis {tm['major_axis_deg']:5.1f} deg")
        print(f"  detected    {best['area_km2']:8.2f} km2  axis {best['major_axis_deg']:5.1f} deg")
        print(f"  IoU {best_iou:.3f}   centroid offset "
              f"{comparison['centroid_offset_km']:.2f} km")

    a.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "incident": a.incident,
        "read_only": True,
        "threshold": thr,
        "checkpoint": {k: v for k, v in meta.items()
                       if isinstance(v, (str, int, float, bool, type(None)))},
        "anomaly_pixel_fraction": float(mask.mean()),
        "max_probability": float(prob.max()),
        "mean_probability_in_mask": float(prob[mask].mean()) if mask.any() else 0.0,
        "gates": {k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in (res.get("gates") or {}).items()},
        "polygons": [{k: v for k, v in r.items() if k != "polygon"} for r in recs],
        "post_hoc_vs_planted": comparison,
        "post_hoc_note": (
            "The planted slick was read after inference, to report accuracy. It "
            "was never an input; see scripts/eval_scene.py."
        ),
    }
    (a.out / f"scene_eval_{a.incident}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- figure ------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), dpi=140)
    ax[0].imshow(db, cmap="gray"); ax[0].set_title("scene.tif (VV, dB)", fontsize=9)
    im = ax[1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    ax[1].set_title("model oil probability", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title(f"mask at t={thr:.2f}  ({mask.mean()*100:.2f}% of scene)", fontsize=9)

    rgb = np.stack([np.clip((db + 35) / 35, 0, 1)] * 3, -1)
    rgb[mask] = 0.4 * rgb[mask] + 0.6 * np.array([1.0, 0.42, 0.29])
    ax[3].imshow(rgb); ax[3].set_title("overlay", fontsize=9)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"{a.incident} — installed checkpoint on scene.tif (read only)",
                 fontsize=10)
    fig.tight_layout()
    out_png = a.out / f"scene_eval_{a.incident}.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {a.out / f'scene_eval_{a.incident}.json'}")
    print(f"wrote {out_png}")
    print("\nNothing in artifacts/%s was modified." % a.incident)


if __name__ == "__main__":
    main()
