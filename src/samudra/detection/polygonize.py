"""Mask to slick polygons with geometric features — CLAUDE.md section 5.2.

This is an explicit stage, not a formality. `area_km2`, `perimeter_km`,
`shape_complexity`, `eccentricity` and `major_axis_deg` are computed here, and
every downstream module depends on them — the attribution score is 25%
orientation and the drift model consumes the polygon directly. Detection is not
finished until these features exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from scipy import ndimage
from shapely.geometry import mapping, shape

from samudra.geo import Projector, polygon_metrics

MIN_AREA_KM2 = 0.5  # below this a "slick" is a speckle cluster, not a discharge


def mask_to_polygons(
    mask: np.ndarray,
    transform,
    min_area_km2: float = MIN_AREA_KM2,
    smooth_px: int = 2,
    proj: Projector | None = None,
) -> list:
    """Binary mask -> list of shapely polygons in EPSG:4326."""
    if smooth_px:
        # Close pinholes and shave single-pixel spurs. Both inflate perimeter,
        # and perimeter drives shape_complexity.
        m = ndimage.binary_closing(mask, np.ones((smooth_px * 2 + 1,) * 2))
        m = ndimage.binary_opening(m, np.ones((smooth_px * 2 + 1,) * 2))
    else:
        m = mask

    out = []
    for geom, val in rio_shapes(m.astype(np.uint8), mask=m, transform=transform):
        if val != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        if proj is not None:
            if proj.polygon_to_m(poly).area / 1e6 < min_area_km2:
                continue
        out.append(poly)

    out.sort(key=lambda p: -p.area)
    return out


def describe(
    polys: list,
    proj: Projector,
    db: np.ndarray | None = None,
    prob: np.ndarray | None = None,
    transform=None,
    scene_id: str = "scene",
) -> list[dict]:
    """Attach the SlickDetection geometric features to each polygon."""
    out = []
    bg_db = float(np.median(db)) if db is not None else None

    for i, poly in enumerate(polys):
        m = polygon_metrics(proj, poly)
        rec = {
            "slick_id": f"slick-{i:03d}",
            "scene_id": scene_id,
            "polygon": mapping(poly),
            "area_km2": m["area_km2"],
            "perimeter_km": m["perimeter_km"],
            "shape_complexity": m["shape_complexity"],
            "eccentricity": m["eccentricity"],
            "major_axis_deg": m["major_axis_deg"],
            "centroid_lat": m["centroid_lat"],
            "centroid_lon": m["centroid_lon"],
        }

        if db is not None and transform is not None:
            inside = rasterio.features.geometry_mask(
                [mapping(poly)], out_shape=db.shape, transform=transform, invert=True
            )
            if inside.any():
                rec["mean_sigma0_db"] = float(db[inside].mean())
                rec["contrast_db"] = float(bg_db - db[inside].mean())
                # Edge gradient: how sharply backscatter changes at the boundary.
                # A real slick has a crisp edge; a low-wind look-alike does not.
                edge = ndimage.binary_dilation(inside, iterations=2) & ~ndimage.binary_erosion(
                    inside, iterations=2
                )
                gy, gx = np.gradient(db)
                rec["edge_gradient"] = (
                    float(np.hypot(gy, gx)[edge].mean()) if edge.any() else 0.0
                )
                if prob is not None:
                    rec["cnn_oil_prob"] = float(prob[inside].mean())

        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Polygonize a segmentation mask.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area-km2", type=float, default=MIN_AREA_KM2)
    a = ap.parse_args()

    from samudra.detection.preprocess import scene_db

    d = Path(a.root) / a.incident
    prob_p = d / "oil_prob.npy"
    if not prob_p.exists():
        raise SystemExit(
            f"{prob_p} not found — run 'python -m samudra.detection.segmenter "
            f"--incident {a.incident}' first"
        )

    prob = np.load(prob_p)
    db, transform, bounds = scene_db(d / "scene.tif")
    proj = Projector((bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2)

    polys = mask_to_polygons(prob >= a.threshold, transform, a.min_area_km2, proj=proj)
    recs = describe(polys, proj, db=db, prob=prob, transform=transform, scene_id=a.incident)

    (d / "detected_slicks.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": r.pop("polygon"), "properties": r}
                    for r in [dict(x) for x in recs]
                ],
            },
            indent=2,
        )
    )

    print(f"threshold         : {a.threshold}")
    print(f"polygons          : {len(recs)} above {a.min_area_km2} km2")
    for r in recs[:8]:
        print(f"  {r['slick_id']}  {r['area_km2']:8.2f} km2  "
              f"axis {r['major_axis_deg']:5.0f} deg  ecc {r['eccentricity']:.3f}  "
              f"complexity {r['shape_complexity']:.2f}  "
              f"contrast {r.get('contrast_db', float('nan')):5.1f} dB")
    print(f"written           : {d / 'detected_slicks.geojson'}")


if __name__ == "__main__":
    main()
