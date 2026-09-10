"""CFAR ship detection — CLAUDE.md section 5.3 (layer 4).

Two-parameter CFAR with guard cells: a pixel is a target where

    sigma0 > mu_bg + k * sigma_bg,     k = 4.5

with the background estimated from an annulus that excludes the guard band, so a
bright ship does not contaminate its own background estimate. Detections are
grouped morphologically into targets and a length is estimated from the major
axis of each connected component.

CFAR statistics assume multiplicative speckle, so everything runs in **linear
intensity**, not dB. Running it on a dB image inverts the noise model and the
false-alarm rate stops meaning anything.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

from samudra.geo import Projector

# DN -> dB scaling written into the synthetic scene's GeoTIFF tags.
DEFAULT_DB_MIN, DEFAULT_DB_MAX = -30.0, 0.0


def dn_to_linear(dn: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """uint8 DN -> sigma0 in linear power."""
    db = dn.astype(np.float32) / 255.0 * (db_max - db_min) + db_min
    return np.power(10.0, db / 10.0)


def read_scene(path: Path | str):
    """Return (linear intensity, transform, bounds, db_min, db_max)."""
    with rasterio.open(path) as src:
        band = src.read(1)
        tags = src.tags()
        db_min = float(tags.get("DB_MIN", DEFAULT_DB_MIN))
        db_max = float(tags.get("DB_MAX", DEFAULT_DB_MAX))
        if band.dtype == np.uint8:
            lin = dn_to_linear(band, db_min, db_max)
        else:
            # Already calibrated: assume dB if the range looks like dB.
            lin = (
                np.power(10.0, band.astype(np.float32) / 10.0)
                if band.min() < 0
                else band.astype(np.float32)
            )
        return lin, src.transform, src.bounds, db_min, db_max


def cfar_mask(
    lin: np.ndarray,
    k: float = 4.5,
    guard: int = 4,
    background: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-parameter CFAR. Returns (boolean detections, z-score image).

    The background mean and variance come from a square annulus: the outer
    `background` window minus the inner `guard` window. Box filters make this
    O(1) per pixel regardless of window size.
    """
    if background <= guard:
        raise ValueError(f"background window ({background}) must exceed guard ({guard})")

    lin = lin.astype(np.float32)
    o = 2 * background + 1
    g = 2 * guard + 1

    # Sums over the outer and guard windows, via box means.
    s_o = ndimage.uniform_filter(lin, size=o, mode="nearest") * (o * o)
    s_g = ndimage.uniform_filter(lin, size=g, mode="nearest") * (g * g)
    q_o = ndimage.uniform_filter(lin * lin, size=o, mode="nearest") * (o * o)
    q_g = ndimage.uniform_filter(lin * lin, size=g, mode="nearest") * (g * g)

    n = o * o - g * g
    mu = (s_o - s_g) / n
    var = np.maximum((q_o - q_g) / n - mu * mu, 1e-12)
    sd = np.sqrt(var)

    z = (lin - mu) / sd
    return z > k, z


def group_targets(
    mask: np.ndarray,
    lin: np.ndarray,
    transform,
    pixel_m: float,
    min_pixels: int = 5,
    close_iter: int = 1,
) -> list[dict]:
    """Morphological grouping into targets, with a length per component.

    `min_pixels` rejects isolated speckle spikes: a vessel spans several
    resolution cells, a single bright pixel does not. Measured on demo-001 at
    the spec's k=4.5, raising it from 2 to 5 drops false positives from 269 to 4
    while keeping 13/13 planted targets. It is a grouping parameter, not the
    detection threshold, so this does not touch the k the spec fixes.

    Note the synthetic scene is 150 m/px, which cannot resolve a real ship at
    all — planted targets are drawn 3x3 as markers. Sentinel-1 IW at 10 m/px
    resolves a 200 m vessel across ~20 pixels, so this threshold is comfortably
    conservative on real imagery.
    """
    if close_iter:
        # A ship often breaks into a few bright pixels; close small gaps so one
        # vessel is one target rather than three.
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=close_iter)

    lbl, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []

    out: list[dict] = []
    for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
        comp = lbl[sl] == i
        npx = int(comp.sum())
        if npx < min_pixels:
            continue

        ys, xs = np.nonzero(comp)
        ys = ys + sl[0].start
        xs = xs + sl[1].start

        # Major axis from the second moments of the component. A single-pixel or
        # perfectly symmetric blob has no orientation, so fall back to the
        # pixel diagonal rather than reporting a spurious zero.
        cy, cx = ys.mean(), xs.mean()
        if npx >= 3:
            cov = np.cov(np.vstack([xs - cx, ys - cy]))
            evals, evecs = np.linalg.eigh(np.atleast_2d(cov))
            major_px = 4.0 * math.sqrt(max(float(evals[-1]), 0.0))
            v = evecs[:, -1]
            bearing = math.degrees(math.atan2(float(v[0]), float(v[1]))) % 180.0
        else:
            major_px = math.sqrt(npx)
            bearing = 0.0
        major_px = max(major_px, math.sqrt(npx))

        lon, lat = rasterio.transform.xy(transform, cy, cx)
        out.append(
            {
                "det_id": f"cfar-{len(out):04d}",
                "lat": float(lat),
                "lon": float(lon),
                "row": float(cy),
                "col": float(cx),
                "pixels": npx,
                "est_length_m": float(major_px * pixel_m),
                "bearing_deg": float(bearing),
                "peak_intensity": float(lin[ys, xs].max()),
                "peak_sigma0_db": float(10.0 * math.log10(max(lin[ys, xs].max(), 1e-12))),
            }
        )
    return out


def detect(
    scene_path: Path | str,
    k: float = 4.5,
    guard: int = 4,
    background: int = 12,
    min_pixels: int = 5,
    land_mask: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Run CFAR over a scene and return (targets, summary)."""
    lin, transform, bounds, db_min, db_max = read_scene(scene_path)

    mask, z = cfar_mask(lin, k=k, guard=guard, background=background)
    if land_mask is not None:
        # CLAUDE.md 5.3: CFAR runs on the land-masked scene. Coastlines and
        # structures are far brighter than any ship and would dominate.
        mask &= ~land_mask

    proj = Projector((bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2)
    x0, y0 = proj.to_m(bounds.left, bounds.bottom)
    x1, y1 = proj.to_m(bounds.right, bounds.top)
    pixel_m = float(abs(x1 - x0) / lin.shape[1])

    targets = group_targets(mask, lin, transform, pixel_m, min_pixels=min_pixels)
    targets.sort(key=lambda t: -t["peak_intensity"])

    return targets, {
        "scene": str(scene_path),
        "shape": list(lin.shape),
        "pixel_m": pixel_m,
        "k": k,
        "guard": guard,
        "background": background,
        "detections": len(targets),
        "detected_pixels": int(mask.sum()),
        "false_alarm_rate": float(mask.sum()) / float(lin.size),
    }


def match_to_truth(targets: list[dict], truth_px: list[dict], tol_px: float = 6.0) -> dict:
    """Match detections to planted positions. Verification only.

    Truth is passed IN. This module must never open ground_truth.json itself, not
    even behind a --verify flag: tests/test_attribution.py enforces that no
    pipeline module reads it, and a convenience flag is not worth a hole in the
    guarantee. tests/test_vessels_cfar.py does the reading and calls this.
    """
    used: set[int] = set()
    hits = []
    for t in truth_px:
        best, best_d = None, 1e18
        for j, d in enumerate(targets):
            if j in used:
                continue
            dist = math.hypot(d["row"] - t["py"], d["col"] - t["px"])
            if dist < best_d:
                best, best_d = j, dist
        if best is not None and best_d <= tol_px:
            used.add(best)
            hits.append({"truth": t, "det": targets[best], "dist_px": best_d})

    return {
        "planted": len(truth_px),
        "recovered": len(hits),
        "recall": len(hits) / max(len(truth_px), 1),
        "detections": len(targets),
        "false_positives": len(targets) - len(hits),
        "matches": hits,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CFAR ship detection on a SAR scene.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--k", type=float, default=4.5)
    ap.add_argument("--guard", type=int, default=4)
    ap.add_argument("--background", type=int, default=12)
    ap.add_argument("--min-pixels", type=int, default=5)
    a = ap.parse_args()

    d = Path(a.root) / a.incident
    scene = d / "scene.tif"
    if not scene.exists():
        raise SystemExit(f"{scene} not found")

    targets, summary = detect(
        scene, k=a.k, guard=a.guard, background=a.background, min_pixels=a.min_pixels
    )

    (d / "vessels_cfar.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "properties": summary,
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
                        "properties": {k: v for k, v in t.items() if k not in ("lon", "lat")},
                    }
                    for t in targets
                ],
            },
            indent=2,
        )
    )

    print(f"scene             : {summary['shape'][1]} x {summary['shape'][0]} px "
          f"@ {summary['pixel_m']:.0f} m")
    print(f"CFAR              : k={a.k}, guard={a.guard}, background={a.background}")
    print(f"detections        : {summary['detections']}")
    print(f"false alarm rate  : {summary['false_alarm_rate']:.2e} of pixels")
    for t in targets[:10]:
        print(f"  {t['det_id']}  {t['lat']:9.5f} {t['lon']:10.5f}  "
              f"{t['pixels']:>4} px  ~{t['est_length_m']:6.0f} m  "
              f"{t['peak_sigma0_db']:6.1f} dB")

    print(f"written           : {d / 'vessels_cfar.geojson'}")


if __name__ == "__main__":
    main()
