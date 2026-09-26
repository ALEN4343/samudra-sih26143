"""Scene preprocessing — CLAUDE.md section 5.2 (layer 2).

Land mask, wind gate, and tiling. Everything downstream — the segmenter and CFAR
— runs on the output of this module, not on raw ingestion.

The wind gate is not a nicety. Below 3 m/s the sea is glassy and every calm patch
reads as a slick; above 10 m/s wind roughening scrubs the signature out entirely.
Pixels outside that band are masked rather than scored, because a detector run on
them produces confident nonsense.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

from samudra.geo import Projector

TILE_PX = 512
TILE_OVERLAP_PX = 64
LAND_BUFFER_M = 500.0


def scene_db(path: Path | str) -> tuple[np.ndarray, object, object]:
    """Read a scene as sigma0 in dB, with its transform and bounds."""
    with rasterio.open(path) as src:
        band = src.read(1)
        tags = src.tags()
        if band.dtype == np.uint8:
            db_min = float(tags.get("DB_MIN", -30.0))
            db_max = float(tags.get("DB_MAX", 0.0))
            db = band.astype(np.float32) / 255.0 * (db_max - db_min) + db_min
        else:
            db = band.astype(np.float32)
        return db, src.transform, src.bounds


def land_mask(
    bounds,
    shape: tuple[int, int],
    transform,
    buffer_m: float = LAND_BUFFER_M,
) -> np.ndarray:
    """Rasterise the coastline with a buffer. True where land.

    Returns all-False when the loaded coastline does not cover this AOI, rather
    than masking the whole scene or none of it by accident.
    """
    from samudra.impact.coastline import COVERAGE_LIMIT_KM, load_coastline

    coast = load_coastline()
    geom = coast["geometry"]

    proj = Projector((bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2)
    minx, miny, maxx, maxy = geom.bounds
    # Cheap rejection: coastline entirely elsewhere.
    if maxx < bounds.left - 10 or minx > bounds.right + 10 or \
       maxy < bounds.bottom - 10 or miny > bounds.top + 10:
        return np.zeros(shape, dtype=bool)

    # Buffer in degrees, approximated at this latitude only for the buffer width.
    lat_mid = (bounds.bottom + bounds.top) / 2
    deg = buffer_m / (111_320.0 * max(np.cos(np.radians(lat_mid)), 0.2))

    # Clip to the scene before buffering. A real Natural Earth file holds the
    # WHOLE WORLD's coastline, and buffering all of it to rasterise one scene
    # took this single call from milliseconds to 142 s — measured, it was the
    # slowest thing in the test suite by two orders of magnitude. The pad is the
    # buffer itself, so coast just outside the scene still thickens into it and
    # the mask is identical to the unclipped one.
    from shapely.geometry import box

    pad = deg * 2
    window = box(bounds.left - pad, bounds.bottom - pad,
                 bounds.right + pad, bounds.top + pad)
    try:
        geom = geom.intersection(window)
    except Exception:  # noqa: BLE001 - topology error on a pathological input
        pass
    if geom.is_empty:
        return np.zeros(shape, dtype=bool)

    buffered = geom.buffer(deg)
    if buffered.is_empty:
        return np.zeros(shape, dtype=bool)

    return rasterize(
        [(buffered, 1)], out_shape=shape, transform=transform, dtype=np.uint8
    ).astype(bool)


def wind_gate(
    env,
    t_epoch: float,
    bounds,
    shape: tuple[int, int],
    lo: float = 3.0,
    hi: float = 10.0,
) -> tuple[np.ndarray, dict]:
    """True where wind is inside the detectable band. Also returns a summary."""
    ny, nx = shape
    # Sample the env grid coarsely and upsample: wind varies on tens of km, not
    # per pixel, and sampling per pixel on a 13 MP scene is pure waste.
    cy = np.linspace(bounds.bottom, bounds.top, 64)
    cx = np.linspace(bounds.left, bounds.right, 64)
    gx, gy = np.meshgrid(cx, cy)
    wu, wv, _, _ = env.sample(t_epoch, gy.ravel(), gx.ravel())
    speed = np.hypot(wu, wv).reshape(64, 64)

    ok = (speed >= lo) & (speed <= hi)
    full = ndimage.zoom(ok.astype(np.float32), (ny / 64, nx / 64), order=0)[:ny, :nx]
    if full.shape != (ny, nx):  # zoom rounding
        pad = np.zeros((ny, nx), dtype=np.float32)
        pad[: full.shape[0], : full.shape[1]] = full
        full = pad
    # Grid rows run south->north; raster rows run north->south.
    full = np.flipud(full)

    return full > 0.5, {
        "wind_min_ms": float(speed.min()),
        "wind_max_ms": float(speed.max()),
        "wind_mean_ms": float(speed.mean()),
        "gate_lo": lo,
        "gate_hi": hi,
        "fraction_in_band": float(ok.mean()),
    }


def tiles(shape: tuple[int, int], size: int = TILE_PX, overlap: int = TILE_OVERLAP_PX):
    """Yield (y0, y1, x0, x1) windows covering the scene with overlap."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile size ({size})")
    ny, nx = shape
    step = size - overlap
    ys = list(range(0, max(ny - overlap, 1), step))
    xs = list(range(0, max(nx - overlap, 1), step))
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + size, ny), min(x0 + size, nx)
            yield (max(y1 - size, 0), y1, max(x1 - size, 0), x1)


def stitch(shape: tuple[int, int], pieces) -> np.ndarray:
    """Overlap-average tile outputs back into a full-scene map.

    Averaging rather than last-write-wins: tile seams otherwise appear as hard
    steps in the probability map and survive into the polygon boundary, which is
    exactly the geometry attribution depends on.
    """
    acc = np.zeros(shape, dtype=np.float32)
    cnt = np.zeros(shape, dtype=np.float32)
    for (y0, y1, x0, x1), arr in pieces:
        acc[y0:y1, x0:x1] += arr
        cnt[y0:y1, x0:x1] += 1.0
    return acc / np.maximum(cnt, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess a scene: land mask, wind gate, tiling.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    a = ap.parse_args()

    from samudra.attribution import drift

    d = Path(a.root) / a.incident
    db, transform, bounds = scene_db(d / "scene.tif")
    slick_gj = json.loads((d / "observed_slick.geojson").read_text())
    from datetime import datetime

    acq = datetime.fromisoformat(slick_gj["features"][0]["properties"]["acquisition_at"])

    land = land_mask(bounds, db.shape, transform)
    env = drift.EnvField.load(d / "env.npz")
    gate, wsum = wind_gate(env, acq.timestamp(), bounds, db.shape)
    n_tiles = sum(1 for _ in tiles(db.shape))

    usable = (~land) & gate
    print(f"scene             : {db.shape[1]} x {db.shape[0]} px, "
          f"sigma0 {db.min():.1f} to {db.max():.1f} dB")
    print(f"land mask         : {land.mean() * 100:.2f}% of pixels "
          f"(buffer {LAND_BUFFER_M:.0f} m)")
    print(f"wind              : {wsum['wind_min_ms']:.1f}-{wsum['wind_max_ms']:.1f} m/s, "
          f"{wsum['fraction_in_band'] * 100:.0f}% inside the {wsum['gate_lo']:.0f}-"
          f"{wsum['gate_hi']:.0f} m/s gate")
    print(f"usable for detect : {usable.mean() * 100:.2f}% of pixels")
    print(f"tiling            : {n_tiles} tiles of {TILE_PX} px, {TILE_OVERLAP_PX} px overlap")


if __name__ == "__main__":
    main()
