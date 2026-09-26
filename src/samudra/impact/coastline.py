"""Coastline intersection and ETA — CLAUDE.md layer 8.

Uses a real Natural Earth coastline when one is present on disk, and reports in
every result which source was actually used.

The TLS-interception problem that previously blocked the download is solved:
`truststore` routes verification through the Windows certificate store, which
trusts this machine's interception CA where Python's bundled CA set does not.
Verification stays ON — nothing here disables it. Fetch with
`scripts/fetch_coastline.py`.

Absent any file, a simplified **west-coast-only** outline is bundled so the
module still runs. That fallback cannot serve the east coast, the Andamans or
Lakshadweep, and `load_coastline()` says so in its `source` string rather than
letting a demo quietly report shoreline impacts against a coast that is not
there.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import unary_union

from samudra.geo import Projector

# Beyond this range the loaded coastline is simply not the coast this AOI sits on.
COVERAGE_LIMIT_KM = 800.0

# Most detailed first. Natural Earth's "10m" is 1:10,000,000 — finer than
# "50m", not coarser — and the difference is not cosmetic here: at 1:50m the
# Lakshadweep group is dropped entirely, putting the nearest modelled shore
# 71 km from Kavaratti instead of 0.3 km. Small islands are precisely the
# ecologically sensitive landfalls an impact ETA exists to warn about, so the
# extra 0.5 s of load is worth paying.
COASTLINE_PATHS = (
    Path("data/raw/ne_10m_coastline.geojson"),
    Path("data/raw/ne_50m_coastline.geojson"),
    Path("data/raw/coastline.geojson"),
)

# Indian west coast, Kanyakumari to the Gulf of Kutch, roughly 1:2,000,000.
# Longitude first. Adequate for a demo AOI of 68-73E, 15-20N, whose eastern edge
# runs within ~30 km of the Konkan shore.
_INDIA_WEST_COAST = [
    (77.54, 8.08), (77.10, 8.35), (76.94, 8.49), (76.60, 8.88), (76.32, 9.47),
    (76.24, 9.93), (76.18, 10.53), (75.93, 10.90), (75.78, 11.25), (75.55, 11.60),
    (75.37, 11.87), (74.98, 12.45), (74.84, 12.87), (74.68, 13.35), (74.45, 13.85),
    (74.30, 14.30), (74.13, 14.81), (73.96, 15.15), (73.83, 15.50), (73.66, 15.90),
    (73.48, 16.40), (73.31, 16.99), (73.15, 17.45), (73.02, 17.90), (72.92, 18.40),
    (72.83, 18.94), (72.78, 19.40), (72.75, 19.90), (72.83, 20.42), (72.70, 20.80),
    (72.65, 21.10), (72.40, 21.45), (72.15, 21.76), (72.00, 21.40), (71.60, 21.00),
    (71.20, 20.85), (70.98, 20.71), (70.60, 20.78), (70.37, 20.90), (70.00, 21.15),
    (69.75, 21.40), (69.60, 21.64), (69.30, 21.95), (68.97, 22.24), (69.08, 22.47),
]


@lru_cache(maxsize=4)
def _parse_coastline(path_str: str, mtime: float, size: int):
    """Parse one coastline file. Cached — see `load_coastline`.

    Keyed on (path, mtime, size) rather than path alone, so replacing the file
    on disk invalidates the entry instead of serving a stale geometry for the
    life of the process.
    """
    p = Path(path_str)
    gj = json.loads(p.read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    geoms = [shape(f["geometry"] if "geometry" in f else f) for f in feats]
    return unary_union(geoms)


def load_coastline() -> dict:
    """Return {'geometry': MultiLineString, 'source': str}.

    Parsing is cached: the 10m file is 10 MB and its `unary_union` costs ~0.6 s
    per call. Callers get a fresh dict each time, so the cached geometry is
    never handed out as a mutable shared object.

    The cache is not what made switching to a real coastline affordable, and it
    is worth recording which fix actually mattered. Profiling put 142 of 212
    test-suite seconds in a single call to `detection.preprocess.land_mask`,
    which buffers and rasterises the geometry — the whole world's, once a real
    file is present. Clipping to the AOI before buffering is the fix; caching
    the parse was worth a few seconds.
    """
    for p in COASTLINE_PATHS:
        if p.exists():
            st = p.stat()
            return {"geometry": _parse_coastline(str(p), st.st_mtime, st.st_size),
                    "source": f"Natural Earth ({p})"}

    return {
        "geometry": MultiLineString([LineString(_INDIA_WEST_COAST)]),
        "source": ("bundled simplified Indian WEST COAST ONLY — no Natural "
                   "Earth file on disk; east coast, Andamans and Lakshadweep "
                   "are not represented. Run scripts/fetch_coastline.py."),
    }


def _clip_near(geom, forecasts, limit_km: float):
    """Coastline within `limit_km` of the forecast polygons, in degrees.

    Returns the input unchanged when the clip would be empty, so an AOI with no
    coast nearby still reaches the out-of-coverage guard and is reported as
    "this coastline does not cover this AOI" rather than silently intersecting
    an empty geometry and reporting no shore contact.
    """
    from shapely.geometry import box

    xs, ys = [], []
    for f in forecasts:
        x0, y0, x1, y1 = f["geometry"].bounds
        xs += [x0, x1]
        ys += [y0, y1]
    # Degrees of longitude shrink with latitude; use the worst case in the AOI.
    lat = max(abs(min(ys)), abs(max(ys)))
    dlat = limit_km / 110.574
    dlon = limit_km / max(111.320 * math.cos(math.radians(min(lat, 89.0))), 1e-6)
    window = box(min(xs) - dlon, min(ys) - dlat, max(xs) + dlon, max(ys) + dlat)
    try:
        clipped = geom.intersection(window)
    except Exception:  # noqa: BLE001 - topology error on a pathological input
        return geom
    return geom if clipped.is_empty else clipped


def coastline_impacts(
    forecasts: list[dict],
    coast: dict,
    acquisition_at: datetime,
    proj: Projector | None = None,
) -> list[dict]:
    """Per-horizon coastline impact, with an interpolated shore ETA.

    ETA is estimated from the closing rate across the forecast series rather than
    reported as the first horizon that happens to intersect, which would quantise
    the answer to 24-hour steps.
    """
    if not forecasts:
        return []
    geom = coast["geometry"]
    if proj is None:
        c = forecasts[0]["geometry"].centroid
        proj = Projector(c.y, c.x)

    # Clip to the neighbourhood before projecting. A real Natural Earth file is
    # the WHOLE WORLD's coastline; projecting all of it to metres for a 50 km
    # AOI took the test suite from 60 s to 226 s and scales with nothing useful.
    # The window is COVERAGE_LIMIT_KM wide so the out-of-coverage guard below
    # still sees any coast it would have seen unclipped — clipping tighter than
    # that would manufacture a "no coastline here" answer.
    geom = _clip_near(geom, forecasts, COVERAGE_LIMIT_KM)
    coast_m = _to_m(proj, geom)

    # Guard against reporting a distance to a coastline on the wrong ocean. The
    # bundled fallback covers the Indian west coast only; run an AOI elsewhere
    # (Houston, say) and the honest answer is "no coastline data", not "no shore
    # contact, closest approach 15,000 km", which looks like a result and is not.
    ref = proj.polygon_to_m(forecasts[0]["geometry"])
    if ref.distance(coast_m) > COVERAGE_LIMIT_KM * 1000.0:
        return [
            {
                "horizon_hours": f["horizon_hours"],
                "coastline_intersects": False,
                "affected_shoreline_km": 0.0,
                "distance_to_coast_km": None,
                "coastline_source": f"{coast['source']} - DOES NOT COVER THIS AOI",
                "coastline_covers_aoi": False,
                "coastline_eta": None,
            }
            for f in forecasts
        ]

    rows = []
    for f in forecasts:
        poly_m = proj.polygon_to_m(f["geometry"])
        hits = poly_m.intersects(coast_m)
        seg = poly_m.intersection(coast_m) if hits else None
        rows.append(
            {
                "horizon_hours": f["horizon_hours"],
                "coastline_intersects": bool(hits),
                "affected_shoreline_km": (seg.length / 1000.0) if hits else 0.0,
                "distance_to_coast_km": 0.0 if hits else poly_m.distance(coast_m) / 1000.0,
                "coastline_source": coast["source"],
                "coastline_covers_aoi": True,
                "coastline_eta": None,
            }
        )

    eta = _estimate_eta(rows, acquisition_at)
    for r in rows:
        if r["coastline_intersects"] or eta is not None:
            r["coastline_eta"] = eta.isoformat() if eta else None
    return rows


def _estimate_eta(rows: list[dict], acquisition_at: datetime) -> datetime | None:
    """Linear interpolation of the moment distance-to-coast reaches zero."""
    if rows[0]["coastline_intersects"]:
        return acquisition_at

    first = next((i for i, r in enumerate(rows) if r["coastline_intersects"]), None)
    if first is None:
        return None

    prev = rows[first - 1]
    h0, d0 = prev["horizon_hours"], prev["distance_to_coast_km"]
    h1 = rows[first]["horizon_hours"]

    if first >= 2:
        # Closing rate from the previous interval, extrapolated forward.
        p2 = rows[first - 2]
        dh = h0 - p2["horizon_hours"]
        rate = (p2["distance_to_coast_km"] - d0) / dh if dh > 0 else 0.0
        if rate > 0:
            return acquisition_at + timedelta(hours=min(h0 + d0 / rate, h1))

    # Fall back to the midpoint of the bracketing interval.
    return acquisition_at + timedelta(hours=(h0 + h1) / 2.0)


def _to_m(proj: Projector, geom):
    """Reproject a (Multi)LineString into local metres."""
    lines = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    out = []
    for ln in lines:
        c = np.asarray(ln.coords)
        x, y = proj.to_m(c[:, 0], c[:, 1])
        out.append(LineString(np.column_stack([x, y])))
    return unary_union(out)
