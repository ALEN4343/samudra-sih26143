"""Coastline intersection and ETA — CLAUDE.md layer 8.

Prefers a real Natural Earth coastline when one is present on disk. This
environment sits behind TLS interception, so the download fails certificate
verification; rather than pretend otherwise, a simplified coastline for the demo
region is bundled and the source actually used is reported in every result.

Drop `ne_50m_coastline.geojson` into data/raw/ and it is picked up automatically
with no code change.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import unary_union

from samudra.geo import Projector

COASTLINE_PATHS = (
    Path("data/raw/ne_50m_coastline.geojson"),
    Path("data/raw/ne_10m_coastline.geojson"),
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


def load_coastline() -> dict:
    """Return {'geometry': MultiLineString, 'source': str}."""
    for p in COASTLINE_PATHS:
        if p.exists():
            gj = json.loads(p.read_text())
            feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
            geoms = [shape(f["geometry"] if "geometry" in f else f) for f in feats]
            return {"geometry": unary_union(geoms), "source": f"Natural Earth ({p})"}

    return {
        "geometry": MultiLineString([LineString(_INDIA_WEST_COAST)]),
        "source": "bundled simplified Indian west coast (Natural Earth unavailable)",
    }


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

    coast_m = _to_m(proj, geom)

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
