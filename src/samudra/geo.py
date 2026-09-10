"""Shared geometry utilities: projection, particle clouds, polygon metrics.

Deliberately neutral infrastructure. The *physics* in synth/generate.py and
attribution/drift.py stays independently implemented — that independence is what
the round-trip test in tests/test_drift.py actually verifies. Sharing a
projection helper does not weaken it; sharing an advection routine would.

All metric work happens in a local azimuthal equidistant projection. Never
approximate with a fixed degrees-per-km constant.
"""

from __future__ import annotations

import math

import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Polygon, shape
from shapely.ops import unary_union


class Projector:
    """lat/lon <-> local metres, centred on a point of interest."""

    def __init__(self, lat0: float, lon0: float):
        crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
            f"+datum=WGS84 +units=m +no_defs"
        )
        self.lat0, self.lon0 = lat0, lon0
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def to_m(self, lon, lat):
        return self._fwd.transform(lon, lat)

    def to_deg(self, x, y):
        return self._inv.transform(x, y)

    def polygon_to_m(self, poly: Polygon) -> Polygon:
        ring = np.asarray(poly.exterior.coords)
        x, y = self.to_m(ring[:, 0], ring[:, 1])
        return Polygon(np.column_stack([x, y]))

    def polygon_to_deg(self, poly: Polygon) -> Polygon:
        ring = np.asarray(poly.exterior.coords)
        lon, lat = self.to_deg(ring[:, 0], ring[:, 1])
        return Polygon(np.column_stack([lon, lat]))

    @classmethod
    def for_bounds(cls, bounds) -> "Projector":
        min_lon, min_lat, max_lon, max_lat = bounds
        return cls((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)


def alpha_shape(xy: np.ndarray, alpha_m: float) -> Polygon:
    """Concave hull of a point cloud, falling back to the convex hull."""
    if len(xy) < 4:
        return MultiPoint(xy).convex_hull
    try:
        tri = Delaunay(xy)
    except Exception:
        return MultiPoint(xy).convex_hull

    keep = []
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = xy[ia], xy[ib], xy[ic]
        a = float(np.linalg.norm(pa - pb))
        b = float(np.linalg.norm(pb - pc))
        c = float(np.linalg.norm(pc - pa))
        s = (a + b + c) / 2.0
        area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-12))
        if a * b * c / (4.0 * area) < alpha_m:
            keep.append(Polygon([pa, pb, pc]))

    if not keep:
        return MultiPoint(xy).convex_hull
    shp = unary_union(keep)
    if shp.geom_type == "MultiPolygon":
        shp = max(shp.geoms, key=lambda g: g.area)
    return shp


def particles_to_polygon(
    proj: Projector,
    lat: np.ndarray,
    lon: np.ndarray,
    spread_m: float,
    simplify_frac: float = 0.10,
) -> Polygon:
    """Particle cloud -> a single lat/lon polygon, buffered by the spread radius."""
    x, y = proj.to_m(np.asarray(lon), np.asarray(lat))
    xy = np.column_stack([x, y])
    poly_m = alpha_shape(xy, alpha_m=max(spread_m * 8.0, 3000.0)).buffer(spread_m)
    if simplify_frac:
        poly_m = poly_m.simplify(spread_m * simplify_frac)
    if poly_m.geom_type == "MultiPolygon":
        poly_m = max(poly_m.geoms, key=lambda g: g.area)
    return proj.polygon_to_deg(poly_m)


def major_axis_deg(poly_m: Polygon) -> float:
    """Orientation of the long axis, degrees clockwise from north, in [0, 180).

    Uses the minimum rotated rectangle rather than PCA: it is stable for the
    ribbon-shaped polygons a drifting slick produces.
    """
    rect = poly_m.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return 0.0
    pts = np.asarray(rect.exterior.coords)[:4]
    edges = [(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    longest = max(edges, key=lambda e: float(np.hypot(*e)))
    dx, dy = float(longest[0]), float(longest[1])
    return math.degrees(math.atan2(dx, dy)) % 180.0


def angular_delta_deg(a: float, b: float) -> float:
    """Smallest angle between two undirected orientations, in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def polygon_metrics(proj: Projector, poly_deg: Polygon) -> dict:
    """Geometric features required by SlickPolygon in contracts.py."""
    pm = proj.polygon_to_m(poly_deg)
    area_m2 = pm.area
    perim_m = pm.length
    rect = pm.minimum_rotated_rectangle
    pts = np.asarray(rect.exterior.coords)[:4]
    side = sorted(float(np.hypot(*(pts[(i + 1) % 4] - pts[i]))) for i in range(4))
    minor, major = side[0], side[-1]
    ecc = math.sqrt(max(1.0 - (minor / major) ** 2, 0.0)) if major > 0 else 0.0
    complexity = perim_m / (2.0 * math.sqrt(math.pi * area_m2)) if area_m2 > 0 else 0.0
    c = poly_deg.centroid
    return {
        "area_km2": area_m2 / 1e6,
        "perimeter_km": perim_m / 1000.0,
        "centroid_lat": c.y,
        "centroid_lon": c.x,
        "major_axis_deg": major_axis_deg(pm),
        "major_axis_m": major,
        "minor_axis_m": minor,
        "eccentricity": ecc,
        "shape_complexity": complexity,
    }


def iou(proj: Projector, a_deg: Polygon, b_deg: Polygon) -> float:
    """Intersection over union, computed in metres."""
    a = proj.polygon_to_m(a_deg).buffer(0)
    b = proj.polygon_to_m(b_deg).buffer(0)
    union = a.union(b).area
    return (a.intersection(b).area / union) if union > 0 else 0.0


def centroid_offset_km(proj: Projector, a_deg: Polygon, b_deg: Polygon) -> float:
    ca, cb = a_deg.centroid, b_deg.centroid
    ax, ay = proj.to_m(ca.x, ca.y)
    bx, by = proj.to_m(cb.x, cb.y)
    return float(math.hypot(ax - bx, ay - by)) / 1000.0


def load_polygon(geojson: dict) -> Polygon:
    """First polygon out of a GeoJSON dict (Feature, FeatureCollection or geometry)."""
    if geojson.get("type") == "FeatureCollection":
        geom = geojson["features"][0]["geometry"]
    elif geojson.get("type") == "Feature":
        geom = geojson["geometry"]
    else:
        geom = geojson
    g = shape(geom)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    return g
