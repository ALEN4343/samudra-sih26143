"""Land mask for drift — so oil beaches instead of sailing across Gujarat.

WHY THIS EXISTS. `attribution/drift.py` integrates particles through a velocity
field and nothing in that field knows where the coast is. Over the open-ocean
demo AOIs that never mattered, because no scenario reached shore inside 72 h.
The moment a scenario starts near a coast — the Gulf of Kutch does — particles
advect straight over land and the forecast polygon covers dry ground. That is
not a cosmetic problem: the +24/48/72 h products and the shoreline impact ETA
are both computed from those polygons.

HOW. Rasterise Natural Earth land polygons once per AOI, then beaching is an
array lookup per particle per step rather than a polygon containment test. At
the default 150 m cell a 200 x 200 km AOI is a 1333 x 1333 bool array — about
1.8 MB, built in well under a second and reused for every step of every
particle.

WHAT BEACHING MEANS HERE. A particle that enters a land cell stops, permanently,
at the last water position it held. It is not reflected and not deleted:
  - reflecting would send oil back out to sea, understating shoreline impact
  - deleting would shrink the slick, which reads as the oil disappearing
Stopping is what oil actually does at a shoreline, and it makes the beached
fraction directly reportable — which is the number a responder wants.

HONEST LIMIT. Natural Earth is a coastline product, not a tidal model. It has
no intertidal zone, no mudflats and no sub-pixel creeks, so in a place like the
Gulf of Kutch the true beaching line moves with the tide and this one does not.
It is the right shape at the wrong instant, and `source` says which file it came
from so that can be checked.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Most detailed first, matching impact/coastline.py. "10m" is 1:10,000,000 —
#: finer than "50m", not coarser.
LAND_PATHS = (
    Path("data/raw/ne_10m_land.geojson"),
    Path("data/raw/ne_50m_land.geojson"),
    Path("data/raw/land.geojson"),
)

DEFAULT_CELL_M = 150.0


@lru_cache(maxsize=4)
def _load_land(path_str: str, mtime: float, size: int):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    gj = json.loads(Path(path_str).read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    return unary_union([shape(f["geometry"]) for f in feats])


def land_geometry() -> tuple[object | None, str]:
    """Union of land polygons, plus the source actually used."""
    for p in LAND_PATHS:
        if p.exists():
            st = p.stat()
            return _load_land(str(p), st.st_mtime, st.st_size), f"Natural Earth ({p})"
    return None, ("no land polygons on disk — run scripts/fetch_coastline.py; "
                  "drift will NOT beach")


class LandMask:
    """Rasterised land over one AOI. `is_land(lat, lon)` is vectorised."""

    def __init__(self, bounds, cell_m: float = DEFAULT_CELL_M,
                 pad_deg: float = 0.25):
        self.available = False
        self.source = ""
        self.bounds = tuple(float(v) for v in bounds)

        geom, source = land_geometry()
        self.source = source
        self.geometry = None
        if geom is None:
            self.grid = None
            return

        from shapely.geometry import box

        min_lon, min_lat, max_lon, max_lat = self.bounds
        win = box(min_lon - pad_deg, min_lat - pad_deg,
                  max_lon + pad_deg, max_lat + pad_deg)
        try:
            clipped = geom.intersection(win)
        except Exception:  # noqa: BLE001 - topology error on a pathological input
            clipped = None

        self.min_lon, self.min_lat = min_lon - pad_deg, min_lat - pad_deg
        self.max_lon, self.max_lat = max_lon + pad_deg, max_lat + pad_deg

        mid_lat = (self.min_lat + self.max_lat) / 2.0
        self.dlat = cell_m / 110_574.0
        self.dlon = cell_m / max(111_320.0 * np.cos(np.radians(mid_lat)), 1.0)
        self.nlat = max(2, int(round((self.max_lat - self.min_lat) / self.dlat)))
        self.nlon = max(2, int(round((self.max_lon - self.min_lon) / self.dlon)))

        # Kept as well as the raster. The raster answers "is this particle on
        # land" in O(1); the vector answers "trim this polygon to water", which
        # a raster cannot do without stair-stepping the coastline.
        self.geometry = clipped

        if clipped is None or clipped.is_empty:
            # No land anywhere near this AOI. That is a legitimate answer for an
            # open-ocean scenario, and an all-water mask is cheaper than None to
            # consume — the caller does not need a special case.
            self.grid = np.zeros((self.nlat, self.nlon), dtype=bool)
            self.available = True
            self.land_fraction = 0.0
            return

        from rasterio.features import rasterize
        from rasterio.transform import from_origin

        transform = from_origin(self.min_lon, self.max_lat, self.dlon, self.dlat)
        arr = rasterize([(clipped, 1)], out_shape=(self.nlat, self.nlon),
                        transform=transform, dtype="uint8")
        # Row 0 is the NORTH edge in a raster; flip so index maths below can be
        # written south-up, which is how lat/lon arithmetic reads.
        self.grid = np.flipud(arr).astype(bool)
        self.available = True
        self.land_fraction = float(self.grid.mean())

    def is_land(self, lat, lon) -> np.ndarray:
        """True where the position falls in a land cell. Outside the grid: False.

        Outside is treated as water deliberately. The grid is padded well beyond
        the AOI, so a particle outside it has left the modelled area entirely;
        calling that land would beach it on nothing.
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        if self.grid is None:
            return np.zeros(lat.shape, dtype=bool)

        r = ((lat - self.min_lat) / self.dlat).astype(int)
        c = ((lon - self.min_lon) / self.dlon).astype(int)
        inside = (r >= 0) & (r < self.nlat) & (c >= 0) & (c < self.nlon)
        out = np.zeros(lat.shape, dtype=bool)
        if inside.any():
            out[inside] = self.grid[r[inside], c[inside]]
        return out

    def clip_to_water(self, poly):
        """Trim a slick polygon so it does not cover land.

        Needed on top of particle beaching, not instead of it. Beaching fixes
        where the oil IS; this fixes where it is DRAWN. `particles_to_polygon`
        wraps a hull and a Fay spreading buffer around the particles, and that
        buffer spills inland even when every particle has stopped at the shore —
        measured at 45% of the polygon area on land in the Gulf of Kutch case.

        Returns the input unchanged if the clip fails or would empty it; a
        polygon that is entirely on land is a signal worth seeing, not something
        to silently discard.
        """
        if poly is None or self.geometry is None or self.geometry.is_empty:
            return poly
        try:
            out = poly.difference(self.geometry)
        except Exception:  # noqa: BLE001 - topology error on a self-touching hull
            return poly
        if out.is_empty:
            return poly
        # difference() can split a slick around a headland or an island. Keep
        # the whole MultiPolygon: the oil really is in two places.
        return out

    def describe(self) -> dict:
        return {
            "available": self.available,
            "source": self.source,
            "cell_m": round(float(self.dlat * 110_574.0), 1),
            "shape": None if self.grid is None else list(self.grid.shape),
            "land_fraction": (None if self.grid is None
                              else round(float(self.grid.mean()), 4)),
            "note": ("Beached particles stop at their last water position — not "
                     "reflected (which would understate shoreline impact) and "
                     "not deleted (which would look like the oil vanishing)."),
        }


def for_bounds(bounds, cell_m: float = DEFAULT_CELL_M) -> LandMask:
    """Convenience constructor; kept so callers do not import the class."""
    return LandMask(bounds, cell_m=cell_m)
