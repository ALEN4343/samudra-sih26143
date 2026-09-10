"""Synthetic scenario generator — the ground truth harness (CLAUDE.md layer 11).

This module is the *simulator*: it plays the role of the real world. It plants a
culprit, releases oil from that vessel, and advects it forward to the acquisition
time. Everything downstream is the *estimator*, which must recover the answer from
observations alone.

CRITICAL: no module other than the test suite may ever read ground_truth.json.
It exists for verification only. If a pipeline module opens it, the entire
demonstration is worthless.

The advection here is deliberately an independent implementation of the physics in
CLAUDE.md section 4.5.1, not a call into attribution/drift.py. Simulator and
estimator must be able to disagree, or the round-trip test proves nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, Polygon, mapping
from shapely.ops import unary_union

# --------------------------------------------------------------------------
# Physics constants — CLAUDE.md section 4.5.1
# --------------------------------------------------------------------------

WIND_FACTOR = 0.03
WIND_DEFLECTION_DEG = 15.0
TIMESTEP_MIN = 5
DIFFUSION_K = 5.0  # m^2/s
FAY_EXPONENT = 0.75

KN_TO_MS = 0.514444

# Scene raster
SCENE_PX_M = 150.0  # metres per pixel
SEA_SIGMA0_DB = -14.0
SLICK_SIGMA0_DB = -26.0
LOOKALIKE_SIGMA0_DB = -22.0
DB_MIN, DB_MAX = -30.0, 0.0  # uint8 scaling range, recorded in the GeoTIFF tags

# Plausible MID prefixes for Arabian Sea traffic, with the flag they encode.
MID_FLAGS: list[tuple[int, str]] = [
    (419, "IN"), (422, "IR"), (463, "PK"), (461, "OM"), (470, "AE"),
    (403, "SA"), (563, "SG"), (636, "LR"), (538, "MH"), (371, "PA"),
    (477, "HK"), (249, "MT"), (240, "GR"), (525, "ID"), (412, "CN"),
]

VESSEL_TYPES: list[tuple[str, int, int]] = [
    # (type, min_length_m, max_length_m)
    ("Crude Oil Tanker", 180, 330),
    ("Product Tanker", 110, 185),
    ("Chemical Tanker", 90, 150),
    ("Bulk Carrier", 150, 290),
    ("Container Ship", 140, 300),
    ("General Cargo", 80, 140),
    ("Fishing Vessel", 20, 45),
    ("Tug", 25, 45),
]

NAME_PARTS_A = ["OCEAN", "STAR", "GULF", "PACIFIC", "ATLANTIC", "DESERT", "MONSOON",
                "CORAL", "SILVER", "GOLDEN", "NORTHERN", "SOUTHERN", "EASTERN",
                "BLUE", "IRON", "GRAND", "ROYAL", "EMERALD", "AMBER", "CRIMSON"]
NAME_PARTS_B = ["TRADER", "VOYAGER", "PIONEER", "SPIRIT", "GLORY", "HORIZON",
                "MARINER", "ENDEAVOUR", "PROSPERITY", "HARMONY", "SENTINEL",
                "EXPRESS", "CHALLENGER", "NAVIGATOR", "ZEPHYR", "MERIDIAN"]


# --------------------------------------------------------------------------
# Projection helpers
# --------------------------------------------------------------------------


def _local_crs(lat0: float, lon0: float) -> CRS:
    """Azimuthal equidistant CRS centred on the AOI. Metres, locally accurate."""
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
        f"+datum=WGS84 +units=m +no_defs"
    )


class Projector:
    """lat/lon <-> local metres. Never approximate with degrees-per-km."""

    def __init__(self, lat0: float, lon0: float):
        crs = _local_crs(lat0, lon0)
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def to_m(self, lon, lat):
        return self._fwd.transform(lon, lat)

    def to_deg(self, x, y):
        return self._inv.transform(x, y)


# --------------------------------------------------------------------------
# Environment field
# --------------------------------------------------------------------------


@dataclass
class EnvGrid:
    """Gridded wind and current. Vectors are 'flowing toward', not 'coming from'.

    Storing toward-vectors avoids the 180-degree sign error that meteorological
    wind convention invites.
    """

    lats: np.ndarray
    lons: np.ndarray
    times: np.ndarray  # epoch seconds, UTC
    wind_u: np.ndarray  # (t, lat, lon) eastward m/s
    wind_v: np.ndarray
    curr_u: np.ndarray
    curr_v: np.ndarray

    def sample(self, t_epoch: float, lat: np.ndarray, lon: np.ndarray):
        """Nearest-neighbour in time, bilinear in space. Returns 4 arrays."""
        ti = int(np.clip(np.searchsorted(self.times, t_epoch), 0, len(self.times) - 1))

        fy = np.interp(lat, self.lats, np.arange(len(self.lats)))
        fx = np.interp(lon, self.lons, np.arange(len(self.lons)))
        y0 = np.clip(np.floor(fy).astype(int), 0, len(self.lats) - 2)
        x0 = np.clip(np.floor(fx).astype(int), 0, len(self.lons) - 2)
        wy = fy - y0
        wx = fx - x0

        def bilin(cube):
            g = cube[ti]
            return (
                g[y0, x0] * (1 - wy) * (1 - wx)
                + g[y0 + 1, x0] * wy * (1 - wx)
                + g[y0, x0 + 1] * (1 - wy) * wx
                + g[y0 + 1, x0 + 1] * wy * wx
            )

        return bilin(self.wind_u), bilin(self.wind_v), bilin(self.curr_u), bilin(self.curr_v)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            lats=self.lats, lons=self.lons, times=self.times,
            wind_u=self.wind_u.astype(np.float32), wind_v=self.wind_v.astype(np.float32),
            curr_u=self.curr_u.astype(np.float32), curr_v=self.curr_v.astype(np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "EnvGrid":
        z = np.load(path)
        return cls(
            lats=z["lats"], lons=z["lons"], times=z["times"],
            wind_u=z["wind_u"], wind_v=z["wind_v"],
            curr_u=z["curr_u"], curr_v=z["curr_v"],
        )


def _smooth_noise(rng, shape, sigma) -> np.ndarray:
    f = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-9)


def build_env(
    rng: np.random.Generator,
    bounds: tuple[float, float, float, float],
    t_start: datetime,
    t_end: datetime,
    wind_range: tuple[float, float] = (4.0, 9.0),
    n_lat: int = 48,
    n_lon: int = 48,
) -> EnvGrid:
    """Smooth, spatially and temporally varying wind and current.

    Wind speed is held inside `wind_range` so the scene sits in the SAR detection
    window (CLAUDE.md section 4.1 gates 3-10 m/s).
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    lats = np.linspace(min_lat, max_lat, n_lat)
    lons = np.linspace(min_lon, max_lon, n_lon)

    n_t = int((t_end - t_start).total_seconds() // 3600) + 1
    times = np.array(
        [(t_start + timedelta(hours=i)).timestamp() for i in range(n_t)], dtype=np.float64
    )

    shape = (n_t, n_lat, n_lon)
    # Smooth in time as well as space: weather does not flicker.
    sig = (max(n_t / 12.0, 1.0), n_lat / 6.0, n_lon / 6.0)

    lo, hi = wind_range
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    # tanh keeps the speed strictly inside the range without clipping artefacts.
    wind_speed = mid + half * np.tanh(_smooth_noise(rng, shape, sig))

    prevailing = math.radians(rng.uniform(0, 360))
    wind_dir = prevailing + math.radians(35.0) * _smooth_noise(rng, shape, sig)
    wind_u = wind_speed * np.sin(wind_dir)
    wind_v = wind_speed * np.cos(wind_dir)

    curr_speed = 0.15 + 0.20 * (0.5 + 0.5 * np.tanh(_smooth_noise(rng, shape, sig)))
    curr_dir = prevailing + math.radians(60.0) + math.radians(50.0) * _smooth_noise(rng, shape, sig)
    curr_u = curr_speed * np.sin(curr_dir)
    curr_v = curr_speed * np.cos(curr_dir)

    return EnvGrid(lats, lons, times, wind_u, wind_v, curr_u, curr_v)


# --------------------------------------------------------------------------
# Advection — CLAUDE.md section 4.5.1
# --------------------------------------------------------------------------


def _rotate_cw(u, v, deg):
    """Rotate a vector clockwise (Ekman deflection, northern hemisphere)."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return u * c + v * s, -u * s + v * c


def advect(
    proj: Projector,
    env: EnvGrid,
    lat0: np.ndarray,
    lon0: np.ndarray,
    t_release: np.ndarray,
    t_target: datetime,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Advect particles from staggered release times to a common target time.

    v = current + 0.03 * wind_rotated_15deg_cw, 5-minute Euler steps, plus a
    random-walk diffusion step with K = 5 m^2/s.
    """
    lat = np.asarray(lat0, dtype=float).copy()
    lon = np.asarray(lon0, dtype=float).copy()
    t_rel = np.asarray(t_release, dtype=float)
    t_end = t_target.timestamp()

    dt = TIMESTEP_MIN * 60.0
    sigma_step = math.sqrt(2.0 * DIFFUSION_K * dt)

    t_now = float(t_rel.min())
    while t_now < t_end:
        step = min(dt, t_end - t_now)
        live = t_rel <= t_now  # a particle only moves once it has been released
        if live.any():
            wu, wv, cu, cv = env.sample(t_now, lat[live], lon[live])
            ru, rv = _rotate_cw(wu, wv, WIND_DEFLECTION_DEG)
            vu = cu + WIND_FACTOR * ru
            vv = cv + WIND_FACTOR * rv

            x, y = proj.to_m(lon[live], lat[live])
            scale = math.sqrt(step / dt)
            x = x + vu * step + rng.normal(0.0, sigma_step * scale, size=x.shape)
            y = y + vv * step + rng.normal(0.0, sigma_step * scale, size=y.shape)
            lon[live], lat[live] = proj.to_deg(x, y)
        t_now += step

    return lat, lon


def fay_radius_m(age_hours: float, base_m: float = 260.0) -> float:
    """Gravity-viscous spreading radius. Area ~ t^0.75, so radius ~ t^0.375."""
    return base_m * max(age_hours, 0.1) ** (FAY_EXPONENT / 2.0)


def alpha_shape(xy: np.ndarray, alpha_m: float) -> Polygon:
    """Alpha shape of a particle cloud, falling back to the convex hull."""
    if len(xy) < 4:
        return MultiPoint(xy).convex_hull
    try:
        tri = Delaunay(xy)
    except Exception:
        return MultiPoint(xy).convex_hull

    keep = []
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = xy[ia], xy[ib], xy[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-12))
        circum = a * b * c / (4.0 * area)
        if circum < alpha_m:
            keep.append(Polygon([pa, pb, pc]))
    if not keep:
        return MultiPoint(xy).convex_hull
    shape = unary_union(keep)
    if shape.geom_type == "MultiPolygon":
        shape = max(shape.geoms, key=lambda g: g.area)
    return shape


def particles_to_polygon(
    proj: Projector, lat: np.ndarray, lon: np.ndarray, age_hours: float
) -> Polygon:
    """Particle cloud -> slick polygon, in lat/lon."""
    x, y = proj.to_m(lon, lat)
    xy = np.column_stack([x, y])
    spread = fay_radius_m(age_hours)
    poly_m = alpha_shape(xy, alpha_m=max(spread * 8.0, 3000.0)).buffer(spread)
    poly_m = poly_m.simplify(spread * 0.10)

    ring = np.asarray(poly_m.exterior.coords)
    lo, la = proj.to_deg(ring[:, 0], ring[:, 1])
    return Polygon(np.column_stack([lo, la]))


# --------------------------------------------------------------------------
# AIS traffic
# --------------------------------------------------------------------------


@dataclass
class Vessel:
    mmsi: int
    name: str
    imo: int | None
    vtype: str
    length_m: float
    flag: str
    rows: list[dict] = field(default_factory=list)


def _make_identity(rng, used: set[int]) -> tuple[int, str, int | None, str, float, str]:
    while True:
        mid, flag = MID_FLAGS[rng.integers(len(MID_FLAGS))]
        mmsi = int(mid) * 1_000_000 + int(rng.integers(100_000, 999_999))
        if mmsi not in used:
            used.add(mmsi)
            break
    name = f"{NAME_PARTS_A[rng.integers(len(NAME_PARTS_A))]} {NAME_PARTS_B[rng.integers(len(NAME_PARTS_B))]}"
    vtype, lo, hi = VESSEL_TYPES[rng.integers(len(VESSEL_TYPES))]
    length = float(rng.integers(lo, hi))
    imo = int(rng.integers(9_000_000, 9_899_999)) if rng.random() < 0.85 else None
    return mmsi, name, imo, vtype, length, flag


def build_tracks(
    rng: np.random.Generator,
    proj: Projector,
    bounds: tuple[float, float, float, float],
    t_start: datetime,
    t_end: datetime,
    n_vessels: int,
    anchors: list[tuple[float, float, float]] | None = None,
) -> list[Vessel]:
    """Vessels transiting the AOI over the window, 1-5 minute reporting.

    Each vessel is anchored to a closest-approach point inside the AOI at a time
    drawn across the window, rather than departing from outside and transiting.
    Anchoring this way keeps AOI occupancy roughly uniform in time; departure-time
    sampling drains the AOI before the acquisition and leaves the scene empty.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    used: set[int] = set()
    vessels: list[Vessel] = []

    window_s = (t_end - t_start).total_seconds()
    if anchors is not None:
        n_vessels = len(anchors)
    attempts = 0
    while len(vessels) < n_vessels and attempts < n_vessels * 40:
        attempts += 1
        mmsi, name, imo, vtype, length, flag = _make_identity(rng, used)
        v = Vessel(mmsi, name, imo, vtype, length, flag)

        speed_kn = float(rng.uniform(8.0, 16.0))
        if vtype in ("Fishing Vessel", "Tug"):
            speed_kn = float(rng.uniform(6.0, 11.0))
        speed_ms = speed_kn * KN_TO_MS
        interval_s = float(rng.integers(1, 6)) * 60.0

        # Anchor point inside the AOI, and the time the vessel is there. The time
        # range overruns t_end so vessels are still arriving at the acquisition.
        if anchors is not None:
            px, py, t_mid = anchors[len(vessels)]
        else:
            px = float(rng.uniform(min_lon, max_lon))
            py = float(rng.uniform(min_lat, max_lat))
            t_mid = t_start.timestamp() + float(rng.uniform(-0.10, 1.15)) * window_s
        anchor_x, anchor_y = proj.to_m(px, py)

        n_steps = int(window_s // interval_s) + 1
        t = t_start.timestamp() + np.arange(n_steps) * interval_s

        course = float(rng.uniform(0, 360))
        wander = np.cumsum(rng.normal(0, 0.25, n_steps)) * 0.6
        hdg = course + wander
        hr = np.radians(hdg)
        dx = np.cumsum(np.sin(hr)) * speed_ms * interval_s
        dy = np.cumsum(np.cos(hr)) * speed_ms * interval_s

        i_mid = int(np.clip(np.searchsorted(t, t_mid), 0, n_steps - 1))
        x = anchor_x + (dx - dx[i_mid])
        y = anchor_y + (dy - dy[i_mid])

        lon, lat = proj.to_deg(x, y)
        sog = np.full(n_steps, speed_kn) + rng.normal(0, 0.25, n_steps)

        inside = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
        if inside.sum() < 20:
            used.discard(mmsi)
            continue

        for i in np.flatnonzero(inside):
            v.rows.append(
                dict(
                    mmsi=mmsi, t=float(t[i]), lat=float(lat[i]), lon=float(lon[i]),
                    sog=float(max(sog[i], 0.0)), cog=float(hdg[i] % 360.0),
                    heading=float((hdg[i] + rng.normal(0, 1.5)) % 360.0),
                )
            )
        vessels.append(v)

    if len(vessels) < n_vessels:
        raise RuntimeError(
            f"Only built {len(vessels)} of {n_vessels} tracks in {attempts} attempts."
        )
    return vessels


def _shift_remainder(rows, start_idx, dlat, dlon):
    """Translate the rest of a track so a rewritten segment stays continuous.

    A vessel that loiters for two hours is two hours behind on its route. Without
    this the track jumps back to where it would have been, which reads to the
    trust layer as a position teleport - an artefact of the generator, not a
    behaviour we meant to plant.
    """
    for r in rows[start_idx:]:
        r["lat"] += dlat
        r["lon"] += dlon


def inject_anomalies(rng, proj, vessels: list[Vessel], protect_mmsi: int) -> dict[str, list[int]]:
    """Plant the behaviours layer 5 must find.

    The culprit is deliberately excluded. If the culprit were also the spoofer,
    the priors would hand attribution the right answer for the wrong reason.

    Each injection must leave the track *physically coherent*, so that the only
    anomaly present is the one being planted. Otherwise the trust layer scores
    well by detecting generator artefacts.
    """
    pool = [v for v in vessels if v.mmsi != protect_mmsi and len(v.rows) > 120]
    rng.shuffle(pool)
    planted: dict[str, list[int]] = {"ais_gap": [], "spoofed": [], "loiter": [], "slowdown": []}

    def take(n):
        out, pool[:] = pool[:n], pool[n:]
        return out

    # AIS gaps: the vessel keeps moving while silent, so simply dropping the
    # reports is already coherent.
    for v in take(3):
        n = len(v.rows)
        i0 = int(rng.integers(n // 4, n * 3 // 4))
        span = int(rng.integers(8, 30))
        del v.rows[i0 : i0 + span]
        planted["ais_gap"].append(v.mmsi)

    # Spoofed segment: an implausible position jump and back. This one is
    # *meant* to be incoherent - that is the whole point of spoofing.
    for v in take(1):
        n = len(v.rows)
        i0 = int(rng.integers(n // 3, n * 2 // 3))
        for r in v.rows[i0 : i0 + 6]:
            r["lat"] += 0.38
            r["lon"] += 0.31
        planted["spoofed"].append(v.mmsi)

    # Loiter: circle in place at low speed, then resume from where it stopped.
    for v in take(2):
        n = len(v.rows)
        i0 = int(rng.integers(n // 4, n * 2 // 3))
        span = min(int(rng.integers(40, 80)), n - i0 - 1)
        if span < 10:
            continue
        base_lat, base_lon = v.rows[i0]["lat"], v.rows[i0]["lon"]
        orig_lat = v.rows[i0 + span - 1]["lat"]
        orig_lon = v.rows[i0 + span - 1]["lon"]
        # Advance around the circle at the speed we report. A fixed angular step
        # implies a tangential speed unrelated to SOG, which the trust layer
        # correctly flags as a disagreement - again, a generator artefact.
        radius_deg = 0.004
        radius_m = radius_deg * 111_000.0
        # Offset the centre so that at ang=0 the vessel is exactly where it
        # already was. Centring on the current position instead makes the track
        # jump one radius on entry, which reads as a SOG disagreement.
        centre_lat, centre_lon = base_lat, base_lon - radius_deg
        ang = 0.0
        for k, r in enumerate(v.rows[i0 : i0 + span]):
            sog_kn = float(abs(rng.normal(0.4, 0.1)))
            if k > 0:
                dt = r["t"] - v.rows[i0 + k - 1]["t"]
                ang += (sog_kn * KN_TO_MS * dt) / radius_m
            r["lat"] = centre_lat + radius_deg * math.sin(ang)
            r["lon"] = centre_lon + radius_deg * math.cos(ang)
            r["sog"] = sog_kn
            r["cog"] = float((math.degrees(ang) + 90) % 360)
        exit_lat = v.rows[i0 + span - 1]["lat"]
        exit_lon = v.rows[i0 + span - 1]["lon"]
        _shift_remainder(v.rows, i0 + span, exit_lat - orig_lat, exit_lon - orig_lon)
        planted["loiter"].append(v.mmsi)

    # Sharp slowdown: actually slow the vessel down. Changing only the reported
    # SOG would be speed spoofing, and would be flagged as a data inconsistency
    # rather than as the manoeuvre we intend to plant.
    for v in take(1):
        n = len(v.rows)
        i0 = int(rng.integers(n // 3, n * 2 // 3))
        span = min(50, n - i0 - 1)
        if span < 10:
            continue
        slow_kn = float(max(rng.normal(3.0, 0.4), 1.5))
        orig_lat = v.rows[i0 + span - 1]["lat"]
        orig_lon = v.rows[i0 + span - 1]["lon"]

        x, y = proj.to_m(
            np.array([r["lon"] for r in v.rows[i0 : i0 + span]]),
            np.array([r["lat"] for r in v.rows[i0 : i0 + span]]),
        )
        px, py = float(x[0]), float(y[0])
        for k, r in enumerate(v.rows[i0 : i0 + span]):
            if k > 0:
                dt = r["t"] - v.rows[i0 + k - 1]["t"]
                hr = math.radians(r["cog"])
                step = slow_kn * KN_TO_MS * dt
                px += math.sin(hr) * step
                py += math.cos(hr) * step
            lon_k, lat_k = proj.to_deg(px, py)
            r["lat"], r["lon"] = float(lat_k), float(lon_k)
            r["sog"] = float(max(rng.normal(slow_kn, 0.15), 0.3))
        _shift_remainder(
            v.rows, i0 + span,
            v.rows[i0 + span - 1]["lat"] - orig_lat,
            v.rows[i0 + span - 1]["lon"] - orig_lon,
        )
        planted["slowdown"].append(v.mmsi)

    return planted


# --------------------------------------------------------------------------
# Synthetic SAR scene
# --------------------------------------------------------------------------


def render_scene(
    rng: np.random.Generator,
    proj: Projector,
    path: Path,
    slick: Polygon,
    vessel_positions: list[tuple[float, float]],
    rng_seed: int,
    aoi: tuple[float, float, float, float] | None = None,
) -> tuple[dict, list[dict]]:
    """A SAR-like GeoTIFF: speckled sea, a dark slick, look-alikes, bright ships.

    Stored as uint8 with sigma0 in dB linearly mapped from [-30, 0], so it opens
    in any image viewer while remaining quantitatively recoverable. The mapping is
    written into the GeoTIFF tags.
    """
    # The scene covers the whole AOI. A real Sentinel-1 IW frame is ~250x180 km and
    # would cover only part of a 5-degree AOI; making them coincide is a deliberate
    # simplification so that "in scene" and "in AOI" mean the same thing, which
    # keeps the layer 6 funnel unambiguous.
    if aoi is not None:
        b = tuple(aoi)
    else:
        cx, cy = slick.centroid.x, slick.centroid.y
        b = (cx - 0.55, cy - 0.42, cx + 0.55, cy + 0.42)

    x0, y0 = proj.to_m(b[0], b[1])
    x1, y1 = proj.to_m(b[2], b[3])
    w = int((x1 - x0) / SCENE_PX_M)
    h = int((y1 - y0) / SCENE_PX_M)

    # Sea backscatter in linear power, with multiplicative speckle.
    sea_lin = 10 ** (SEA_SIGMA0_DB / 10.0)
    img = np.full((h, w), sea_lin, dtype=np.float32)

    # Gentle large-scale modulation (wind streaks). Generated on a coarse grid and
    # zoomed: a gaussian_filter with sigma of a few hundred pixels on a scene this
    # size takes minutes, and the result is identical to smooth-then-upsample.
    coarse = _smooth_noise(rng, (48, 48), (48 / 14.0, 48 / 14.0))
    streaks = zoom(coarse, (h / 48.0, w / 48.0), order=3)[:h, :w]
    img *= (1.0 + 0.10 * streaks).astype(np.float32)

    lon_grid = np.linspace(b[0], b[2], w)
    lat_grid = np.linspace(b[3], b[1], h)  # north-up raster

    def burn(poly: Polygon, target_db: float, feather_px: int = 3):
        from shapely.prepared import prep
        from rasterio.features import rasterize

        tr = from_bounds(b[0], b[1], b[2], b[3], w, h)
        mask = rasterize([(poly, 1)], out_shape=(h, w), transform=tr, dtype=np.uint8)
        soft = gaussian_filter(mask.astype(np.float32), sigma=feather_px)
        target_lin = 10 ** (target_db / 10.0)
        np.copyto(img, img * (1 - soft) + target_lin * soft)

    burn(slick, SLICK_SIGMA0_DB)

    # Two dark look-alikes that are NOT oil — low-wind patches. These are the
    # false positives detection has to reject.
    lookalikes = []
    for _ in range(2):
        for _try in range(40):
            lo = float(rng.uniform(b[0] + 0.20, b[2] - 0.20))
            la = float(rng.uniform(b[1] + 0.20, b[3] - 0.20))
            if not slick.buffer(0.10).contains(Point(lo, la)):
                break
        rx, ry = float(rng.uniform(0.03, 0.07)), float(rng.uniform(0.02, 0.05))
        ang = float(rng.uniform(0, math.pi))
        th = np.linspace(0, 2 * math.pi, 40)
        ex = rx * np.cos(th) * math.cos(ang) - ry * np.sin(th) * math.sin(ang)
        ey = rx * np.cos(th) * math.sin(ang) + ry * np.sin(th) * math.cos(ang)
        p = Polygon(np.column_stack([lo + ex, la + ey]))
        burn(p, LOOKALIKE_SIGMA0_DB, feather_px=6)
        lookalikes.append(mapping(p))

    # Multiplicative speckle: single-look intensity is exponentially distributed.
    img *= rng.gamma(shape=3.0, scale=1 / 3.0, size=(h, w)).astype(np.float32)

    # Bright point targets at each vessel's true position.
    planted_px = []
    for vlon, vlat in vessel_positions:
        if not (b[0] < vlon < b[2] and b[1] < vlat < b[3]):
            continue
        px = int((vlon - b[0]) / (b[2] - b[0]) * w)
        py = int((b[3] - vlat) / (b[3] - b[1]) * h)
        if 2 <= px < w - 2 and 2 <= py < h - 2:
            img[py - 1 : py + 2, px - 1 : px + 2] += sea_lin * float(rng.uniform(25, 70))
            planted_px.append({"lon": vlon, "lat": vlat, "px": px, "py": py})

    db = 10.0 * np.log10(np.maximum(img, 1e-6))
    u8 = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN) * 255.0, 0, 255).astype(np.uint8)

    transform = from_bounds(b[0], b[1], b[2], b[3], w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8",
        crs="EPSG:4326", transform=transform, compress="deflate",
    ) as dst:
        dst.write(u8, 1)
        dst.update_tags(
            SAMUDRA_SYNTHETIC="true",
            DB_MIN=str(DB_MIN), DB_MAX=str(DB_MAX),
            DB_FORMULA="sigma0_db = DN / 255 * (DB_MAX - DB_MIN) + DB_MIN",
            SEED=str(rng_seed),
        )

    return {"bounds": list(b), "width": w, "height": h,
            "lookalikes": lookalikes, "planted_vessel_px": planted_px}, lookalikes


# --------------------------------------------------------------------------
# Scenario assembly
# --------------------------------------------------------------------------


def generate(
    scenario_id: str = "demo-001",
    seed: int = 20260101,
    bounds: tuple[float, float, float, float] = (68.0, 15.0, 73.0, 20.0),
    acquisition_at: datetime | None = None,
    n_vessels: int = 40,
    n_decoys: int = 7,
    wind_range: tuple[float, float] = (4.0, 9.0),
    out_root: Path = Path("artifacts"),
) -> dict:
    rng = np.random.default_rng(seed)
    if acquisition_at is None:
        acquisition_at = datetime(2026, 1, 14, 6, 30, tzinfo=timezone.utc)
    acquisition_at = acquisition_at.astimezone(timezone.utc)

    t_start = acquisition_at - timedelta(hours=72)
    # Env runs past acquisition so layer 8 can forecast +72h without extrapolating.
    t_env_end = acquisition_at + timedelta(hours=78)

    min_lon, min_lat, max_lon, max_lat = bounds
    proj = Projector((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)

    out = out_root / scenario_id
    out.mkdir(parents=True, exist_ok=True)

    env = build_env(rng, bounds, t_start - timedelta(hours=2), t_env_end, wind_range)
    env.save(out / "env.npz")

    vessels = build_tracks(rng, proj, bounds, t_start, acquisition_at, n_vessels)
    if len(vessels) < 10:
        raise RuntimeError(
            f"Only {len(vessels)} vessels survived AOI clipping — the transit geometry "
            f"is wrong, not merely unlucky. Refusing to emit a degenerate scenario."
        )

    # --- pick the culprit -------------------------------------------------
    # It must be inside the AOI long enough before acquisition for the slick to
    # have drifted a measurable distance.
    age_hours = float(rng.uniform(6.0, 14.0))
    release_at = acquisition_at - timedelta(hours=age_hours)
    release_duration_min = 45.0

    # The release must also sit in the AOI interior, so the drifted slick and its
    # scene footprint stay inside the area of interest rather than running off a
    # corner.
    ilon = (max_lon - min_lon) * 0.20
    ilat = (max_lat - min_lat) * 0.20
    interior = (min_lon + ilon, min_lat + ilat, max_lon - ilon, max_lat - ilat)

    eligible, interior_eligible = [], []
    for v in vessels:
        ts = np.array([r["t"] for r in v.rows])
        if not (
            ts.min() <= release_at.timestamp()
            and ts.max() >= release_at.timestamp() + release_duration_min * 60
        ):
            continue
        eligible.append(v)
        at = min(v.rows, key=lambda r: abs(r["t"] - release_at.timestamp()))
        if (
            interior[0] <= at["lon"] <= interior[2]
            and interior[1] <= at["lat"] <= interior[3]
        ):
            interior_eligible.append(v)

    if not eligible:
        raise RuntimeError("No vessel is in the AOI across the release window.")

    # Prefer a vessel that could plausibly discharge this volume. A tug producing a
    # 50 km2 slick invites the obvious objection during the demo.
    pool_ = interior_eligible or eligible
    plausible = [v for v in pool_ if "Tanker" in v.vtype or v.vtype in ("Bulk Carrier", "Container Ship", "General Cargo")]
    pool_ = plausible or pool_
    culprit = pool_[int(rng.integers(len(pool_)))]

    # Release oil continuously while the vessel steams — a line source. This is
    # what makes a real discharge elongated and aligned with the ship's course,
    # and it is what gives the orientation term in section 4.5.2 any meaning.
    ct = np.array([r["t"] for r in culprit.rows])
    seg = np.flatnonzero(
        (ct >= release_at.timestamp())
        & (ct <= release_at.timestamp() + release_duration_min * 60)
    )
    seg_rows = [culprit.rows[i] for i in seg]
    true_lat = seg_rows[0]["lat"]
    true_lon = seg_rows[0]["lon"]
    course_at_release = float(seg_rows[0]["cog"])

    n_particles = 900
    pick = rng.integers(0, len(seg_rows), n_particles)
    p_lat = np.array([seg_rows[i]["lat"] for i in pick])
    p_lon = np.array([seg_rows[i]["lon"] for i in pick])
    p_t = np.array([seg_rows[i]["t"] for i in pick])
    # Jitter across the ship's beam so the source is a ribbon, not a hairline.
    p_lat += rng.normal(0, 0.0015, n_particles)
    p_lon += rng.normal(0, 0.0015, n_particles)

    f_lat, f_lon = advect(proj, env, p_lat, p_lon, p_t, acquisition_at, rng)
    slick = particles_to_polygon(proj, f_lat, f_lon, age_hours)

    sx, sy = proj.to_m(*slick.exterior.coords.xy)
    area_km2 = Polygon(np.column_stack([sx, sy])).area / 1e6

    # --- decoys ------------------------------------------------------------
    # Vessels that also transited near the origin around the release window. In a
    # real shipping lane several ships pass through the same water, and that is
    # precisely what makes attribution hard. Without them the origin envelope
    # holds exactly one candidate and ranking never has to discriminate — the
    # pipeline would look correct while proving nothing.
    d_km = 111.0
    decoy_anchors = []
    for _ in range(n_decoys):
        bearing = rng.uniform(0, 2 * math.pi)
        dist_km = rng.uniform(6.0, 38.0)
        dlat = dist_km * math.cos(bearing) / d_km
        dlon = dist_km * math.sin(bearing) / (d_km * math.cos(math.radians(true_lat)))
        decoy_anchors.append(
            (
                float(np.clip(true_lon + dlon, min_lon + 0.05, max_lon - 0.05)),
                float(np.clip(true_lat + dlat, min_lat + 0.05, max_lat - 0.05)),
                release_at.timestamp() + float(rng.uniform(-3.5, 3.5)) * 3600.0,
            )
        )
    decoys = build_tracks(
        rng, proj, bounds, t_start, acquisition_at, len(decoy_anchors), anchors=decoy_anchors
    )
    vessels.extend(decoys)

    # --- planted behaviours ----------------------------------------------
    planted = inject_anomalies(rng, proj, vessels, protect_mmsi=culprit.mmsi)

    # --- write AIS ---------------------------------------------------------
    rows = []
    for v in vessels:
        for r in v.rows:
            rows.append(
                dict(
                    mmsi=v.mmsi,
                    t=datetime.fromtimestamp(r["t"], tz=timezone.utc),
                    lat=r["lat"], lon=r["lon"], sog=r["sog"],
                    cog=r["cog"], heading=r["heading"],
                    vessel_name=v.name, imo=v.imo, vessel_type=v.vtype,
                    length_m=v.length_m, flag=v.flag,
                )
            )
    ais = pd.DataFrame(rows).sort_values(["mmsi", "t"]).reset_index(drop=True)
    ais.to_parquet(out / "ais.parquet", index=False)

    # --- vessel positions at acquisition, for the scene and CFAR check ----
    # Nearest fix in time to acquisition, not the last fix: a track clipped at the
    # AOI edge still has a valid position if it was reporting near acquisition.
    positions = []
    acq_ts = acquisition_at.timestamp()
    for v in vessels:
        if not v.rows:
            continue
        near = min(v.rows, key=lambda r: abs(r["t"] - acq_ts))
        if abs(near["t"] - acq_ts) <= 1800:
            positions.append((near["lon"], near["lat"]))

    scene_meta, _ = render_scene(
        rng, proj, out / "scene.tif", slick, positions, seed, aoi=bounds
    )

    # --- observed slick ----------------------------------------------------
    (out / "observed_slick.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(slick),
                        "properties": {
                            "area_km2": round(area_km2, 3),
                            "acquisition_at": acquisition_at.isoformat(),
                            "synthetic": True,
                        },
                    }
                ],
            },
            indent=2,
        )
    )

    # --- ground truth (TEST SUITE ONLY) -----------------------------------
    gt = {
        "_warning": "VERIFICATION ONLY. No pipeline module may read this file.",
        "scenario_id": scenario_id,
        "seed": seed,
        "culprit_mmsi": culprit.mmsi,
        "culprit_name": culprit.name,
        "culprit_type": culprit.vtype,
        "true_release_at": release_at.isoformat(),
        "true_release_lat": true_lat,
        "true_release_lon": true_lon,
        "release_duration_min": release_duration_min,
        "course_at_release_deg": course_at_release,
        "acquisition_at": acquisition_at.isoformat(),
        "aoi_bounds": list(bounds),
        "slick_area_km2": round(area_km2, 3),
        "slick_age_hours": round(age_hours, 3),
        "n_vessels": len(vessels),
        "n_decoys": len(decoys),
        "decoy_mmsis": [d.mmsi for d in decoys],
        "planted_anomalies": planted,
        "scene": scene_meta,
    }
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))

    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic SAMUDRA scenario.")
    ap.add_argument("--scenario", default="demo-001")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--vessels", type=int, default=40)
    ap.add_argument("--decoys", type=int, default=7)
    ap.add_argument("--wind-min", type=float, default=4.0)
    ap.add_argument("--wind-max", type=float, default=9.0)
    ap.add_argument("--bounds", type=float, nargs=4, default=[68.0, 15.0, 73.0, 20.0],
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--acquisition", default="2026-01-14T06:30:00+00:00")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()

    gt = generate(
        scenario_id=a.scenario,
        seed=a.seed,
        bounds=tuple(a.bounds),
        acquisition_at=datetime.fromisoformat(a.acquisition),
        n_vessels=a.vessels,
        n_decoys=a.decoys,
        wind_range=(a.wind_min, a.wind_max),
        out_root=Path(a.out),
    )

    print(f"scenario            : {gt['scenario_id']}  (seed {gt['seed']})")
    print(f"tracks generated    : {gt['n_vessels']}  (incl. {gt['n_decoys']} decoys near origin)")
    print(f"culprit MMSI        : {gt['culprit_mmsi']}  {gt['culprit_name']} ({gt['culprit_type']})")
    print(f"true release at     : {gt['true_release_at']}")
    print(f"true release pos    : {gt['true_release_lat']:.5f} N, {gt['true_release_lon']:.5f} E")
    print(f"course at release   : {gt['course_at_release_deg']:.1f} deg")
    print(f"acquisition at      : {gt['acquisition_at']}")
    print(f"slick age           : {gt['slick_age_hours']:.2f} h")
    print(f"slick area          : {gt['slick_area_km2']:.2f} km2")
    print(f"scene               : {gt['scene']['width']} x {gt['scene']['height']} px")
    print(f"planted anomalies   : {gt['planted_anomalies']}")


if __name__ == "__main__":
    main()
