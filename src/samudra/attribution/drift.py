"""Drift model — CLAUDE.md section 4.5.1 (layer 7).

The *estimator* half of the system. Independently implements the same physics the
synthetic generator uses to build the world, so that agreement between them is
evidence the model is right rather than evidence they share a bug.

  v = current + 0.03 * wind_rotated_15deg_clockwise
  5-minute Euler steps
  random-walk diffusion, K = 5 m^2/s per particle per step

Forward: where does oil released here end up?
Reverse: where could oil observed here have come from?
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import Polygon, mapping

from samudra.geo import Projector, particles_to_polygon

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _load_cfg(path: Path | str = "config/weights.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


_CFG = None


def cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = _load_cfg()
    return _CFG


# --------------------------------------------------------------------------
# Environment field
# --------------------------------------------------------------------------


@dataclass
class EnvField:
    """Gridded wind and current read from env.npz.

    Vectors are 'flowing toward'. env.npz is a pipeline input, not ground truth:
    in production it would come from ERA5 / HYCOM instead.
    """

    lats: np.ndarray
    lons: np.ndarray
    times: np.ndarray  # epoch seconds UTC, ascending
    wind_u: np.ndarray  # (t, lat, lon)
    wind_v: np.ndarray
    curr_u: np.ndarray
    curr_v: np.ndarray

    @classmethod
    def load(cls, path: Path | str) -> "EnvField":
        z = np.load(path)
        missing = {"lats", "lons", "times", "wind_u", "wind_v", "curr_u", "curr_v"} - set(z.files)
        if missing:
            raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
        return cls(
            lats=z["lats"], lons=z["lons"], times=z["times"],
            wind_u=z["wind_u"], wind_v=z["wind_v"],
            curr_u=z["curr_u"], curr_v=z["curr_v"],
        )

    def sample(self, t_epoch: float, lat: np.ndarray, lon: np.ndarray):
        """Nearest in time, bilinear in space."""
        ti = int(np.clip(np.searchsorted(self.times, t_epoch), 0, len(self.times) - 1))
        fy = np.interp(lat, self.lats, np.arange(len(self.lats)))
        fx = np.interp(lon, self.lons, np.arange(len(self.lons)))
        y0 = np.clip(np.floor(fy).astype(int), 0, len(self.lats) - 2)
        x0 = np.clip(np.floor(fx).astype(int), 0, len(self.lons) - 2)
        wy, wx = fy - y0, fx - x0

        def bilin(cube):
            g = cube[ti]
            return (
                g[y0, x0] * (1 - wy) * (1 - wx)
                + g[y0 + 1, x0] * wy * (1 - wx)
                + g[y0, x0 + 1] * (1 - wy) * wx
                + g[y0 + 1, x0 + 1] * wy * wx
            )

        return bilin(self.wind_u), bilin(self.wind_v), bilin(self.curr_u), bilin(self.curr_v)

    def summary(self, t_epoch: float) -> dict:
        """Mean conditions over the AOI at a time, for the environmental panel."""
        ti = int(np.clip(np.searchsorted(self.times, t_epoch), 0, len(self.times) - 1))
        wu, wv = float(self.wind_u[ti].mean()), float(self.wind_v[ti].mean())
        cu, cv = float(self.curr_u[ti].mean()), float(self.curr_v[ti].mean())
        ws = float(np.hypot(self.wind_u[ti], self.wind_v[ti]).mean())
        cs = float(np.hypot(self.curr_u[ti], self.curr_v[ti]).mean())
        d = cfg()["detection"]
        return {
            "mean_wind_speed_ms": ws,
            "mean_wind_dir_deg": math.degrees(math.atan2(wu, wv)) % 360.0,
            "mean_current_speed_ms": cs,
            "mean_current_dir_deg": math.degrees(math.atan2(cu, cv)) % 360.0,
            "wind_gate_pass": bool(d["wind_gate_min_ms"] <= ws <= d["wind_gate_max_ms"]),
        }


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------


def _rotate_cw(u, v, deg: float):
    """Rotate a vector clockwise. Ekman deflection, northern hemisphere."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return u * c + v * s, -u * s + v * c


def drift_velocity(env: EnvField, t_epoch: float, lat, lon):
    """v = current + 0.03 * wind rotated 15 deg clockwise. Returns (u, v) in m/s."""
    d = cfg()["drift"]
    wu, wv, cu, cv = env.sample(t_epoch, lat, lon)
    ru, rv = _rotate_cw(wu, wv, d["wind_deflection_deg"])
    return cu + d["wind_factor"] * ru, cv + d["wind_factor"] * rv


def _integrate(
    proj: Projector,
    env: EnvField,
    lat: np.ndarray,
    lon: np.ndarray,
    t_from: np.ndarray,
    t_to: float,
    rng: np.random.Generator,
    snapshot_interval_s: float | None = None,
    snapshots: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler integration with diffusion, forward or backward in time.

    `t_from` is per-particle so a line source can release over a window. Direction
    is inferred from the sign of (t_to - t_from).
    """
    d = cfg()["drift"]
    dt = d["timestep_min"] * 60.0
    k = d["diffusion_k_m2s"]

    lat = np.asarray(lat, dtype=float).copy()
    lon = np.asarray(lon, dtype=float).copy()
    t_from = np.asarray(t_from, dtype=float)

    backward = t_to < float(t_from.max())
    sign = -1.0 if backward else 1.0

    t_now = float(t_from.max()) if backward else float(t_from.min())
    next_snap = t_now
    guard = 0
    while (t_now > t_to) if backward else (t_now < t_to):
        guard += 1
        if snapshots is not None and snapshot_interval_s and (
            (t_now <= next_snap) if backward else (t_now >= next_snap)
        ):
            released = (t_from >= t_now) if backward else (t_from <= t_now)
            snapshots.append(
                {
                    "t": t_now,
                    "lat": lat[released].tolist(),
                    "lon": lon[released].tolist(),
                }
            )
            next_snap = t_now + sign * snapshot_interval_s
        if guard > 200_000:
            raise RuntimeError("Drift integration failed to terminate — check timestamps.")

        step = min(dt, abs(t_to - t_now))
        # A particle only moves once its own release time has been passed.
        live = (t_from >= t_now) if backward else (t_from <= t_now)
        if live.any():
            vu, vv = drift_velocity(env, t_now, lat[live], lon[live])
            x, y = proj.to_m(lon[live], lat[live])
            # Diffusion magnitude scales with sqrt(time), and stays positive when
            # integrating backwards: uncertainty grows in both directions.
            sigma = math.sqrt(2.0 * k * step)
            x = x + sign * vu * step + rng.normal(0.0, sigma, size=x.shape)
            y = y + sign * vv * step + rng.normal(0.0, sigma, size=y.shape)
            lon[live], lat[live] = proj.to_deg(x, y)
        t_now += sign * step

    return lat, lon


def fay_radius_m(age_hours: float, base_m: float = 260.0) -> float:
    """Gravity-viscous spreading. Area ~ t^0.75, so radius ~ t^(0.75/2)."""
    e = cfg()["fay"]["spreading_exponent"]
    return base_m * max(age_hours, 0.1) ** (e / 2.0)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def advect_forward(
    release_lat: float,
    release_lon: float,
    release_time: datetime,
    target_time: datetime,
    env_field: EnvField,
    n_particles: int = 500,
    track: np.ndarray | None = None,
    release_duration_min: float | None = None,
    seed: int = 0,
    proj: Projector | None = None,
) -> tuple[np.ndarray, np.ndarray, Polygon]:
    """Advect a release forward. Returns (lat, lon, polygon).

    A moving vessel discharging over some minutes is a *line* source, and that is
    what makes a real slick elongated and aligned with the ship's course. Pass
    `track` as an (N, 3) array of (epoch, lat, lon) to seed particles along the
    vessel's own path; without it the release is a point and the resulting polygon
    carries no meaningful orientation — which would make the 25% orientation term
    in section 4.5.2 dead weight.
    """
    if target_time <= release_time:
        raise ValueError(
            f"target_time {target_time.isoformat()} is not after "
            f"release_time {release_time.isoformat()}"
        )

    rng = np.random.default_rng(seed)
    t0 = release_time.timestamp()
    t1 = target_time.timestamp()
    dur_s = (release_duration_min if release_duration_min is not None else 45.0) * 60.0

    if proj is None:
        proj = Projector(release_lat, release_lon)

    if track is not None and len(track) > 0:
        seg = track[(track[:, 0] >= t0) & (track[:, 0] <= t0 + dur_s)]
        if len(seg) < 2:
            seg = None
    else:
        seg = None

    if seg is not None:
        pick = rng.integers(0, len(seg), n_particles)
        p_lat = seg[pick, 1] + rng.normal(0, 0.0015, n_particles)
        p_lon = seg[pick, 2] + rng.normal(0, 0.0015, n_particles)
        p_t = seg[pick, 0]
    else:
        p_lat = np.full(n_particles, release_lat) + rng.normal(0, 0.0015, n_particles)
        p_lon = np.full(n_particles, release_lon) + rng.normal(0, 0.0015, n_particles)
        p_t = np.full(n_particles, t0)

    lat, lon = _integrate(proj, env_field, p_lat, p_lon, p_t, t1, rng)
    age_h = (t1 - t0) / 3600.0
    poly = particles_to_polygon(proj, lat, lon, fay_radius_m(age_h))
    return lat, lon, poly


def replay_forward(
    release_lat: float,
    release_lon: float,
    release_time: datetime,
    target_time: datetime,
    env_field: EnvField,
    track: np.ndarray | None = None,
    release_duration_min: float = 45.0,
    n_particles: int = 300,
    n_frames: int = 40,
    seed: int = 0,
) -> list[dict]:
    """Same forward simulation, but recording particle positions along the way.

    Feeds the dashboard's drift replay: the animation is the actual winning
    hypothesis being re-simulated, not a decorative loop.
    """
    rng = np.random.default_rng(seed)
    t0, t1 = release_time.timestamp(), target_time.timestamp()
    if t1 <= t0:
        raise ValueError("target_time must be after release_time")

    proj = Projector(release_lat, release_lon)
    dur_s = release_duration_min * 60.0

    seg = None
    if track is not None and len(track):
        s = track[(track[:, 0] >= t0) & (track[:, 0] <= t0 + dur_s)]
        seg = s if len(s) >= 2 else None

    if seg is not None:
        pick = rng.integers(0, len(seg), n_particles)
        p_lat = seg[pick, 1] + rng.normal(0, 0.0015, n_particles)
        p_lon = seg[pick, 2] + rng.normal(0, 0.0015, n_particles)
        p_t = seg[pick, 0]
    else:
        p_lat = np.full(n_particles, release_lat)
        p_lon = np.full(n_particles, release_lon)
        p_t = np.full(n_particles, t0)

    frames: list[dict] = []
    interval = max((t1 - t0) / max(n_frames, 1), 60.0)
    lat, lon = _integrate(
        proj, env_field, p_lat, p_lon, p_t, t1, rng,
        snapshot_interval_s=interval, snapshots=frames,
    )
    frames.append({"t": t1, "lat": lat.tolist(), "lon": lon.tolist()})
    for f in frames:
        f["time"] = datetime.fromtimestamp(f["t"], tz=timezone.utc).isoformat()
        f["hours_after_release"] = round((f["t"] - t0) / 3600.0, 3)
    return frames


def advect_reverse(
    slick_polygon: Polygon,
    acquisition_time: datetime,
    hours_back: float,
    env_field: EnvField,
    n_particles: int = 800,
    step_hours: float = 1.0,
    seed: int = 0,
    proj: Projector | None = None,
) -> list[dict]:
    """Advect an observed slick backwards. Returns one entry per time step.

    Each entry is {hours_back, time, polygon} — where the oil could have been that
    many hours before acquisition. Their union is the origin envelope that layer 6
    prunes candidates against.
    """
    if hours_back <= 0:
        raise ValueError(f"hours_back must be positive, got {hours_back}")

    rng = np.random.default_rng(seed)
    if proj is None:
        c = slick_polygon.centroid
        proj = Projector(c.y, c.x)

    # Seed particles uniformly inside the observed slick.
    minx, miny, maxx, maxy = slick_polygon.bounds
    pts_lat, pts_lon = [], []
    tries = 0
    while len(pts_lat) < n_particles and tries < n_particles * 200:
        tries += 1
        px = rng.uniform(minx, maxx)
        py = rng.uniform(miny, maxy)
        if slick_polygon.contains(Polygon([(px, py), (px, py), (px, py)]).centroid):
            pts_lat.append(py)
            pts_lon.append(px)
    if len(pts_lat) < 10:
        raise RuntimeError("Could not seed particles inside the slick polygon.")

    lat = np.array(pts_lat)
    lon = np.array(pts_lon)
    t_now = acquisition_time.timestamp()

    out: list[dict] = []
    elapsed = 0.0
    while elapsed < hours_back:
        step = min(step_hours, hours_back - elapsed)
        t_next = t_now - step * 3600.0
        lat, lon = _integrate(
            proj, env_field, lat, lon, np.full(len(lat), t_now), t_next, rng
        )
        t_now = t_next
        elapsed += step
        # The envelope widens with time-back: the further back, the less certain.
        poly = particles_to_polygon(proj, lat, lon, fay_radius_m(elapsed) * 1.4)
        out.append(
            {
                "hours_back": round(elapsed, 3),
                "time": datetime.fromtimestamp(t_now, tz=timezone.utc),
                "polygon": poly,
            }
        )
    return out


def origin_envelope(steps: list[dict]) -> Polygon:
    """Union of every reverse-advected step — the full origin envelope."""
    from shapely.ops import unary_union

    u = unary_union([s["polygon"] for s in steps]).buffer(0)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    return u


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Reverse-advect a slick to its origin envelope.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--hours-back", type=float, default=18.0)
    ap.add_argument("--root", default="artifacts")
    a = ap.parse_args()

    from samudra.geo import load_polygon

    d = Path(a.root) / a.incident
    env = EnvField.load(d / "env.npz")
    gj = json.loads((d / "observed_slick.geojson").read_text())
    slick = load_polygon(gj)
    acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])

    steps = advect_reverse(slick, acq, a.hours_back, env)
    env_poly = origin_envelope(steps)

    proj = Projector(slick.centroid.y, slick.centroid.x)
    (d / "origin_envelope.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(env_poly),
                        "properties": {"hours_back": a.hours_back},
                    }
                ]
                + [
                    {
                        "type": "Feature",
                        "geometry": mapping(s["polygon"]),
                        "properties": {"hours_back": s["hours_back"], "time": s["time"].isoformat()},
                    }
                    for s in steps
                ],
            },
            indent=2,
        )
    )

    print(f"incident        : {a.incident}")
    print(f"acquisition     : {acq.isoformat()}")
    print(f"reverse steps   : {len(steps)} over {a.hours_back} h")
    print(f"envelope area   : {proj.polygon_to_m(env_poly).area / 1e6:.1f} km2")
    print(f"written         : {d / 'origin_envelope.geojson'}")


if __name__ == "__main__":
    main()
