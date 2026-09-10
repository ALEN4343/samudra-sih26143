"""Fabricate an incident over REAL vessel traffic — CLAUDE.md build order step 9.

The synthetic scenarios prove the engine is correct. They do not prove it
survives real data: real AIS has reporting gaps, duplicate broadcasts, missing
headings, hundreds of moored vessels, and traffic that does not politely spread
itself across the AOI.

This module keeps the known answer and throws away the convenient traffic. It
picks one real vessel out of the Houston Ship Channel feed, treats it as the
culprit, releases oil along its actual reported track, and advects that forward
with the same simulator physics `generate.py` uses. Everything the pipeline then
sees — every other vessel, every gap, every dropped field — is real.

Only the environment field and the SAR scene are synthetic, because no wind,
current or radar acquisition is available for that date and box. Both are
labelled as such in ground_truth.json and in the dossier.

CRITICAL: ground_truth.json is for verification only. No pipeline module may
read it.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import mapping

from samudra.synth.generate import (
    Projector,
    advect,
    build_env,
    particles_to_polygon,
    render_scene,
)
from samudra.timeutil import epoch_seconds

# Vessel types that could plausibly discharge oil in this waterway. Inland and
# coastal petroleum barges under tow are the classic Houston Ship Channel case,
# so "Towing" belongs here alongside the tankers.
OIL_CAPABLE = ("Tanker", "Cargo", "Towing")


def _longest_underway_run(ts: np.ndarray, sog: np.ndarray, min_kn: float):
    """(duration_s, start_idx, end_idx) of the longest sustained transit."""
    fast = sog > min_kn
    best = (0.0, 0, 0)
    i = 0
    while i < len(fast):
        if fast[i]:
            j = i
            while j + 1 < len(fast) and fast[j + 1]:
                j += 1
            d = ts[j] - ts[i]
            if d > best[0]:
                best = (float(d), i, j)
            i = j + 1
        else:
            i += 1
    return best


def _track_separation_km(df, proj, mmsi, t0, t1, min_kn=1.0) -> float:
    """Median distance from this vessel to the nearest other MOVING vessel.

    In a single-lane fairway several vessels run the same line minutes apart, and
    their simulated slicks become geometrically indistinguishable. This measures
    how isolated a candidate actually is, so a scenario can be built either way
    deliberately rather than by accident.
    """
    g = df[df.mmsi == mmsi].sort_values("t")
    ts = epoch_seconds(g.t)
    m = (ts >= t0) & (ts <= t1)
    if m.sum() < 3:
        return 0.0
    gx, gy = proj.to_m(g.lon.to_numpy()[m], g.lat.to_numpy()[m])
    gt_ = ts[m]

    others = df[(df.mmsi != mmsi) & (df.sog > min_kn)]
    ots = epoch_seconds(others.t)
    win = (ots >= t0 - 1800) & (ots <= t1 + 1800)
    if win.sum() < 5:
        return 999.0
    ox, oy = proj.to_m(others.lon.to_numpy()[win], others.lat.to_numpy()[win])

    step = max(1, len(gx) // 25)
    d = [float(np.min(np.hypot(ox - x, oy - y))) for x, y in zip(gx[::step], gy[::step])]
    return float(np.median(d)) / 1000.0


def choose_culprit(
    df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    min_run_h: float = 1.0,
    min_kn: float = 5.0,
    interior_frac: float = 0.12,
    prefer_distinctive: bool = False,
    proj=None,
) -> dict:
    """Pick a real vessel that could plausibly have discharged.

    Requirements: an oil-capable type, a sustained transit long enough to lay a
    line source, and a track that stays inside the AOI interior so the drifted
    slick does not immediately leave the box.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    dx = (max_lon - min_lon) * interior_frac
    dy = (max_lat - min_lat) * interior_frac
    interior = (min_lon + dx, min_lat + dy, max_lon - dx, max_lat - dy)

    best = None
    for mmsi, g in df.groupby("mmsi", sort=False):
        if len(g) < 60:
            continue
        vt = g.vessel_type.iloc[0]
        if vt is None or not any(k in str(vt) for k in OIL_CAPABLE):
            continue

        ts = epoch_seconds(g.t)
        dur, i, j = _longest_underway_run(ts, g.sog.to_numpy(), min_kn)
        if dur < min_run_h * 3600.0:
            continue

        lat = g.lat.to_numpy()[i : j + 1]
        lon = g.lon.to_numpy()[i : j + 1]
        if not (
            interior[0] < lon.min() and lon.max() < interior[2]
            and interior[1] < lat.min() and lat.max() < interior[3]
        ):
            continue

        length = g.length_m.iloc[0]
        length = 0.0 if pd.isna(length) else float(length)
        # Prefer a long, fast, large transit: more oil aboard, longer line source.
        rank = dur * float(g.sog.to_numpy()[i : j + 1].mean()) * (1.0 + length / 100.0)
        sep = None
        if prefer_distinctive and proj is not None:
            sep = _track_separation_km(df, proj, int(mmsi), ts[i], ts[j])
            # Weight isolation heavily: a vessel alone on its line is the case
            # where geometric attribution is actually identifiable.
            rank = sep * dur * float(g.sog.to_numpy()[i : j + 1].mean())
        cand = {
            "mmsi": int(mmsi),
            "name": None if pd.isna(g.vessel_name.iloc[0]) else str(g.vessel_name.iloc[0]),
            "vessel_type": str(vt),
            "length_m": length,
            "flag": g.flag.iloc[0],
            "run_hours": dur / 3600.0,
            "mean_sog_kn": float(g.sog.to_numpy()[i : j + 1].mean()),
            "run_start": ts[i],
            "run_end": ts[j],
            "rank": rank,
            "track_separation_km": sep,
        }
        if best is None or rank > best["rank"]:
            best = cand

    if best is None:
        raise RuntimeError(
            "No real vessel met the culprit criteria. Relax --min-run-h or "
            "--min-kn, or widen the AOI."
        )
    return best


def build(
    incident_id: str = "houston-001",
    ais_parquet: Path = Path("data/raw/ais/houston.parquet"),
    out_root: Path = Path("artifacts"),
    age_hours: float = 5.0,
    release_duration_min: float = 45.0,
    seed: int = 4242,
    wind_range: tuple[float, float] = (4.0, 9.0),
    prefer_distinctive: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(ais_parquet)
    if df.empty:
        raise ValueError(f"{ais_parquet} is empty")

    bounds = (
        float(df.lon.min()), float(df.lat.min()),
        float(df.lon.max()), float(df.lat.max()),
    )
    proj = Projector((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)

    culprit = choose_culprit(
        df, bounds, prefer_distinctive=prefer_distinctive, proj=proj
    )
    g = df[df.mmsi == culprit["mmsi"]].sort_values("t")
    ts = epoch_seconds(g.t)

    # Release partway into the sustained transit, leaving room for the full line
    # source before the vessel slows again.
    release_ts = culprit["run_start"] + 0.15 * (culprit["run_end"] - culprit["run_start"])
    release_ts = min(release_ts, culprit["run_end"] - release_duration_min * 60.0)
    release_at = datetime.fromtimestamp(release_ts, tz=timezone.utc)
    acquisition_at = release_at + timedelta(hours=age_hours)

    day_end = datetime.fromtimestamp(float(ts.max()), tz=timezone.utc)
    if acquisition_at > df.t.max().to_pydatetime():
        raise RuntimeError(
            f"acquisition {acquisition_at.isoformat()} falls past the end of the AIS "
            f"feed ({df.t.max()}). Reduce --age-hours."
        )

    # Environment: synthetic. No wind/current field is available for this date and
    # box, and CLAUDE.md 7 already commits to Open-Meteo as the production path.
    env = build_env(
        rng, bounds,
        df.t.min().to_pydatetime() - timedelta(hours=2),
        df.t.max().to_pydatetime() + timedelta(hours=78),
        wind_range,
    )

    out = out_root / incident_id
    out.mkdir(parents=True, exist_ok=True)
    env.save(out / "env.npz")

    # Line source along the vessel's ACTUAL reported track.
    seg = (ts >= release_ts) & (ts <= release_ts + release_duration_min * 60.0)
    seg_lat = g.lat.to_numpy()[seg]
    seg_lon = g.lon.to_numpy()[seg]
    seg_t = ts[seg]
    if len(seg_lat) < 2:
        raise RuntimeError("release window contains fewer than two real AIS fixes")

    n_particles = 900
    pick = rng.integers(0, len(seg_lat), n_particles)
    p_lat = seg_lat[pick] + rng.normal(0, 0.0006, n_particles)
    p_lon = seg_lon[pick] + rng.normal(0, 0.0006, n_particles)
    p_t = seg_t[pick]

    f_lat, f_lon = advect(proj, env, p_lat, p_lon, p_t, acquisition_at, rng)
    slick = particles_to_polygon(proj, f_lat, f_lon, age_hours)
    sx, sy = proj.to_m(*slick.exterior.coords.xy)
    from shapely.geometry import Polygon as _P

    area_km2 = _P(np.column_stack([sx, sy])).area / 1e6

    # Real vessel positions at acquisition, for the synthetic scene.
    acq_ts = acquisition_at.timestamp()
    positions = []
    for mmsi, gg in df.groupby("mmsi", sort=False):
        tt = epoch_seconds(gg.t)
        k = int(np.argmin(np.abs(tt - acq_ts)))
        if abs(tt[k] - acq_ts) <= 1800:
            positions.append((float(gg.lon.iloc[k]), float(gg.lat.iloc[k])))

    scene_meta, _ = render_scene(
        rng, proj, out / "scene.tif", slick, positions, seed, aoi=bounds
    )

    # The real AIS feed, copied unchanged into the incident.
    df.to_parquet(out / "ais.parquet", index=False)

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
                            "traffic": "real",
                        },
                    }
                ],
            },
            indent=2,
        )
    )

    gt = {
        "_warning": "VERIFICATION ONLY. No pipeline module may read this file.",
        "scenario_id": incident_id,
        "traffic": "real",
        "ais_source": str(ais_parquet),
        "synthetic_components": ["environment field", "SAR scene", "the release itself"],
        "seed": seed,
        "culprit_mmsi": culprit["mmsi"],
        "culprit_name": culprit["name"],
        "culprit_type": culprit["vessel_type"],
        "culprit_length_m": culprit["length_m"],
        "culprit_flag": culprit["flag"],
        "culprit_run_hours": round(culprit["run_hours"], 3),
        "culprit_mean_sog_kn": round(culprit["mean_sog_kn"], 2),
        "culprit_track_separation_km": culprit.get("track_separation_km"),
        "true_release_at": release_at.isoformat(),
        "true_release_lat": float(seg_lat[0]),
        "true_release_lon": float(seg_lon[0]),
        "release_duration_min": release_duration_min,
        "acquisition_at": acquisition_at.isoformat(),
        "aoi_bounds": list(bounds),
        "slick_area_km2": round(area_km2, 3),
        "slick_age_hours": age_hours,
        "n_vessels": int(df.mmsi.nunique()),
        "planted_anomalies": {},  # none: the anomalies in this feed are real
        "scene": scene_meta,
    }
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fabricate a known-answer incident over real AIS traffic."
    )
    ap.add_argument("--incident", default="houston-001")
    ap.add_argument("--ais", default="data/raw/ais/houston.parquet")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--age-hours", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--distinctive", action="store_true",
                    help="pick a culprit that is isolated from other traffic, "
                         "rather than one sharing a crowded fairway")
    a = ap.parse_args()

    gt = build(a.incident, Path(a.ais), Path(a.out), age_hours=a.age_hours,
               seed=a.seed, prefer_distinctive=a.distinctive)

    print(f"incident            : {gt['scenario_id']}   (traffic: REAL)")
    print(f"AIS source          : {gt['ais_source']}")
    print(f"vessels in feed     : {gt['n_vessels']}")
    print(f"culprit MMSI        : {gt['culprit_mmsi']}  {gt['culprit_name']} "
          f"({gt['culprit_type']}, {gt['culprit_length_m']:.0f} m, {gt['culprit_flag']})")
    print(f"sustained transit   : {gt['culprit_run_hours']:.2f} h at "
          f"{gt['culprit_mean_sog_kn']:.1f} kn")
    if gt.get("culprit_track_separation_km") is not None:
        print(f"track separation    : {gt['culprit_track_separation_km']:.2f} km "
              f"from the nearest other moving vessel")
    print(f"true release at     : {gt['true_release_at']}")
    print(f"true release pos    : {gt['true_release_lat']:.5f} N, "
          f"{gt['true_release_lon']:.5f} E")
    print(f"acquisition at      : {gt['acquisition_at']}")
    print(f"slick age / area    : {gt['slick_age_hours']:.1f} h / "
          f"{gt['slick_area_km2']:.2f} km2")
    print(f"synthetic parts     : {', '.join(gt['synthetic_components'])}")


if __name__ == "__main__":
    main()
