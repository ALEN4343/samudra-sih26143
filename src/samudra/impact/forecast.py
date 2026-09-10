"""Forward forecast — CLAUDE.md layer 8.

Takes the observed slick and pushes it forward to +24/48/72h with the same drift
model used for attribution. The uncertainty cone comes from the actual spread of
the particle cloud, not from a fixed buffer: it widens because the particles
genuinely diverge under diffusion and a spatially varying current.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import Polygon, mapping

from samudra.attribution import drift
from samudra.geo import Projector, load_polygon, particles_to_polygon, polygon_metrics

HORIZONS = (24, 48, 72)


def seed_particles(
    slick: Polygon, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform particles inside the observed slick."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    minx, miny, maxx, maxy = slick.bounds
    pre = prep(slick)
    lat, lon = [], []
    tries = 0
    while len(lat) < n and tries < n * 300:
        tries += 1
        px, py = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if pre.contains(Point(px, py)):
            lat.append(py)
            lon.append(px)
    if len(lat) < 10:
        raise RuntimeError("could not seed particles inside the observed slick")
    return np.array(lat), np.array(lon)


def forecast(
    slick: Polygon,
    acquisition_at: datetime,
    env: drift.EnvField,
    slick_age_hours: float = 0.0,
    horizons: tuple[int, ...] = HORIZONS,
    n_particles: int = 700,
    seed: int = 17,
) -> list[dict]:
    """Advect the observed slick forward, returning one entry per horizon."""
    rng = np.random.default_rng(seed)
    proj = Projector(slick.centroid.y, slick.centroid.x)
    lat, lon = seed_particles(slick, n_particles, rng)

    out: list[dict] = []
    t_now = acquisition_at.timestamp()
    for h in sorted(horizons):
        t_target = acquisition_at.timestamp() + h * 3600.0
        lat, lon = drift._integrate(
            proj, env, lat, lon, np.full(len(lat), t_now), t_target, rng
        )
        t_now = t_target

        # Total age drives Fay spreading: oil keeps spreading from its release,
        # not from the moment we happened to photograph it.
        total_age = slick_age_hours + h
        spread = drift.fay_radius_m(total_age)
        poly = particles_to_polygon(proj, lat, lon, spread)

        # Uncertainty cone: the convex hull of the particle cloud inflated by one
        # standard deviation of its own spread.
        x, y = proj.to_m(lon, lat)
        sd = float(np.hypot(x.std(), y.std()))
        hull = Polygon(np.column_stack([x, y])).convex_hull.buffer(spread + sd)
        cone = proj.polygon_to_deg(hull)

        m = polygon_metrics(proj, poly)
        out.append(
            {
                "horizon_hours": h,
                "valid_at": datetime.fromtimestamp(t_target, tz=acquisition_at.tzinfo),
                "geometry": poly,
                "uncertainty_cone": cone,
                "area_km2": m["area_km2"],
                "cone_area_km2": polygon_metrics(proj, cone)["area_km2"],
                "particle_spread_km": sd / 1000.0,
                "centroid_lat": m["centroid_lat"],
                "centroid_lon": m["centroid_lon"],
            }
        )
    return out


def main() -> None:
    from samudra.impact.coastline import coastline_impacts, load_coastline

    ap = argparse.ArgumentParser(description="Forecast a slick forward and assess impact.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    a = ap.parse_args()

    d = Path(a.root) / a.incident
    gj = json.loads((d / "observed_slick.geojson").read_text())
    slick = load_polygon(gj)
    acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])
    env = drift.EnvField.load(d / "env.npz")

    age = 0.0
    inc_path = d / "incident.json"
    if inc_path.exists():
        age = json.loads(inc_path.read_text())["slick_age"]["best_hours"]

    # Integrate on a 6-hour grid so the shore ETA is not quantised to the three
    # reported horizons, then publish only 24/48/72h. Stepping finer costs almost
    # nothing because the integration is incremental either way.
    grid = tuple(range(6, 73, 6))
    fine = forecast(slick, acq, env, slick_age_hours=age, horizons=grid)

    coast = load_coastline()
    impacts = coastline_impacts(fine, coast, acq)
    by_h = {i["horizon_hours"]: i for i in impacts}
    for f in fine:
        f["coastline"] = by_h[f["horizon_hours"]]

    print(f"incident {a.incident}   slick age at acquisition {age:.1f} h")
    print(f"coastline: {coast['source']}")
    print()
    print(f"{'HORIZON':>8}{'AREA km2':>11}{'CONE km2':>11}{'SPREAD km':>11}{'TO SHORE km':>13}")
    print("-" * 56)

    for f in fine:
        if f["horizon_hours"] not in HORIZONS:
            continue
        imp = f["coastline"]
        (d / f"forecast_{f['horizon_hours']}h.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": mapping(f["geometry"]),
                         "properties": {"kind": "forecast",
                                        "horizon_hours": f["horizon_hours"],
                                        "valid_at": f["valid_at"].isoformat(),
                                        "area_km2": f["area_km2"], **imp}},
                        {"type": "Feature", "geometry": mapping(f["uncertainty_cone"]),
                         "properties": {"kind": "uncertainty_cone",
                                        "horizon_hours": f["horizon_hours"],
                                        "area_km2": f["cone_area_km2"]}},
                    ],
                },
                indent=2,
            )
        )
        print(f"{f['horizon_hours']:>6}h{f['area_km2']:>11.1f}{f['cone_area_km2']:>11.1f}"
              f"{f['particle_spread_km']:>11.1f}"
              f"{'n/a' if imp['distance_to_coast_km'] is None else format(imp['distance_to_coast_km'], '.1f'):>13}")

    hit = next((i for i in impacts if i["coastline_intersects"]), None)
    print()
    if hit:
        print(f"COASTLINE IMPACT: shore contact by +{hit['horizon_hours']}h, "
              f"{hit['affected_shoreline_km']:.1f} km affected")
        print(f"ETA {hit['coastline_eta']}")
    elif impacts and not impacts[0].get("coastline_covers_aoi", True):
        print(f"COASTLINE IMPACT: not assessed - {impacts[0]['coastline_source']}")
    else:
        closest = min(impacts, key=lambda i: i["distance_to_coast_km"])
        print(f"COASTLINE IMPACT: none within 72 h; closest approach "
              f"{closest['distance_to_coast_km']:.1f} km at +{closest['horizon_hours']}h")

    print()
    print("written: forecast_24h.geojson, forecast_48h.geojson, forecast_72h.geojson")


if __name__ == "__main__":
    main()
