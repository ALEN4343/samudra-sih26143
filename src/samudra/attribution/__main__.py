"""Attribution CLI — python -m samudra.attribution --incident demo-001

Runs the full layer 6 + 7 chain: reverse-drift, prune, hypothesise, rank, and
write incident.json for the dashboard and dossier.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import mapping

from samudra.attribution import drift, hypothesis, rank
from samudra.attribution.prune import load_tracks, prune
from samudra.geo import Projector, load_polygon, polygon_metrics


def run(
    incident: str,
    root: Path = Path("artifacts"),
    hours_back: float = 16.0,
    interval_min: float = 30.0,
    n_particles: int = 250,
    max_candidates: int = 12,
    quiet: bool = False,
) -> dict:
    d = root / incident
    cfg = yaml.safe_load(open("config/weights.yaml"))

    slick_gj = json.loads((d / "observed_slick.geojson").read_text())
    observed = load_polygon(slick_gj)
    acq = datetime.fromisoformat(slick_gj["features"][0]["properties"]["acquisition_at"])

    env = drift.EnvField.load(d / "env.npz")
    aoi = (
        float(env.lons.min()), float(env.lats.min()),
        float(env.lons.max()), float(env.lats.max()),
    )
    proj = Projector.for_bounds(aoi)

    def say(msg):
        if not quiet:
            print(msg, flush=True)

    # --- layer 6: reverse drift and prune --------------------------------
    t0 = time.time()
    steps = drift.advect_reverse(observed, acq, hours_back, env, n_particles=800, seed=3)
    envelope = drift.origin_envelope(steps)
    say(f"[1/4] reverse drift   : {len(steps)} steps over {hours_back:.0f} h, "
        f"envelope {proj.polygon_to_m(envelope).area / 1e6:.0f} km2")

    tracks = load_tracks(d / "ais.parquet")
    window = (acq.timestamp() - hours_back * 3600.0, acq.timestamp())
    candidates, funnel = prune(tracks, steps, aoi_bounds=aoi, search_window=window)
    say(f"[2/4] prune           : {funnel['total_in_scene']} in scene -> "
        f"{funnel['in_envelope']} in envelope")

    candidates = candidates[:max_candidates]

    # --- layer 7: hypotheses ---------------------------------------------
    scored = []
    for c in candidates:
        hs = hypothesis.hypotheses_for(
            c, observed, acq, env, cfg, proj,
            interval_min=interval_min, n_particles=n_particles,
        )
        if hs:
            scored.append({"candidate": c, "hypotheses": hs})
    funnel["scored"] = len(scored)
    n_hyp = sum(len(s["hypotheses"]) for s in scored)
    say(f"[3/4] hypotheses      : {n_hyp} releases simulated across {len(scored)} vessels")

    ranked = rank.rank_candidates(scored, cfg)
    funnel["ranked"] = len(ranked)
    say(f"[4/4] ranked          : {len(ranked)} suspects  ({time.time() - t0:.1f}s)")

    if not ranked:
        raise RuntimeError(
            "No candidate produced a scoreable hypothesis. Either the envelope is "
            "empty or every candidate's AIS coverage misses the release window."
        )

    # --- slick age --------------------------------------------------------
    metrics = polygon_metrics(proj, observed)
    age = hypothesis.slick_age(
        ranked[0]["hypotheses"], acq, metrics["area_km2"], cfg
    )

    # --- write incident.json ---------------------------------------------
    out = {
        "incident_id": incident,
        "acquisition_at": acq.isoformat(),
        "aoi_bounds": list(aoi),
        "synthetic": bool(slick_gj["features"][0]["properties"].get("synthetic")),
        "observed_slick": {"geometry": mapping(observed), **metrics},
        "origin_envelope": mapping(envelope),
        "envelope_steps": [
            {"hours_back": s["hours_back"], "time": s["time"].isoformat(),
             "geometry": mapping(s["polygon"])}
            for s in steps
        ],
        "env_summary": env.summary(acq.timestamp()),
        "funnel": funnel,
        "slick_age": age,
        "suspects": [
            {
                "rank": r["rank"],
                "mmsi": r["mmsi"],
                "vessel_name": r["vessel_name"],
                "vessel_type": r["candidate"].track.meta.get("vessel_type"),
                "length_m": r["candidate"].track.meta.get("length_m"),
                "flag": r["candidate"].track.meta.get("flag"),
                "posterior": r["posterior"],
                "trust_prior": r["trust_prior"],
                "behaviour_prior": r["behaviour_prior"],
                "proximity_prior": r["proximity_prior"],
                "rationale": r["rationale"],
                "best_hypothesis": _hyp_json(r["best"]),
                "top_hypotheses": [_hyp_json(h) for h in r["hypotheses"][:3]],
            }
            for r in ranked
        ],
    }
    (d / "incident.json").write_text(json.dumps(out, indent=2))
    return out


def _hyp_json(h: dict) -> dict:
    return {
        "hypothesis_id": h["hypothesis_id"],
        "mmsi": h["mmsi"],
        "release_at": h["release_at"].isoformat(),
        "release_lat": h["release_lat"],
        "release_lon": h["release_lon"],
        "age_hours": h["age_hours"],
        "iou": h["iou"],
        "centroid_offset_km": h["centroid_offset_km"],
        "centroid_term": h["centroid_term"],
        "orientation_delta_deg": h["orientation_delta_deg"],
        "orientation_term": h["orientation_term"],
        "area_ratio": h["area_ratio"],
        "score": h["score"],
        "simulated_geometry": mapping(h["geometry"]),
    }


def print_report(out: dict) -> None:
    f = out["funnel"]
    print("\n" + "=" * 78)
    print(f"INCIDENT {out['incident_id']}   acquisition {out['acquisition_at']}")
    print("=" * 78)
    e = out["env_summary"]
    print(f"wind {e['mean_wind_speed_ms']:.1f} m/s toward {e['mean_wind_dir_deg']:.0f} deg   "
          f"current {e['mean_current_speed_ms']:.2f} m/s toward {e['mean_current_dir_deg']:.0f} deg   "
          f"gate {'PASS' if e['wind_gate_pass'] else 'FAIL'}")
    s = out["observed_slick"]
    print(f"observed slick: {s['area_km2']:.1f} km2, long axis {s['major_axis_deg']:.0f} deg, "
          f"eccentricity {s['eccentricity']:.3f}")

    print(f"\nFUNNEL   in scene {f['total_in_scene']}  ->  in envelope {f['in_envelope']}"
          f"  ->  scored {f['scored']}  ->  ranked {f['ranked']}")

    a = out["slick_age"]
    agree = "agrees" if a["agrees_with_fay"] else "DISAGREES"
    print(f"SLICK AGE  {a['best_hours']:.1f} h  (range {a['low_hours']:.1f}-{a['high_hours']:.1f} h "
          f"across top 3);  Fay cross-check {a['fay_estimate_hours']:.1f} h -> {agree}")

    print(f"\n{'#':<3}{'MMSI':<12}{'NAME':<20}{'POST':>7}{'SCORE':>7}{'IoU':>7}"
          f"{'dCEN':>7}{'dORI':>7}{'AREA':>7}{'AGE':>7}")
    print("-" * 78)
    for sp in out["suspects"]:
        b = sp["best_hypothesis"]
        print(f"{sp['rank']:<3}{sp['mmsi']:<12}{(sp['vessel_name'] or '')[:19]:<20}"
              f"{sp['posterior']:>7.3f}{b['score']:>7.3f}{b['iou']:>7.3f}"
              f"{b['centroid_offset_km']:>7.1f}{b['orientation_delta_deg']:>7.1f}"
              f"{b['area_ratio']:>7.2f}{b['age_hours']:>7.1f}")

    print("\nTOP SUSPECT RATIONALE")
    for line in out["suspects"][0]["rationale"]:
        print(f"  - {line}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SAMUDRA attribution for an incident.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--hours-back", type=float, default=16.0)
    ap.add_argument("--interval-min", type=float, default=30.0)
    ap.add_argument("--particles", type=int, default=250)
    a = ap.parse_args()

    out = run(
        a.incident, Path(a.root),
        hours_back=a.hours_back, interval_min=a.interval_min, n_particles=a.particles,
    )
    print_report(out)


if __name__ == "__main__":
    main()
