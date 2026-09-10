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
from samudra.trust.score import priors_from, score_tracks
from samudra.geo import Projector, load_polygon, polygon_metrics


def resolve_observed_slick(d: Path, mode: str = "auto") -> tuple[object, datetime, dict]:
    """Choose the observed slick: real detection output, or the synthetic one.

    CLAUDE.md build order step 11 requires real detection when a checkpoint
    exists and a fall back to the synthetic slick when it does not, with
    run_demo.sh working either way.

    "auto" additionally refuses a checkpoint that its own metadata marks as NOT
    representative. A 2-epoch CPU smoke checkpoint technically exists, and
    silently routing the demo through it would replace a known-good input with a
    known-bad one while looking like an upgrade. Pass --detection always to
    override.
    """
    gj = json.loads((d / "observed_slick.geojson").read_text())
    acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])

    det_path = d / "detected_slicks.geojson"
    if mode == "never" or not det_path.exists():
        return load_polygon(gj), acq, {
            "source": "synthetic",
            "reason": "no detection output" if mode != "never" else "detection disabled",
        }

    fc = json.loads(det_path.read_text())
    feats = [f for f in fc.get("features", []) if f["properties"].get("confidence", 0) > 0]
    if not feats:
        return load_polygon(gj), acq, {
            "source": "synthetic",
            "reason": "detector produced no confident polygon",
        }

    representative = True
    for f in feats:
        if f["properties"].get("checkpoint_representative") is False:
            representative = False
    meta_p = d / "detection_meta.json"
    if meta_p.exists():
        representative = bool(json.loads(meta_p.read_text()).get("representative", True))

    if mode == "auto" and not representative:
        return load_polygon(gj), acq, {
            "source": "synthetic",
            "reason": (
                "a detection checkpoint exists but is marked NOT representative "
                "(smoke-test weights). Using it would degrade the demo. Run the "
                "full GPU training, or pass --detection always to force it."
            ),
        }

    best = max(feats, key=lambda f: f["properties"]["confidence"])
    return load_polygon(best["geometry"]), acq, {
        "source": "detected",
        "slick_id": best["properties"].get("slick_id"),
        "confidence": best["properties"].get("confidence"),
        "cnn_oil_prob": best["properties"].get("cnn_oil_prob"),
        "baseline_anomaly_z": best["properties"].get("baseline_anomaly_z"),
        "candidates": len(feats),
        "representative": representative,
    }


def run(
    incident: str,
    root: Path = Path("artifacts"),
    hours_back: float = 16.0,
    interval_min: float = 30.0,
    n_particles: int = 250,
    max_candidates: int = 12,
    detection: str = "auto",
    quiet: bool = False,
) -> dict:
    d = root / incident
    cfg = yaml.safe_load(open("config/weights.yaml"))

    slick_gj = json.loads((d / "observed_slick.geojson").read_text())
    observed, acq, slick_source = resolve_observed_slick(d, detection)

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
    if slick_source["source"] == "detected":
        say(f"[0/5] slick source    : DETECTED by the segmenter "
            f"({slick_source['slick_id']}, confidence {slick_source['confidence']:.3f}, "
            f"{slick_source['candidates']} candidate polygons)")
    else:
        say(f"[0/5] slick source    : synthetic — {slick_source['reason']}")

    say(f"[1/5] reverse drift   : {len(steps)} steps over {hours_back:.0f} h, "
        f"envelope {proj.polygon_to_m(envelope).area / 1e6:.0f} km2")

    tracks = load_tracks(d / "ais.parquet")
    window = (acq.timestamp() - hours_back * 3600.0, acq.timestamp())
    candidates, funnel = prune(tracks, steps, aoi_bounds=aoi, search_window=window)
    say(f"[2/5] prune           : {funnel['total_in_scene']} in scene -> "
        f"{funnel['in_envelope']} in envelope")

    candidates = candidates[:max_candidates]

    # --- layer 5: trust and behaviour ------------------------------------
    trust = score_tracks(tracks, cfg, proj)
    priors = priors_from(trust, cfg)
    for mmsi, t in trust.items():
        t["priors"] = priors[mmsi]
    (d / "trust.json").write_text(json.dumps(trust, indent=2))
    n_flagged = sum(1 for t in trust.values() if t["trust_flags"] or t["behaviour_flags"])
    say(f"[3/5] trust           : {n_flagged} of {len(trust)} vessels carry a flag")

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
    say(f"[4/5] hypotheses      : {n_hyp} releases simulated across {len(scored)} vessels")

    ranked = rank.rank_candidates(scored, cfg, priors_by_mmsi=priors)
    funnel["ranked"] = len(ranked)
    say(f"[5/5] ranked          : {len(ranked)} suspects  ({time.time() - t0:.1f}s)")

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
        "slick_source": slick_source,
        "observed_slick": {"geometry": mapping(observed), **metrics},
        "origin_envelope": mapping(envelope),
        "envelope_steps": [
            {"hours_back": s["hours_back"], "time": s["time"].isoformat(),
             "geometry": mapping(s["polygon"])}
            for s in steps
        ],
        "env_summary": env.summary(acq.timestamp()),
        "funnel": funnel,
        "trust_summary": {
            "vessels_scored": len(trust),
            "vessels_flagged": n_flagged,
            "identity_mismatch": [
                m for m, t in trust.items() if t["classification"] == "IDENTITY_MISMATCH"
            ],
        },
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
                "likelihood": r["likelihood"],
                "geometric_score": r["geometric_score"],
                "trust_prior": r["trust_prior"],
                "behaviour_prior": r["behaviour_prior"],
                "proximity_prior": r["proximity_prior"],
                "type_risk_prior": r["type_risk_prior"],
                "gap_coincidence": r["gap_coincidence"],
                "prior_offences": r["prior_offences"],
                "rationale": r["rationale"],
                "trust_score": trust[r["mmsi"]]["score"],
                "classification": trust[r["mmsi"]]["classification"],
                "trust_flags": trust[r["mmsi"]]["trust_flags"],
                "behaviour_flags": trust[r["mmsi"]]["behaviour_flags"],
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
    t = out.get("trust_summary")
    if t:
        print(f"TRUST      {t['vessels_flagged']} of {t['vessels_scored']} vessels flagged;"
              f"  identity mismatch: {t['identity_mismatch'] or 'none'}")

    print()
    print(f"{'#':<3}{'MMSI':<12}{'NAME':<19}{'POST':>7}{'SCORE':>7}{'IoU':>6}"
          f"{'dORI':>6}{'AGE':>6}{'TRUST':>7}{'BEHAV':>7}{'TYPE':>7}  {'CLASS':<18}")
    print("-" * 101)
    for sp in out["suspects"]:
        b = sp["best_hypothesis"]
        print(f"{sp['rank']:<3}{sp['mmsi']:<12}{(sp['vessel_name'] or '')[:18]:<19}"
              f"{sp['posterior']:>7.3f}{b['score']:>7.3f}{b['iou']:>6.3f}"
              f"{b['orientation_delta_deg']:>6.1f}{b['age_hours']:>6.1f}"
              f"{sp['trust_prior']:>7.2f}{sp['behaviour_prior']:>7.2f}"
              f"{sp.get('type_risk_prior', 1.0):>7.2f}  "
              f"{(sp.get('classification') or ''):<18}")

    flagged = [s for s in out["suspects"] if s.get("trust_flags") or s.get("behaviour_flags")]
    if flagged:
        print()
        print("FLAGS ON RANKED SUSPECTS")
        for sp in flagged:
            codes = [f"{fl['code']}({fl['severity'][0]})" for fl in sp.get("trust_flags", [])]
            bcodes = [fl["code"] for fl in sp.get("behaviour_flags", [])]
            print(f"  #{sp['rank']} {sp['mmsi']}  {', '.join(codes + bcodes)}")


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
    ap.add_argument("--detection", choices=["auto", "always", "never"], default="auto",
                    help="use the segmenter's output as the observed slick")
    a = ap.parse_args()

    out = run(
        a.incident, Path(a.root),
        hours_back=a.hours_back, interval_min=a.interval_min, n_particles=a.particles,
        detection=a.detection,
    )
    print_report(out)


if __name__ == "__main__":
    main()
