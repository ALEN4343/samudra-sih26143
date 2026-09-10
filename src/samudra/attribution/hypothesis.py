"""Release hypothesis generation and scoring — CLAUDE.md section 4.5.2.

For each surviving candidate, sample a release every 30 minutes along its own
track inside the plausible window, forward-simulate the discharge, and score the
simulated slick against the observed one.

    score = 0.40*iou + 0.25*centroid + 0.25*orientation + 0.10*area

Weights come from config/weights.yaml and are never tuned to make a scenario
produce the expected answer.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from shapely.geometry import Polygon

from samudra.attribution import drift
from samudra.attribution.prune import Candidate
from samudra.geo import (
    Projector,
    angular_delta_deg,
    centroid_offset_km,
    iou,
    major_axis_deg,
)
from samudra.timeutil import from_epoch


def score_terms(
    proj: Projector, simulated: Polygon, observed: Polygon, cfg: dict
) -> dict:
    """The four terms of section 4.5.2, each in [0, 1], plus the weighted total."""
    w = cfg["score_weights"]
    st = cfg["score_terms"]

    i = iou(proj, simulated, observed)
    off_km = centroid_offset_km(proj, simulated, observed)
    centroid_term = math.exp(-off_km / st["centroid_decay_km"])

    sm = major_axis_deg(proj.polygon_to_m(simulated))
    om = major_axis_deg(proj.polygon_to_m(observed))
    delta = angular_delta_deg(sm, om)
    if st.get("orientation_mode", "cos") == "cos":
        # CLAUDE.md 5.5 specifies cos(radians(delta)).
        orient_term = math.cos(math.radians(delta))
    else:
        orient_term = max(0.0, min(1.0, 1.0 - delta / st["orientation_span_deg"]))
    orient_term = max(0.0, min(1.0, orient_term))

    a = proj.polygon_to_m(simulated).area
    b = proj.polygon_to_m(observed).area
    # CLAUDE.md 5.5 writes this as exp(-abs(log(area_ratio))), which is
    # algebraically identical to min/max and avoids a log of zero.
    area_ratio = (min(a, b) / max(a, b)) if max(a, b) > 0 else 0.0

    total = (
        w["iou"] * i
        + w["centroid"] * centroid_term
        + w["orientation"] * orient_term
        + w["area"] * area_ratio
    )
    return {
        "iou": i,
        "centroid_offset_km": off_km,
        "centroid_term": centroid_term,
        "orientation_delta_deg": delta,
        "orientation_term": orient_term,
        "area_ratio": area_ratio,
        "score": total,
    }


def hypotheses_for(
    candidate: Candidate,
    observed: Polygon,
    acquisition_at: datetime,
    env: drift.EnvField,
    cfg: dict,
    proj: Projector,
    interval_min: float = 30.0,
    min_age_h: float = 1.0,
    window_pad_h: float = 2.0,
    n_particles: int = 250,
    release_duration_min: float = 45.0,
    seed: int = 11,
) -> list[dict]:
    """Sample and score every plausible release along one vessel's track."""
    acq_ts = acquisition_at.timestamp()
    lo, hi = candidate.window

    # Pad the window around the steps the vessel actually matched. The envelope is
    # sampled hourly, so a vessel may intersect only one snapshot while the real
    # release happened somewhat either side of it. Without this padding a single
    # -step match collapses the window to zero width and yields no hypotheses at
    # all — which silently drops the correct answer.
    pad = window_pad_h * 3600.0
    lo = max(lo - pad, float(candidate.track.t[0]))
    hi = min(hi + pad, float(candidate.track.t[-1]))

    # Do not consider releases so recent the oil could not have spread.
    hi = min(hi, acq_ts - min_age_h * 3600.0)
    if hi <= lo:
        return []

    track = candidate.track.xyt()
    out: list[dict] = []
    step = interval_min * 60.0
    ts = lo
    k = 0
    while ts <= hi:
        pos = candidate.track.position_at(ts)
        if pos is not None:
            lat, lon = pos
            try:
                _, _, sim = drift.advect_forward(
                    lat, lon, from_epoch(ts), acquisition_at, env,
                    n_particles=n_particles,
                    track=track,
                    release_duration_min=release_duration_min,
                    seed=seed + k,
                    proj=proj,
                )
            except Exception as exc:  # never swallow silently
                raise RuntimeError(
                    f"forward simulation failed for MMSI {candidate.mmsi} "
                    f"at {from_epoch(ts).isoformat()}"
                ) from exc

            terms = score_terms(proj, sim, observed, cfg)
            out.append(
                {
                    "hypothesis_id": f"hyp-{candidate.mmsi}-{k}",
                    "mmsi": candidate.mmsi,
                    "release_at": from_epoch(ts),
                    "release_lat": lat,
                    "release_lon": lon,
                    "age_hours": (acq_ts - ts) / 3600.0,
                    "geometry": sim,
                    **terms,
                }
            )
        ts += step
        k += 1

    out.sort(key=lambda h: -h["score"])
    return out


def fay_age_hours(observed_area_km2: float, cfg: dict) -> float:
    """Invert the Fay spreading law: area ~ t^0.75.

    `reference_area_km2_at_1h` encodes an assumed discharge volume and oil type.
    This is a consistency check against an assumed spreading rate, not an
    independent measurement, and the report says so.
    """
    f = cfg["fay"]
    a0 = f["reference_area_km2_at_1h"]
    e = f["spreading_exponent"]
    if observed_area_km2 <= 0 or a0 <= 0:
        return float("nan")
    return float((observed_area_km2 / a0) ** (1.0 / e))


def slick_age(top: list[dict], acquisition_at: datetime, observed_area_km2: float, cfg: dict) -> dict:
    """Age from the best hypothesis, ranged over the top 3, cross-checked vs Fay."""
    if not top:
        raise ValueError("cannot estimate slick age with no hypotheses")
    ages = [h["age_hours"] for h in top[:3]]
    best = top[0]["age_hours"]
    fay = fay_age_hours(observed_area_km2, cfg)

    # Agreement is a RATIO test, not an absolute one. A flat +/- 6 h tolerance
    # passes 1.2 h against 5.5 h, which is a factor of four and obviously not
    # agreement; it only looks acceptable because both numbers are small. Fay
    # spreading carries large uncertainty, so a factor-of-N band is the honest
    # standard and it scales correctly with slick age.
    f = cfg["fay"]
    factor = float(f.get("agreement_factor", 2.0))
    ok = False
    ratio = float("nan")
    if fay == fay and fay > 0 and best > 0:
        ratio = fay / best
        ok = (1.0 / factor) <= ratio <= factor

    return {
        "best_hours": best,
        "low_hours": min(ages),
        "high_hours": max(ages),
        "fay_estimate_hours": fay,
        "fay_ratio": ratio,
        "agreement_factor": factor,
        "agrees_with_fay": bool(ok),
    }
