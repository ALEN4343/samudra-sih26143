"""Ranking and posteriors — CLAUDE.md section 4.5.4.

    weighted_i  = best_score_i * trust_prior_i * behaviour_prior_i * proximity_prior_i
    posterior_i = softmax(weighted / T)_i

Until layer 5 exists every prior is 1.0. Each suspect carries a rationale a human
investigator can read without knowing the formula.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"softmax temperature must be positive, got {temperature}")
    z = (x - x.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


def _rationale(h: dict, cand, priors: dict, rank: int) -> list[str]:
    """Plain-English justification. No formula literacy required."""
    bullets = [
        f"Inside the reverse-drift origin envelope at {len(cand.hit_times)} of the "
        f"reconstructed time steps, from {min(cand.hit_hours_back):.0f} to "
        f"{max(cand.hit_hours_back):.0f} hours before acquisition.",
        f"Simulating a discharge at {h['release_at'].strftime('%Y-%m-%d %H:%M UTC')} "
        f"reproduces {h['iou'] * 100:.0f}% overlap with the observed slick.",
        f"Simulated slick centre lands {h['centroid_offset_km']:.1f} km from the "
        f"observed centre.",
        f"Slick long axis differs from this vessel's course by "
        f"{h['orientation_delta_deg']:.0f} degrees.",
        f"Simulated area is within a factor of {1 / max(h['area_ratio'], 1e-6):.2f} "
        f"of the observed area.",
    ]
    vt = cand.track.meta.get("vessel_type")
    if vt:
        bullets.append(f"Vessel type on AIS: {vt}.")
    for name, val in priors.items():
        if abs(val - 1.0) > 1e-9:
            direction = "raised" if val > 1.0 else "lowered"
            bullets.append(f"Posterior {direction} by the {name} prior ({val:.2f}x).")
    return bullets


def rank_candidates(
    scored: list[dict],
    cfg: dict,
    priors_by_mmsi: dict[int, dict] | None = None,
) -> list[dict]:
    """Rank scored candidates into posteriors.

    `scored` is one entry per candidate: {candidate, hypotheses (sorted)}.
    `priors_by_mmsi` supplies trust/behaviour priors once layer 5 exists.
    """
    live = [s for s in scored if s["hypotheses"]]
    if not live:
        return []

    priors_by_mmsi = priors_by_mmsi or {}
    rows = []
    for s in live:
        cand = s["candidate"]
        best = s["hypotheses"][0]
        p = priors_by_mmsi.get(cand.mmsi, {})
        trust = float(p.get("trust_prior", 1.0))
        behaviour = float(p.get("behaviour_prior", 1.0))
        proximity = float(p.get("proximity_prior", 1.0))
        rows.append(
            {
                "candidate": cand,
                "mmsi": cand.mmsi,
                "vessel_name": cand.track.meta.get("vessel_name"),
                "best": best,
                "hypotheses": s["hypotheses"],
                "trust_prior": trust,
                "behaviour_prior": behaviour,
                "proximity_prior": proximity,
                "weighted": best["score"] * trust * behaviour * proximity,
            }
        )

    post = softmax(
        np.array([r["weighted"] for r in rows]), cfg["priors"]["temperature"]
    )
    for r, pv in zip(rows, post):
        r["posterior"] = float(pv)

    rows.sort(key=lambda r: -r["posterior"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
        r["rationale"] = _rationale(
            r["best"],
            r["candidate"],
            {
                "trust": r["trust_prior"],
                "behaviour": r["behaviour_prior"],
                "proximity": r["proximity_prior"],
            },
            i,
        )
    return rows
