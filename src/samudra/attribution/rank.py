"""Ranking and posteriors — CLAUDE.md section 5.5.

The spec is explicit about the order of operations:

    likelihood_i = softmax(geometric_score)_i          # across candidates
    posterior_i  = likelihood_i
                 * trust_prior_i
                 * behaviour_prior_i
                 * type_risk_prior_i
                 * (1.8 if gap_coincidence_i else 1.0)
                 * (1 + 0.4 * prior_offences_i)
    renormalise

Softmax first, then priors, then renormalise — not one combined softmax. The two
orders give different numbers, because a softmax is not linear.

Each Suspect carries a rationale a human investigator can read without knowing
the formula. Never output a binary accusation: ranked candidates with likelihood
ratios only.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"softmax temperature must be positive, got {temperature}")
    z = (x - x.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


def type_risk_prior(vessel_type: str | None, cfg: dict) -> float:
    """tanker/bulk > container > fishing, matched on substrings of the AIS type."""
    table = cfg["priors"]["type_risk"]
    if not vessel_type:
        return float(table["default"])
    vt = str(vessel_type).lower()
    # Most specific wins: check the higher-risk keys before the generic ones.
    for key in ("tanker", "bulk", "container", "cargo", "tug", "fishing"):
        if key in vt and key in table:
            return float(table[key])
    return float(table["default"])


def gap_coincides_with_release(cand, release_at, window_h: float = 2.0) -> bool:
    """Did the vessel go dark around the time it is estimated to have discharged?

    Going silent exactly over the release window is one of the strongest single
    behavioural signals in the whole system, so it is scored per-hypothesis
    rather than as a general 'this vessel has gaps somewhere' flag.
    """
    t = cand.track.t
    if len(t) < 2:
        return False
    rel = release_at.timestamp()
    gaps = np.diff(t)
    starts = t[:-1]
    big = gaps > 15 * 60.0
    if not big.any():
        return False
    lo, hi = rel - window_h * 3600.0, rel + window_h * 3600.0
    for s, g in zip(starts[big], gaps[big]):
        if s <= hi and (s + g) >= lo:
            return True
    return False


def _rationale(h: dict, cand, priors: dict) -> list[str]:
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
    if priors.get("gap_coincidence"):
        bullets.append(
            "This vessel stopped reporting AIS around the estimated release time."
        )
    if priors.get("prior_offences"):
        bullets.append(
            f"{priors['prior_offences']} previously confirmed discharge(s) on record "
            f"for this MMSI."
        )
    for name, val in (
        ("trust", priors["trust_prior"]),
        ("behaviour", priors["behaviour_prior"]),
        ("vessel-type risk", priors["type_risk_prior"]),
    ):
        if abs(val - 1.0) > 1e-9:
            direction = "raised" if val > 1.0 else "lowered"
            bullets.append(f"Posterior {direction} by the {name} prior ({val:.2f}x).")
    return bullets


def rank_candidates(
    scored: list[dict],
    cfg: dict,
    priors_by_mmsi: dict[int, dict] | None = None,
    offences_by_mmsi: dict[int, int] | None = None,
) -> list[dict]:
    """Rank scored candidates into posteriors, per section 5.5.

    `scored` is one entry per candidate: {candidate, hypotheses (sorted)}.
    `offences_by_mmsi` comes from the vessel dossier DB (connector A). Absent it,
    every vessel is treated as a first offender.
    """
    live = [s for s in scored if s["hypotheses"]]
    if not live:
        return []

    priors_by_mmsi = priors_by_mmsi or {}
    offences_by_mmsi = offences_by_mmsi or {}
    pcfg = cfg["priors"]

    rows = []
    for s in live:
        cand = s["candidate"]
        best = s["hypotheses"][0]
        p = priors_by_mmsi.get(cand.mmsi, {})

        gap = gap_coincides_with_release(cand, best["release_at"])
        offences = int(offences_by_mmsi.get(cand.mmsi, 0))

        rows.append(
            {
                "candidate": cand,
                "mmsi": cand.mmsi,
                "vessel_name": cand.track.meta.get("vessel_name"),
                "best": best,
                "hypotheses": s["hypotheses"],
                "geometric_score": best["score"],
                "trust_prior": float(p.get("trust_prior", 1.0)),
                "behaviour_prior": float(p.get("behaviour_prior", 1.0)),
                "proximity_prior": float(p.get("proximity_prior", 1.0)),
                "type_risk_prior": type_risk_prior(
                    cand.track.meta.get("vessel_type"), cfg
                ),
                "gap_coincidence": bool(gap),
                "prior_offences": offences,
            }
        )

    # Stage 1: softmax over the geometric score alone -> likelihood.
    like = softmax(
        np.array([r["geometric_score"] for r in rows]), pcfg["temperature"]
    )
    for r, lv in zip(rows, like):
        r["likelihood"] = float(lv)

    # Stage 2: multiply the priors through.
    for r in rows:
        r["posterior"] = (
            r["likelihood"]
            * r["trust_prior"]
            * r["behaviour_prior"]
            * r["proximity_prior"]
            * r["type_risk_prior"]
            * (pcfg["gap_coincidence"] if r["gap_coincidence"] else 1.0)
            * (1.0 + pcfg["prior_offence_weight"] * r["prior_offences"])
        )

    # Stage 3: renormalise.
    total = sum(r["posterior"] for r in rows)
    if total <= 0:
        raise RuntimeError("all posteriors collapsed to zero — check the priors")
    for r in rows:
        r["posterior"] /= total

    rows.sort(key=lambda r: -r["posterior"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
        r["rationale"] = _rationale(r["best"], r["candidate"], r)
    return rows
