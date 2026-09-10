"""Trust, identity and behaviour — CLAUDE.md section 4.4 (layer 5).

Two independent question sets over the same AIS track:

  Trust     — is this report internally consistent? Does the vessel's own data
              contradict itself, or contradict physics?
  Behaviour — is this vessel acting like a ship that is about to discharge?

Trust flags lower confidence in the *data*. Behaviour flags raise suspicion of
the *vessel*. They must not be conflated: a vessel with clean AIS that loiters
for three hours is a strong suspect with trustworthy data.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from samudra.attribution.prune import Track, load_tracks
from samudra.trust.behaviour import behaviour_flags
from samudra.geo import Projector

KN_TO_MS = 0.514444

# MID prefix -> flag code. Mirrors the generator's table; in production this is
# the ITU MID register.
MID_FLAGS: dict[int, str] = {
    419: "IN", 422: "IR", 463: "PK", 461: "OM", 470: "AE", 403: "SA",
    563: "SG", 636: "LR", 538: "MH", 371: "PA", 477: "HK", 249: "MT",
    240: "GR", 525: "ID", 412: "CN", 366: "US", 367: "US", 368: "US",
    369: "US", 338: "US", 316: "CA", 232: "GB", 235: "GB",
}


def _flag(code: str, severity: str, detail: str, at=None, lat=None, lon=None) -> dict:
    return {
        "code": code,
        "severity": severity,
        "detail": detail,
        "at": at.isoformat() if isinstance(at, datetime) else at,
        "lat": lat,
        "lon": lon,
    }


def _segment_speeds(proj: Projector, tr: Track) -> tuple[np.ndarray, np.ndarray]:
    """Derived speed (kn) and elapsed time (s) between consecutive fixes."""
    x, y = proj.to_m(tr.lon, tr.lat)
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(tr.t)
    dist = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_ms = np.where(dt > 0, dist / np.maximum(dt, 1e-9), 0.0)
    return speed_ms / KN_TO_MS, dt


# --------------------------------------------------------------------------
# Trust checks
# --------------------------------------------------------------------------


def trust_flags(proj: Projector, tr: Track, cfg: dict, all_tracks=None) -> list[dict]:
    out: list[dict] = []
    t = cfg["trust"]

    if len(tr.t) < 3:
        out.append(_flag("SPARSE_TRACK", "INFO", f"only {len(tr.t)} AIS fixes"))
        return out

    derived_kn, dt = _segment_speeds(proj, tr)

    # Derived speed is a MEAN over the interval, so it must be compared against
    # the mean of the SOG reported at each end - not against the instantaneous
    # SOG at the closing fix. Comparing against the endpoint alone flags every
    # genuine sharp deceleration as a data inconsistency, because the vessel
    # really was moving faster for most of that interval.
    reported_kn = (tr.sog[:-1] + tr.sog[1:]) / 2.0

    # Exclude segments spanning a real manoeuvre. Across a sharp acceleration or
    # deceleration the vessel's speed is not linear within the interval, so even
    # the endpoint mean understates how far it actually travelled. A loiter has
    # exactly two such transitions (entry and exit), which is indistinguishable
    # from a spoof's jump-out-and-back on segment count alone. Spoofing is
    # detectable because the position moves while the reported speed stays flat.
    steady = np.abs(tr.sog[1:] - tr.sog[:-1]) <= t.get("manoeuvre_sog_delta_kn", 3.0)

    disagree = np.abs(derived_kn - reported_kn)
    bad = np.flatnonzero((disagree > t["sog_disagreement_kn"]) & steady)
    if len(bad) >= 2:
        i = int(bad[np.argmax(disagree[bad])])
        out.append(
            _flag(
                "SOG_DISAGREEMENT", "WARN",
                f"reported {reported_kn[i]:.1f} kn vs {derived_kn[i]:.1f} kn derived "
                f"from position, on {len(bad)} segment(s)",
                at=datetime.fromtimestamp(tr.t[i + 1], tz=timezone.utc),
                lat=float(tr.lat[i + 1]), lon=float(tr.lon[i + 1]),
            )
        )

    # Position teleport: no vessel makes 60 kn over ground.
    tele = np.flatnonzero(derived_kn > 60.0)
    if len(tele):
        i = int(tele[np.argmax(derived_kn[tele])])
        x, y = proj.to_m(tr.lon[i : i + 2], tr.lat[i : i + 2])
        jump_km = math.hypot(x[1] - x[0], y[1] - y[0]) / 1000.0
        out.append(
            _flag(
                "POSITION_TELEPORT", "CRITICAL",
                f"{jump_km:.1f} km in {dt[i]:.0f} s — implied {derived_kn[i]:.0f} kn",
                at=datetime.fromtimestamp(tr.t[i + 1], tz=timezone.utc),
                lat=float(tr.lat[i + 1]), lon=float(tr.lon[i + 1]),
            )
        )

    # Impossible acceleration.
    if len(derived_kn) > 1:
        dv = np.diff(derived_kn) * KN_TO_MS
        acc = np.abs(dv) / np.maximum(dt[1:], 1.0)
        hot = np.flatnonzero(acc > 1.0)  # m/s^2; a loaded ship manages ~0.1
        if len(hot):
            i = int(hot[np.argmax(acc[hot])])
            out.append(
                _flag(
                    "IMPOSSIBLE_ACCELERATION", "CRITICAL",
                    f"{acc[i]:.2f} m/s2 between fixes",
                    at=datetime.fromtimestamp(tr.t[i + 2], tz=timezone.utc),
                )
            )

    # MMSI MID prefix vs declared flag.
    mid = int(str(tr.mmsi)[:3]) if len(str(tr.mmsi)) >= 9 else None
    declared = tr.meta.get("flag")
    if mid is not None and declared:
        expected = MID_FLAGS.get(mid)
        if expected and expected != declared:
            out.append(
                _flag(
                    "MID_FLAG_MISMATCH", "CRITICAL",
                    f"MMSI prefix {mid} implies {expected}, AIS declares {declared}",
                )
            )
        elif expected is None:
            out.append(_flag("UNKNOWN_MID", "INFO", f"MMSI prefix {mid} not in MID register"))

    # AIS gaps inside the AOI.
    gap_min = t["ais_gap_min"]
    gaps = np.flatnonzero(dt > gap_min * 60.0)
    for i in gaps[:3]:
        i = int(i)
        out.append(
            _flag(
                "AIS_GAP", "WARN",
                f"silent for {dt[i] / 60.0:.0f} minutes",
                at=datetime.fromtimestamp(tr.t[i], tz=timezone.utc),
                lat=float(tr.lat[i]), lon=float(tr.lon[i]),
            )
        )

    return out


def duplicate_mmsi_flags(proj: Projector, tracks: dict[int, Track]) -> dict[int, list[dict]]:
    """Same MMSI broadcasting from two places at once.

    Only detectable across tracks, so it cannot live in the per-vessel pass. With
    one row per MMSI in this schema it fires when a single track contains two
    fixes at the same instant far apart.
    """
    out: dict[int, list[dict]] = {}
    for mmsi, tr in tracks.items():
        if len(tr.t) < 2:
            continue
        same_t = np.flatnonzero(np.diff(tr.t) <= 0.0)
        if len(same_t):
            i = int(same_t[0])
            x, y = proj.to_m(tr.lon[i : i + 2], tr.lat[i : i + 2])
            d_km = math.hypot(x[1] - x[0], y[1] - y[0]) / 1000.0
            if d_km > 1.0:
                out.setdefault(mmsi, []).append(
                    _flag(
                        "DUPLICATE_MMSI", "CRITICAL",
                        f"two positions {d_km:.1f} km apart at the same timestamp",
                    )
                )
    return out


# --------------------------------------------------------------------------
# Scoring and classification
# --------------------------------------------------------------------------

_SEVERITY_COST = {"INFO": 0.03, "WARN": 0.12, "CRITICAL": 0.35}


def classify(tflags: list[dict], radar_matched: bool | None) -> str:
    codes = {f["code"] for f in tflags}
    if codes & {"MID_FLAG_MISMATCH", "DUPLICATE_MMSI", "POSITION_TELEPORT"}:
        return "IDENTITY_MISMATCH"
    if radar_matched is False:
        return "PHANTOM"
    return "MATCHED"


def score_tracks(
    tracks: dict[int, Track],
    cfg: dict,
    proj: Projector,
    radar_matched: dict[int, bool] | None = None,
) -> dict[int, dict]:
    dupes = duplicate_mmsi_flags(proj, tracks)
    out: dict[int, dict] = {}
    for mmsi, tr in tracks.items():
        tf = trust_flags(proj, tr, cfg) + dupes.get(mmsi, [])
        bf = behaviour_flags(proj, tr, cfg)
        penalty = sum(_SEVERITY_COST[f["severity"]] for f in tf)
        out[mmsi] = {
            "mmsi": mmsi,
            "score": float(max(0.0, min(1.0, 1.0 - penalty))),
            "classification": classify(
                tf, (radar_matched or {}).get(mmsi) if radar_matched else None
            ),
            "trust_flags": tf,
            "behaviour_flags": bf,
        }
    return out


def priors_from(scores: dict[int, dict], cfg: dict) -> dict[int, dict]:
    """Turn flags into the multipliers section 4.5.4 applies before the softmax."""
    tp = cfg["priors"]["trust"]
    bp = cfg["priors"]["behaviour"]
    code_to_prior = {
        "LOITERING": bp["loitering"],
        "SPEED_REDUCTION": bp["unexplained_slowdown"],
        "COURSE_DEVIATION": bp["course_deviation"],
        "NIGHT_MANOEUVRING": bp["night_manoeuvring"],
    }

    out: dict[int, dict] = {}
    for mmsi, s in scores.items():
        codes = {f["code"] for f in s["trust_flags"]}
        sev = {f["severity"] for f in s["trust_flags"]}

        if s["classification"] == "IDENTITY_MISMATCH":
            # Spoofing is evidence of intent, not innocence.
            trust_prior = tp["identity_mismatch"]
        elif "CRITICAL" in sev:
            trust_prior = tp["major_flags"]
        elif "WARN" in sev:
            trust_prior = tp["minor_flags"]
        else:
            trust_prior = tp["clean"]

        behaviour_prior = 1.0
        for f in s["behaviour_flags"]:
            behaviour_prior *= code_to_prior.get(f["code"], 1.0)
        if "AIS_GAP" in codes:
            behaviour_prior *= bp["ais_gap_in_aoi"]

        out[mmsi] = {
            "trust_prior": float(trust_prior),
            "behaviour_prior": float(behaviour_prior),
            "proximity_prior": 1.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score AIS trust and behaviour.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    a = ap.parse_args()

    d = Path(a.root) / a.incident
    cfg = yaml.safe_load(open("config/weights.yaml"))
    tracks = load_tracks(d / "ais.parquet")

    lons = np.concatenate([t.lon for t in tracks.values()])
    lats = np.concatenate([t.lat for t in tracks.values()])
    proj = Projector(float(lats.mean()), float(lons.mean()))

    scores = score_tracks(tracks, cfg, proj)
    priors = priors_from(scores, cfg)
    for mmsi, s in scores.items():
        s["priors"] = priors[mmsi]
    (d / "trust.json").write_text(json.dumps(scores, indent=2))

    flagged = {m: s for m, s in scores.items() if s["trust_flags"] or s["behaviour_flags"]}
    print(f"scored {len(scores)} vessels, {len(flagged)} carry at least one flag\n")
    for m, s in sorted(flagged.items(), key=lambda kv: kv[1]["score"]):
        p = priors[m]
        codes = [f"{f['code']}({f['severity'][0]})" for f in s["trust_flags"]]
        bcodes = [f["code"] for f in s["behaviour_flags"]]
        print(f"  {m}  trust={s['score']:.2f}  {s['classification']:<18} "
              f"prior t={p['trust_prior']:.2f} b={p['behaviour_prior']:.2f}")
        if codes:
            print(f"      trust    : {', '.join(codes)}")
        if bcodes:
            print(f"      behaviour: {', '.join(bcodes)}")
    print(f"\nwritten: {d / 'trust.json'}")


if __name__ == "__main__":
    main()
