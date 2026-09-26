"""Synthetic AIS seeded from REAL vessel positions measured in a real SAR scene.

WHAT IS REAL HERE AND WHAT IS NOT — read this before quoting anything from it.

REAL, measured from ISRO EOS-04 pixels by `vessels/cfar.py`:
    * how many vessels are present
    * where each one is, to the product's own georeferencing
    * how long each one is, from the component's major axis
    * which way each one is oriented, from the component's second moments
    * the acquisition instant those positions are true at

SYNTHETIC, invented by this module and labelled as such everywhere:
    * every identity — MMSI, name, IMO, flag, declared type
    * the track through each position: course held, speed, and history
    * which vessels broadcast at all

Why this split rather than fully synthetic traffic: the brief permits synthetic
AIS ("Real AIS if available may be used else synthetic data can be prepared")
and no open bulk AIS exists for Indian waters. But a fully invented traffic
scene tests the attribution engine against a geometry someone chose. Seeding
from CFAR means the *spatial* problem — how many candidates, how close together,
how they are distributed across the AOI — is the one the real Gulf of Kutch
actually poses. That is the part that makes pruning and ranking hard, and it is
the part that cannot be faked convincingly.

This module NEVER produces a slick, a culprit or an attribution answer. It
produces traffic. Anything claiming a vessel discharged oil comes from
`attribution/`, against a slick that some other module put there.

STOPPED VERSUS UNDERWAY IS DECLARED, NOT INFERRED — AND THAT WAS MEASURED.
The obvious idea is to read it off the real positions: an anchorage should show
as a tight cluster against scattered transiting traffic. It was tried, and on
this scene it fails. Nearest-neighbour distance across the 195 EOS-04
detections is unimodal with no gap to cut at, so any threshold just slides the
answer continuously. `_anchored_flags` therefore assigns a declared minority at
random, runs the spacing test anyway, and records in the manifest that it came
back inconclusive. The split is kept because it matters physically — a stopped
vessel cannot lay an elongated slick — but it is synthetic like the tracks, and
it says so.

CLI
---
    python -m samudra.synth.ais_from_cfar \
        --scene data/satellite/incoming/eos04_kutch_water_hh.tif \
        --acquired 2026-07-05T01:17:53.956Z \
        --out artifacts/<incident>/ais.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Maritime Identification Digits and the flag each implies. The mix is the one
#: a Gulf of Kutch approach actually carries: Indian coastal traffic alongside
#: the open-registry tonnage that calls at Vadinar, Kandla and Mundra.
MIDS: list[tuple[int, str, float]] = [
    (419, "IN", 0.34),   # India
    (352, "PA", 0.14),   # Panama
    (636, "LR", 0.12),   # Liberia
    (538, "MH", 0.10),   # Marshall Islands
    (563, "SG", 0.08),   # Singapore
    (470, "AE", 0.08),   # United Arab Emirates
    (249, "MT", 0.07),   # Malta
    (477, "HK", 0.07),   # Hong Kong
]

#: Declared type by measured length. Bands follow the tonnage that calls here:
#: crude tankers at Vadinar, bulk and container at Mundra, coastal fishing
#: throughout. `type_risk_prior` in attribution reads this field.
TYPE_BANDS: list[tuple[float, float, tuple[str, ...]]] = [
    (0.0, 50.0, ("Fishing",)),
    (50.0, 100.0, ("General Cargo", "Tug", "Supply Vessel")),
    (100.0, 160.0, ("General Cargo", "Bulk Carrier", "Chemical Tanker")),
    (160.0, 220.0, ("Bulk Carrier", "Chemical Tanker", "Crude Oil Tanker")),
    (220.0, 1e9, ("Crude Oil Tanker", "Bulk Carrier")),
]

_FIRST = ("OCEAN", "DESERT", "SAGAR", "MARITIME", "GULF", "ASIAN", "PACIFIC",
          "ARABIAN", "NORTHERN", "SOUTHERN", "EASTERN", "GREAT", "SILVER",
          "GOLDEN", "ROYAL", "STAR", "BLUE", "CRYSTAL", "GLOBAL", "PIONEER")
_SECOND = ("TRADER", "PIONEER", "VOYAGER", "MARINER", "CARRIER", "EXPRESS",
           "HORIZON", "SPIRIT", "MERCHANT", "ENDEAVOUR", "PROSPERITY",
           "HARMONY", "TRIUMPH", "LEGACY", "GUARDIAN", "SENTINEL")

#: SAR and AIS never agree exactly: the AIS report nearest an acquisition is
#: seconds to minutes away, and the antenna is not at the radar centroid. A
#: perfect coincidence would make SAR/AIS matching trivially easy and would
#: flatter the matcher, so a realistic offset is applied.
AIS_SAR_OFFSET_M = 90.0


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _pick_type(length_m: float, rng: np.random.Generator) -> str:
    for lo, hi, names in TYPE_BANDS:
        if lo <= length_m < hi:
            return str(rng.choice(names))
    return "General Cargo"


def _make_identity(rng: np.random.Generator, length_m: float,
                   used: set[int]) -> dict[str, Any]:
    """An invented but internally consistent identity.

    Consistent matters: `trust/score.py` flags an MMSI whose MID disagrees with
    the declared flag as IDENTITY_MISMATCH. If identities here were incoherent
    by accident, every vessel would trip that check and the flag would carry no
    information. Mismatches must be planted deliberately, not leak in.
    """
    mids = [m for m, _, _ in MIDS]
    probs = np.array([p for _, _, p in MIDS], dtype=float)
    probs /= probs.sum()
    idx = int(rng.choice(len(mids), p=probs))
    mid, flag = mids[idx], MIDS[idx][1]

    while True:
        mmsi = int(mid) * 1_000_000 + int(rng.integers(100_000, 999_999))
        if mmsi not in used:
            used.add(mmsi)
            break

    name = f"{rng.choice(_FIRST)} {rng.choice(_SECOND)}"
    return {
        "mmsi": mmsi,
        "vessel_name": name,
        "imo": float(rng.integers(9_000_000, 9_999_999)),
        "vessel_type": _pick_type(length_m, rng),
        "length_m": float(round(length_m)),
        "flag": flag,
    }


def nearest_neighbour_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Distance from each vessel to its closest neighbour."""
    n = len(lats)
    out = np.empty(n, dtype=float)
    for i in range(n):
        d = _haversine_km(lats[i], lons[i], lats, lons)
        d[i] = np.inf
        out[i] = d.min()
    return out


def _anchored_flags(lats: np.ndarray, lons: np.ndarray,
                    rng: np.random.Generator,
                    declared_fraction: float = 0.15
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Which vessels are treated as stopped, and an honest account of why.

    The first design inferred this from spacing, on the theory that an
    anchorage shows up as a tight cluster against scattered transiting traffic.
    **Measured on the real scene, it does not.** Nearest-neighbour distance over
    the 195 EOS-04 detections is unimodal — p25 0.58 km, p50 1.04 km, p95
    2.09 km, max 2.74 km — with no gap to cut at. Every candidate threshold
    just slides the split continuously (2.0 km gives 181 anchored, 1.0 km gives
    96, 0.4 km gives 16), which is the signature of a decision the data is not
    making.

    So it is not made from the data. A declared minority is stopped, chosen at
    random, and this function reports that the spacing test was run and came
    back inconclusive. Fabricating a data-driven classification here would have
    been invisible in the output and wrong in the dossier.

    The split still matters physically — a stopped vessel cannot lay an
    elongated slick, so it scores differently in attribution — which is why it
    exists at all rather than being dropped.
    """
    nn = nearest_neighbour_km(lats, lons)
    q = np.percentile(nn, [25, 50, 95])
    # Bimodal spacing would show as a wide spread between quartile and tail.
    bimodal = bool(q[2] > 4.0 * q[0])

    n = len(lats)
    k = int(round(declared_fraction * n))
    flags = np.zeros(n, dtype=bool)
    flags[rng.permutation(n)[:k]] = True

    return flags, {
        "method": "declared fraction, assigned at random",
        "declared_fraction": declared_fraction,
        "stopped": int(flags.sum()),
        "underway": int((~flags).sum()),
        "spacing_test": {
            "ran": True,
            "nn_km_p25": round(float(q[0]), 2),
            "nn_km_p50": round(float(q[1]), 2),
            "nn_km_p95": round(float(q[2]), 2),
            "nn_km_max": round(float(nn.max()), 2),
            "bimodal": bimodal,
            "conclusion": (
                "Spacing is bimodal; an anchorage is distinguishable."
                if bimodal else
                "Spacing is unimodal — the real positions do NOT separate "
                "anchored from underway traffic, so this split is declared, "
                "not inferred. It is synthetic like the rest of the tracks."),
        },
    }


def _track(lat0: float, lon0: float, bearing_deg: float, sog_kn: float,
           t0: datetime, times: np.ndarray, anchored: bool,
           rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One vessel's track, constrained to pass through (lat0, lon0) at t0.

    The constraint is what makes this worth doing: at the acquisition instant
    the synthetic vessel is where CFAR actually measured a vessel, so SAR/AIS
    matching is being tested against real geometry.

    `times` must be tz-naive UTC datetime64. A tz-aware DatetimeIndex converts
    to object dtype, which silently fails to subtract — the caller strips the
    zone once rather than each track paying for it.
    """
    dt_h = (times - np.datetime64(t0.replace(tzinfo=None))) / np.timedelta64(1, "h")

    if anchored:
        # Swinging at anchor: a slow bounded wander about the moored position,
        # not a random walk, which would drift away over a 14-hour window.
        phase = rng.uniform(0, 2 * math.pi)
        swing_km = rng.uniform(0.05, 0.25)
        ang = phase + 2 * math.pi * dt_h / rng.uniform(4.0, 9.0)
        dx = swing_km * np.cos(ang) - swing_km * math.cos(phase)
        dy = swing_km * np.sin(ang) - swing_km * math.sin(phase)
        sog = np.abs(rng.normal(0.2, 0.15, size=len(dt_h)))
        cog = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        heading = (bearing_deg + rng.normal(0, 6, size=len(dt_h))) % 360
    else:
        # Underway: a steady course with slow, correlated wander, so derived
        # speed and reported SOG agree the way a real track's do.
        drift = np.cumsum(rng.normal(0, 0.12, size=len(dt_h)))
        course = bearing_deg + drift
        sog = np.clip(sog_kn + np.cumsum(rng.normal(0, 0.04, size=len(dt_h))),
                      1.0, 22.0)
        # Integrate along the course, then shift so t0 lands on the CFAR fix.
        step_h = np.gradient(dt_h)
        km = sog * 1.852 * step_h
        dx = np.cumsum(km * np.sin(np.radians(course)))
        dy = np.cumsum(km * np.cos(np.radians(course)))
        i0 = int(np.argmin(np.abs(dt_h)))
        dx, dy = dx - dx[i0], dy - dy[i0]
        cog = course % 360
        heading = (cog + rng.normal(0, 2.5, size=len(dt_h))) % 360

    lat = lat0 + dy / 110.574
    lon = lon0 + dx / (111.320 * math.cos(math.radians(lat0)))
    return {"lat": lat, "lon": lon, "sog": sog, "cog": cog, "heading": heading}


def seed(
    targets: list[dict],
    acquired_at: datetime,
    *,
    broadcast_fraction: float = 0.78,
    phantom_count: int = 4,
    hours_before: float = 12.0,
    hours_after: float = 2.0,
    step_seconds: int = 60,
    seed_value: int = 20260705,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build an AIS table from CFAR targets. Returns (dataframe, manifest).

    `broadcast_fraction` below 1.0 is deliberate. Every CFAR target that is not
    given a track becomes a DARK vessel under CLAUDE.md 5.3 — a real radar
    return with no AIS to explain it, which is the single strongest adversarial
    signal the trust layer has. A scene where every ship broadcasts would never
    exercise it. `phantom_count` does the converse: AIS positions inside the
    footprint with no radar return.
    """
    rng = np.random.default_rng(seed_value)
    if not targets:
        raise ValueError("no CFAR targets to seed from")

    acquired_at = acquired_at.astimezone(timezone.utc)
    t_start = acquired_at - timedelta(hours=hours_before)
    t_end = acquired_at + timedelta(hours=hours_after)
    times = pd.date_range(t_start, t_end, freq=f"{step_seconds}s", tz="UTC")
    tnp = times.tz_localize(None).to_numpy()

    lats = np.array([t["lat"] for t in targets], dtype=float)
    lons = np.array([t["lon"] for t in targets], dtype=float)
    anchored, motion_report = _anchored_flags(lats, lons, rng)

    order = rng.permutation(len(targets))
    n_broadcast = int(round(broadcast_fraction * len(targets)))
    broadcasting = set(order[:n_broadcast].tolist())

    used_mmsi: set[int] = set()
    frames: list[pd.DataFrame] = []
    dark: list[dict] = []
    seeded: list[dict] = []

    for i, tg in enumerate(targets):
        length_m = float(tg["est_length_m"])
        if i not in broadcasting:
            dark.append({"det_id": tg["det_id"], "lat": tg["lat"],
                         "lon": tg["lon"], "est_length_m": round(length_m, 1),
                         "anchored": bool(anchored[i])})
            continue

        ident = _make_identity(rng, length_m, used_mmsi)
        # Offset the AIS fix from the radar centroid, as a real pair differs.
        brg = rng.uniform(0, 360)
        off_km = abs(rng.normal(0, AIS_SAR_OFFSET_M / 1000.0))
        lat0 = tg["lat"] + off_km * math.cos(math.radians(brg)) / 110.574
        lon0 = (tg["lon"] + off_km * math.sin(math.radians(brg))
                / (111.320 * math.cos(math.radians(tg["lat"]))))

        if anchored[i]:
            sog_kn = 0.0
        elif length_m >= 200:
            sog_kn = float(rng.uniform(10.5, 14.5))
        elif length_m >= 100:
            sog_kn = float(rng.uniform(9.0, 14.0))
        else:
            sog_kn = float(rng.uniform(6.0, 11.0))

        tr = _track(lat0, lon0, float(tg.get("bearing_deg", 0.0)), sog_kn,
                    acquired_at, tnp, bool(anchored[i]), rng)

        df = pd.DataFrame({
            "mmsi": ident["mmsi"], "t": times,
            "lat": tr["lat"], "lon": tr["lon"], "sog": tr["sog"],
            "cog": tr["cog"], "heading": tr["heading"],
            "vessel_name": ident["vessel_name"], "imo": ident["imo"],
            "vessel_type": ident["vessel_type"],
            "length_m": ident["length_m"], "flag": ident["flag"],
        })
        frames.append(df)
        seeded.append({**ident, "det_id": tg["det_id"],
                       "anchored": bool(anchored[i]),
                       "sar_lat": tg["lat"], "sar_lon": tg["lon"],
                       "sar_length_m": round(length_m, 1)})

    # PHANTOMs: AIS inside the footprint where the radar saw nothing. They must
    # clear the 2 km matching gate in 5.3, or they would pair with a genuine
    # ship and never be classified PHANTOM at all.
    #
    # The clearance cannot be much larger than the gate: this scene's traffic is
    # dense enough that the largest gap between real detections is 2.74 km, so a
    # 3 km rule placed zero phantoms and silently produced none. Pick the
    # emptiest candidate rather than the first acceptable one, and report the
    # clearance actually achieved so a too-crowded scene is visible instead of
    # quietly dropping the PHANTOM case.
    phantom_clearance_km = 2.4
    phantoms: list[dict] = []
    lo_lat, hi_lat = float(lats.min()), float(lats.max())
    lo_lon, hi_lon = float(lons.min()), float(lons.max())
    placed = np.empty((0, 2), dtype=float)
    tries = 0
    while len(phantoms) < phantom_count and tries < 4000:
        tries += 1
        plat = float(rng.uniform(lo_lat, hi_lat))
        plon = float(rng.uniform(lo_lon, hi_lon))
        clear = float(_haversine_km(plat, plon, lats, lons).min())
        if clear < phantom_clearance_km:
            continue
        # And not on top of another phantom.
        if len(placed) and _haversine_km(plat, plon, placed[:, 0],
                                         placed[:, 1]).min() < 2.0:
            continue
        placed = np.vstack([placed, [plat, plon]])
        length_m = float(rng.uniform(80, 190))
        ident = _make_identity(rng, length_m, used_mmsi)
        tr = _track(plat, plon, float(rng.uniform(0, 360)),
                    float(rng.uniform(8, 13)), acquired_at, tnp, False, rng)
        frames.append(pd.DataFrame({
            "mmsi": ident["mmsi"], "t": times,
            "lat": tr["lat"], "lon": tr["lon"], "sog": tr["sog"],
            "cog": tr["cog"], "heading": tr["heading"],
            "vessel_name": ident["vessel_name"], "imo": ident["imo"],
            "vessel_type": ident["vessel_type"],
            "length_m": ident["length_m"], "flag": ident["flag"],
        }))
        phantoms.append({**ident, "lat": plat, "lon": plon,
                         "clearance_km": round(clear, 2)})

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["mmsi", "t"]).reset_index(drop=True)
    out["mmsi"] = out["mmsi"].astype("int64")

    manifest = {
        "generator": "samudra.synth.ais_from_cfar",
        "seed": seed_value,
        "acquired_at": acquired_at.isoformat().replace("+00:00", "Z"),
        "window": [times[0].isoformat(), times[-1].isoformat()],
        "step_seconds": step_seconds,
        "real": {
            "source": "vessel positions, lengths and orientations measured by "
                      "samudra.vessels.cfar on real EOS-04 (ISRO) pixels",
            "cfar_targets": len(targets),
        },
        "motion": motion_report,
        "synthetic": {
            "what": ["mmsi", "vessel_name", "imo", "flag", "vessel_type",
                     "course and speed history", "which vessels broadcast"],
            "vessels_broadcasting": len(seeded),
            "dark_vessels": len(dark),
            "phantom_vessels": len(phantoms),
            "ais_sar_offset_m_sigma": AIS_SAR_OFFSET_M,
            "phantom_clearance_km": phantom_clearance_km,
            "phantoms_requested": phantom_count,
        },
        "expected_classifications": {
            "MATCHED": len(seeded),
            "DARK": len(dark),
            "PHANTOM": len(phantoms),
            "note": "Expected under CLAUDE.md 5.3's 2 km matching gate. These "
                    "are what this generator planted, not what trust/score.py "
                    "found — comparing the two is the test.",
        },
        "dark_targets": dark,
        "phantom_vessels": phantoms,
        "seeded_vessels": seeded,
        "warning": "SYNTHETIC AIS. Identities and tracks are invented. Only the "
                   "vessel positions, lengths and orientations at the "
                   "acquisition instant derive from real measurements.",
    }
    return out, manifest


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Seed synthetic AIS from real CFAR vessel detections.")
    ap.add_argument("--scene", required=True, help="calibrated SAR GeoTIFF")
    ap.add_argument("--acquired", help="ISO8601 UTC; read from the scene tags "
                                       "when the product carries one")
    ap.add_argument("--out", required=True, help="ais.parquet to write")
    ap.add_argument("--manifest", help="defaults to <out>.manifest.json")
    ap.add_argument("--broadcast-fraction", type=float, default=0.78)
    ap.add_argument("--phantoms", type=int, default=4)
    ap.add_argument("--k", type=float, default=4.5, help="CFAR threshold")
    ap.add_argument("--min-pixels", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260705)
    a = ap.parse_args()

    from samudra.vessels import cfar

    targets, summary = cfar.detect(a.scene, k=a.k, min_pixels=a.min_pixels)

    acquired = a.acquired
    if not acquired:
        import rasterio

        with rasterio.open(a.scene) as ds:
            acquired = ds.tags().get("SAMUDRA_ACQUIRED_AT")
    if not acquired:
        ap.error("--acquired is required: this scene publishes no acquisition "
                 "time, and inventing one would corrupt every drift result")
    t0 = datetime.fromisoformat(acquired.replace("Z", "+00:00"))

    df, manifest = seed(targets, t0, broadcast_fraction=a.broadcast_fraction,
                        phantom_count=a.phantoms, seed_value=a.seed)
    manifest["scene"] = summary

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    mf = Path(a.manifest) if a.manifest else out.with_suffix(".manifest.json")
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 74)
    print("SYNTHETIC AIS SEEDED FROM REAL SAR VESSEL DETECTIONS")
    print("=" * 74)
    print(f"  scene               {a.scene}")
    print(f"  acquired            {t0.isoformat()}")
    mo = manifest["motion"]
    st = mo["spacing_test"]
    print(f"  CFAR targets (real) {len(targets)}")
    print(f"  stopped / underway  {mo['stopped']} / {mo['underway']}   "
          f"({mo['method']})")
    print(f"  spacing test        nn km p25 {st['nn_km_p25']}  p50 "
          f"{st['nn_km_p50']}  p95 {st['nn_km_p95']}  max {st['nn_km_max']}  "
          f"-> bimodal {st['bimodal']}")
    print(f"    {st['conclusion']}")
    print()
    print(f"  broadcasting        {len(manifest['seeded_vessels'])}  -> expect MATCHED")
    print(f"  not broadcasting    {len(manifest['dark_targets'])}  -> expect DARK")
    print(f"  phantom tracks      {len(manifest['phantom_vessels'])}  -> expect PHANTOM")
    print(f"  rows                {len(df):,}   over "
          f"{manifest['window'][0][:16]} .. {manifest['window'][1][:16]}")
    print()
    print(f"  ais.parquet         {out}")
    print(f"  manifest            {mf}")
    print()
    print("  REAL: positions, lengths, orientations, count — measured from")
    print("        EOS-04 pixels by CFAR.")
    print("  SYNTHETIC: every identity and every track. Labelled in the")
    print("        manifest and never presented as a real vessel.")


if __name__ == "__main__":
    main()
