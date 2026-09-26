"""How long would OUR three-satellite architecture take to see this slick?

The honest answer to "where are we lacking", computed rather than asserted.

THE THREE SENSORS DO NOT DO THE SAME JOB, and the whole point of this module is
to show that clearly instead of implying a continuous oil-spill feed:

  INSAT-3DS   geostationary, ~48 looks/day, but 1 km visible / 4 km thermal.
              Ten pixels is the least anyone should call a detection, so it
              needs roughly 10 km2 before it can see anything at all — and it
              is optical, so it needs daylight AND a gap in the cloud.
  EOS-06      OCM-3 at 360 m, ~2 day revisit. Same daylight and cloud
              constraints; over the Arabian Sea in monsoon that is most of the
              season gone.
  EOS-04      C-band SAR at 18 m, 5-11 day revisit. The ONLY one that sees
              through cloud and at night, and the only one whose detection
              probability here is measured rather than reasoned.

WHERE THE NUMBERS COME FROM, and which are measured:

  EOS-04 detection   MEASURED. scripts/detection_limit.py plants slicks of
                     known area and contrast into real EOS-04 HH ocean and runs
                     the real checkpoint: 54 cells, 14 trials each. Loaded from
                     artifacts/detection_limit.json. If that file is absent this
                     module says so rather than substituting a guess.
  EOS-04 revisit     DERIVED from orbit geometry: 225 km swath against 37,024 km
                     of circumference at 22.5 N is 165 tracks, and at 14.8
                     orbits/day that is 11.1 days for the same geometry, ~5.6
                     for any look.
  INSAT-3DS cadence  MEASURED against the live MOSDAC catalogue: 148 SST
                     granules in 3 days over the Gulf of Kutch is ~48/day.
  Cloud, daylight    OPERATOR INPUT. This module does not know the weather and
                     does not pretend to; the caller states an assumption and it
                     is echoed back in the result.
  Slick persistence  ASSUMPTION with a stated range. A thin discharge film stays
                     SAR-detectable for roughly 12-48 h before weathering and
                     wind break the contrast. This is the least certain input
                     here and it is flagged as such.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DETECTION_LIMIT_PATH = Path("artifacts/detection_limit.json")

#: The size classes the console offers.
SIZE_CLASSES: dict[str, tuple[float, float, str]] = {
    "large": (10.0, 1e6, ">= 10 km2 — major spill or tanker casualty"),
    "medium": (5.0, 10.0, "5-10 km2 — large discharge"),
    "small": (0.0, 5.0, "< 5 km2 — routine illegal discharge, the actual target"),
}

#: Least number of pixels anyone should call a detection on. Below this a
#: "detection" is a handful of cells and indistinguishable from noise.
MIN_PIXELS_FOR_DETECTION = 10

SENSORS: dict[str, dict[str, Any]] = {
    "INSAT-3DS": {
        "instrument": "Imager (VIS 1 km / TIR 4 km)",
        "pixel_m": 1000.0,
        "looks_per_day": 48.0,          # measured: 148 SST granules / 3 days
        "revisit_hours": 0.5,
        "needs_daylight": True,
        "needs_clear_sky": True,
        "sees_through_cloud": False,
        "role": "continuous watch for large events; SST context",
    },
    "EOS-06": {
        "instrument": "OCM-3 (360 m)",
        "pixel_m": 360.0,
        "looks_per_day": 0.5,
        "revisit_hours": 48.0,
        "needs_daylight": True,
        "needs_clear_sky": True,
        "sees_through_cloud": False,
        "role": "ocean colour; biogenic look-alike rejection",
    },
    "EOS-04": {
        "instrument": "C-band SAR MRS (18 m)",
        "pixel_m": 18.0,
        "looks_per_day": 1 / (5.6 * 24),
        "revisit_hours": 5.6 * 24,      # any look; 11.1 d for same geometry
        "needs_daylight": False,
        "needs_clear_sky": False,
        "sees_through_cloud": True,
        "role": "detection, characterisation and attribution — the product",
    },
}


def _load_measured() -> dict | None:
    if not DETECTION_LIMIT_PATH.exists():
        return None
    try:
        return json.loads(DETECTION_LIMIT_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def eos04_detection_rate(area_km2: float, contrast_db: float) -> dict[str, Any]:
    """Measured EOS-04 detection probability, interpolated from the real table.

    Nearest-cell lookup rather than a fitted curve: the table is what was
    measured, and fitting a smooth surface through 54 noisy cells would invent
    confidence between them. The cell actually used is reported so the number
    can be traced back.
    """
    m = _load_measured()
    if not m:
        return {"measured": False, "detection_rate": None,
                "detail": ("artifacts/detection_limit.json absent — run "
                           "scripts/detection_limit.py. No value is guessed.")}

    cells = m["cells"]
    # Above the largest measured area the rate is taken from that largest cell,
    # not extrapolated upward: bigger is never harder, so this is a floor.
    max_area = max(c["area_km2"] for c in cells)
    a = min(float(area_km2), max_area)
    best = min(cells, key=lambda c: (abs(math.log10(max(c["area_km2"], 1e-6))
                                         - math.log10(max(a, 1e-6))),
                                     abs(c["contrast_db"] - contrast_db)))
    return {
        "measured": True,
        "detection_rate": best["detection_rate"],
        "matched_cell": {"area_km2": best["area_km2"],
                         "contrast_db": best["contrast_db"],
                         "planted_px": best["planted_px"]},
        "clamped_from_km2": float(area_km2) if area_km2 > max_area else None,
        "trials_per_cell": m.get("trials_per_cell"),
        "source": ("measured on real EOS-04 HH ocean by "
                   "scripts/detection_limit.py"),
    }


def _pixels(area_km2: float, pixel_m: float) -> float:
    return (area_km2 * 1e6) / (pixel_m * pixel_m)


def assess(
    area_km2: float,
    contrast_db: float = 8.0,
    cloud_fraction: float = 0.5,
    daylight_fraction: float = 0.5,
    persistence_hours: float = 24.0,
) -> dict[str, Any]:
    """Per-sensor verdict and an expected time to first detection.

    `cloud_fraction` and `daylight_fraction` are OPERATOR INPUTS — this module
    has no weather feed. They are echoed into the result so a reader can see
    what was assumed.
    """
    out: dict[str, Any] = {
        "area_km2": area_km2,
        "size_class": next((k for k, (lo, hi, _) in SIZE_CLASSES.items()
                            if lo <= area_km2 < hi), "large"),
        "assumptions": {
            "contrast_db": contrast_db,
            "cloud_fraction": cloud_fraction,
            "daylight_fraction": daylight_fraction,
            "persistence_hours": persistence_hours,
            "note": ("Cloud, daylight and persistence are stated assumptions, "
                     "not observations. Persistence is the least certain: a "
                     "thin film stays SAR-detectable for roughly 12-48 h."),
        },
        "sensors": [],
    }

    for name, s in SENSORS.items():
        px = _pixels(area_km2, s["pixel_m"])
        resolves = px >= MIN_PIXELS_FOR_DETECTION

        # Fraction of passes that are actually usable.
        usable = 1.0
        blockers = []
        if s["needs_clear_sky"]:
            usable *= max(0.0, 1.0 - cloud_fraction)
            blockers.append(f"cloud ({cloud_fraction:.0%})")
        if s["needs_daylight"]:
            usable *= max(0.0, daylight_fraction)
            blockers.append(f"daylight only ({daylight_fraction:.0%} of passes)")

        entry: dict[str, Any] = {
            "sensor": name,
            "instrument": s["instrument"],
            "role": s["role"],
            "pixels_on_target": round(px, 1),
            "resolves": bool(resolves),
            "revisit_hours": s["revisit_hours"],
            "usable_pass_fraction": round(usable, 3),
            "constraints": blockers or ["none — SAR sees through cloud and at night"],
        }

        if name == "EOS-04":
            meas = eos04_detection_rate(area_km2, contrast_db)
            entry["measured_detection_rate"] = meas.get("detection_rate")
            entry["measurement"] = meas
            p_detect = meas.get("detection_rate")
            if p_detect is None:
                entry["verdict"] = "UNKNOWN — no measurement on disk"
                entry["expected_hours_to_detection"] = None
                out["sensors"].append(entry)
                continue
        else:
            # Optical/thermal: no measured curve exists for these sensors here,
            # so the only honest statement is whether the slick is resolvable
            # at all. Resolution is a hard limit; it is not a probability.
            p_detect = 1.0 if resolves else 0.0
            entry["measured_detection_rate"] = None
            entry["measurement"] = {
                "measured": False,
                "detail": ("resolution limit only — no detection experiment has "
                           "been run for this sensor, so no probability is "
                           "claimed beyond whether the slick spans "
                           f"{MIN_PIXELS_FOR_DETECTION} pixels"),
            }

        if not resolves:
            entry["verdict"] = (
                f"CANNOT SEE IT — {px:.1f} px at {s['pixel_m']:.0f} m, "
                f"needs {MIN_PIXELS_FOR_DETECTION}")
            entry["expected_hours_to_detection"] = None
        else:
            # Mean wait to the next pass is half the revisit; passes that are
            # cloudy or dark are skipped, which multiplies the wait.
            mean_wait = s["revisit_hours"] / 2.0
            eff = usable * (p_detect or 0.0)
            if eff <= 0:
                entry["verdict"] = "BLOCKED — " + ", ".join(blockers)
                entry["expected_hours_to_detection"] = None
            else:
                hours = mean_wait / eff
                entry["expected_hours_to_detection"] = round(hours, 1)
                caught = hours <= persistence_hours
                entry["catches_before_it_disperses"] = bool(caught)
                # Being slower than the slick's life does NOT mean never. The
                # satellite arrives at a random point in its cycle, so the
                # chance of a usable look while the oil is still visible is the
                # ratio of the two — which is the deterrence number, and the
                # one an enforcement case actually rests on.
                p_catch = min(1.0, persistence_hours / hours)
                entry["catch_probability"] = round(p_catch, 3)
                entry["verdict"] = (
                    f"DETECTS in ~{hours:.1f} h" if caught else
                    f"MISSES MOST — ~{hours:.1f} h expected vs "
                    f"{persistence_hours:.0f} h slick life, "
                    f"{p_catch:.0%} of discharges caught")
        out["sensors"].append(entry)

    winners = [s for s in out["sensors"]
               if s.get("expected_hours_to_detection") is not None
               and s.get("catches_before_it_disperses")]
    if winners:
        first = min(winners, key=lambda s: s["expected_hours_to_detection"])
        out["first_detection"] = {
            "sensor": first["sensor"],
            "expected_hours": first["expected_hours_to_detection"],
            "expected_days": round(first["expected_hours_to_detection"] / 24.0, 2),
        }
        out["summary"] = (
            f"{first['sensor']} detects it in about "
            f"{first['expected_hours_to_detection']:.0f} h "
            f"({first['expected_hours_to_detection']/24:.1f} days).")
    else:
        out["first_detection"] = None
        out["summary"] = (
            "NOT DETECTED before it disperses. This is the gap: at this size "
            "and these conditions no sensor in the architecture gets a usable "
            "look in time.")
    return out


def compare_size_classes(contrast_db: float = 8.0, cloud_fraction: float = 0.5,
                         daylight_fraction: float = 0.5,
                         persistence_hours: float = 24.0) -> dict[str, Any]:
    """One assessment per size class — the three-way comparison for the console."""
    reps = {"large": 15.0, "medium": 7.0, "small": 1.0}
    return {
        "classes": [
            {"key": k, "label": SIZE_CLASSES[k][2], "area_km2": reps[k],
             **assess(reps[k], contrast_db, cloud_fraction,
                      daylight_fraction, persistence_hours)}
            for k in ("large", "medium", "small")
        ],
    }
