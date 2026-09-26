"""Dissemination and alerting — CLAUDE.md layer 11, section 5.8.

Turns a finished incident into a severity-tiered alert addressed to the Indian
Coast Guard units that would actually respond, plus a GeoJSON/REST export for
response agencies.

Two things this module deliberately does NOT do:

* It does not transmit. It writes `alert.json` and serves it; the dashboard shows
  the exact payload and message that *would* go out. Wiring a real transport is a
  deployment decision with real recipients attached, not something a demo should
  own. `transmitted` is always false and the payload says so.
* It does not re-run any analysis. Like `api.py`, it reads what the pipeline
  already wrote, so the alert, the dashboard and the dossier cannot diverge.

Severity follows NOS-DCP tiering (Tier I local / II regional / III national)
rather than an invented scale, because that is the plan the Coast Guard actually
operates under and the tier determines who is on the distribution list.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

STATIONS_FILE = Path(__file__).with_name("stations.json")
CONFIG = Path("config/weights.yaml")

# Fallback thresholds if config/weights.yaml carries no dissemination block.
DEFAULT_TIERING = {
    "tier2_area_km2": 25.0,
    "tier3_area_km2": 150.0,
    "tier3_shore_eta_hours": 24.0,
    "tier2_shore_eta_hours": 72.0,
    "max_recipients": 6,
}


def _tiering() -> dict[str, float]:
    cfg = dict(DEFAULT_TIERING)
    if CONFIG.exists():
        loaded = yaml.safe_load(CONFIG.read_text()) or {}
        cfg.update(loaded.get("dissemination", {}) or {})
    return cfg


# ---------------------------------------------------------------- geometry

R_EARTH_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees true."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


_COMPASS = (
    "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
)


def compass(deg: float) -> str:
    return _COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def dms(value: float, is_lat: bool) -> str:
    """Position in the deg-min form a maritime message uses."""
    hemi = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    v = abs(value)
    d = int(v)
    m = (v - d) * 60
    return f"{d:02d}°{m:05.2f}'{hemi}"


# ---------------------------------------------------------------- stations


def load_stations() -> dict[str, Any]:
    return json.loads(STATIONS_FILE.read_text(encoding="utf-8"))


def nearest_stations(
    lat: float, lon: float, tier: int, limit: int, stations: list[dict]
) -> list[dict]:
    """Distribution list for a tier, nearest first within each eligible role.

    A tier does not simply widen the radius — it adds *levels of command*. Tier I
    goes to the nearest local units; Tier II adds the regional HQ and the MRCC;
    Tier III adds Coast Guard HQ. Sorting purely by distance would put Gandhinagar
    ahead of a closer station and bury the unit that actually sails.
    """
    scored = []
    for s in stations:
        if s.get("nosdcp_tier_min", 1) > tier:
            continue
        d = haversine_km(lat, lon, s["lat"], s["lon"])
        scored.append({**s, "distance_km": round(d, 1),
                       "bearing_deg": round(bearing_deg(s["lat"], s["lon"], lat, lon), 1)})

    rank = {"NATIONAL_HQ": 0, "REGION_HQ": 1, "DISTRICT_HQ": 2, "STATION": 3}
    local = sorted((s for s in scored if s["role"] in ("DISTRICT_HQ", "STATION")),
                   key=lambda s: s["distance_km"])
    command = sorted((s for s in scored if s["role"] in ("REGION_HQ", "NATIONAL_HQ")),
                     key=lambda s: (rank[s["role"]], s["distance_km"]))

    # The regional HQ that owns the water matters more than the nearest one by
    # straight-line distance, so keep the region of the closest local unit first.
    if local and command:
        owning = local[0].get("region")
        command.sort(key=lambda s: (s["region"] != owning, rank[s["role"]], s["distance_km"]))

    out: list[dict] = []
    for s in local[: max(1, limit - len(command[:2]))]:
        out.append({**s, "action": "RESPONSE — nearest unit, pollution response team"})
    for s in command[:2]:
        act = ("COORDINATION — MRCC / regional response coordination"
               if s["role"] == "REGION_HQ" else
               "NOTIFICATION — national coordinating authority, NOS-DCP Tier III")
        out.append({**s, "action": act})
    return out[:limit]


# ---------------------------------------------------------------- severity


def classify(inc: dict, forecasts: dict, cfg: dict) -> dict[str, Any]:
    """NOS-DCP tier from spill size and time-to-shore.

    Both inputs come from the pipeline: area from the detection polygon, shore
    contact from `impact/forecast.py`. Nothing here is a new estimate.
    """
    area = float(inc["observed_slick"]["area_km2"])

    shore_eta_h: float | None = None
    shore_contact = False
    affected_km = 0.0
    nearest_coast_km: float | None = None
    acq = datetime.fromisoformat(inc["acquisition_at"])

    for h in (24, 48, 72):
        fc = forecasts.get(f"forecast_{h}h")
        if not fc:
            continue
        for f in fc.get("features", []):
            p = f.get("properties", {})
            if p.get("kind") != "forecast":
                continue
            if p.get("distance_to_coast_km") is not None:
                d = float(p["distance_to_coast_km"])
                nearest_coast_km = d if nearest_coast_km is None else min(nearest_coast_km, d)
            if p.get("coastline_intersects") and not shore_contact:
                shore_contact = True
                affected_km = float(p.get("affected_shoreline_km") or 0.0)
                eta = p.get("coastline_eta")
                shore_eta_h = (
                    (datetime.fromisoformat(eta) - acq).total_seconds() / 3600.0
                    if eta else float(h)
                )

    tier = 1
    reasons = []
    if area >= cfg["tier2_area_km2"]:
        tier = max(tier, 2)
        reasons.append(f"slick area {area:.1f} km² ≥ {cfg['tier2_area_km2']:g} km²")
    if area >= cfg["tier3_area_km2"]:
        tier = max(tier, 3)
        reasons.append(f"slick area {area:.1f} km² ≥ {cfg['tier3_area_km2']:g} km²")
    if shore_contact and shore_eta_h is not None:
        if shore_eta_h <= cfg["tier3_shore_eta_hours"]:
            tier = max(tier, 3)
            reasons.append(f"shoreline contact forecast in {shore_eta_h:.0f} h")
        elif shore_eta_h <= cfg["tier2_shore_eta_hours"]:
            tier = max(tier, 2)
            reasons.append(f"shoreline contact forecast in {shore_eta_h:.0f} h")
    if not reasons:
        reasons.append(
            f"slick area {area:.1f} km², no shoreline contact within 72 h"
            + (f" (closest approach {nearest_coast_km:.0f} km)" if nearest_coast_km else "")
        )

    severity = {1: "ADVISORY", 2: "ALERT", 3: "EMERGENCY"}[tier]
    return {
        "nosdcp_tier": tier,
        "tier_label": {1: "TIER I — local", 2: "TIER II — regional",
                       3: "TIER III — national"}[tier],
        "severity": severity,
        "reasons": reasons,
        "shore_contact": shore_contact,
        "shore_eta_hours": shore_eta_h,
        "affected_shoreline_km": affected_km,
        "nearest_coast_km": nearest_coast_km,
    }


# ---------------------------------------------------------------- message


def compose_message(inc: dict, cls: dict, recipients: list[dict], suspect: dict | None) -> str:
    slick = inc["observed_slick"]
    lat, lon = slick["centroid_lat"], slick["centroid_lon"]
    env = inc["env_summary"]
    age = inc.get("slick_age") or {}
    acq = datetime.fromisoformat(inc["acquisition_at"])
    to = ", ".join(r["name"] for r in recipients) or "(no unit within range)"

    lines = [
        f"PRIORITY {cls['severity']}  //  MARPOL ANNEX I — SUSPECTED OPERATIONAL DISCHARGE",
        f"NOS-DCP {cls['tier_label'].upper()}",
        "",
        f"TO   : {to}",
        f"FROM : SAMUDRA attribution engine — incident {inc['incident_id']}",
        f"DTG  : {datetime.now(timezone.utc).strftime('%d%H%MZ %b %Y').upper()}",
        "",
        "1. POSITION AND EXTENT",
        f"   Slick centre {dms(lat, True)} {dms(lon, False)}",
        f"   Area {slick['area_km2']:.1f} km2, long axis {slick['major_axis_deg']:.0f} deg true,"
        f" eccentricity {slick['eccentricity']:.3f}",
        f"   Detected on SAR scene acquired {acq.strftime('%d %b %Y %H%M')}Z",
        "",
        "2. CONDITIONS AT ACQUISITION",
        f"   Wind {env['mean_wind_speed_ms']:.1f} m/s from {env['mean_wind_dir_deg']:.0f} deg"
        f" — detection wind gate {'PASS' if env['wind_gate_pass'] else 'FAIL'}",
        f"   Surface current {env['mean_current_speed_ms']:.2f} m/s setting"
        f" {env['mean_current_dir_deg']:.0f} deg",
    ]
    if age:
        lines.append(
            f"   Estimated slick age {age.get('best_hours', 0):.1f} h"
            f" ({age.get('low_hours', 0):.1f}–{age.get('high_hours', 0):.1f} h)"
        )
    lines += ["", "3. FORECAST"]
    if cls["shore_contact"]:
        lines.append(
            f"   SHORELINE CONTACT FORECAST in {cls['shore_eta_hours']:.0f} h,"
            f" approx {cls['affected_shoreline_km']:.1f} km of coast affected."
        )
    else:
        near = cls["nearest_coast_km"]
        lines.append(
            "   No shoreline contact within 72 h."
            + (f" Closest approach {near:.0f} km." if near is not None else "")
        )
    lines += ["", "4. ATTRIBUTION"]
    if suspect:
        lines += [
            f"   Highest-ranked candidate: {suspect.get('vessel_name') or 'unknown'}"
            f" (MMSI {suspect['mmsi']}, flag {suspect.get('flag') or '--'},"
            f" {suspect.get('vessel_type') or 'type unknown'})",
            f"   Posterior {suspect['posterior']:.3f} over"
            f" {len(inc['suspects'])} ranked candidates.",
            "   THIS IS A RANKED LIKELIHOOD, NOT AN ACCUSATION. Full reasoning,"
            " hypothesis geometry and",
            "   hash-chained evidence are in the attached dossier.",
        ]
    else:
        lines.append("   No candidate vessel survived pruning.")
    lines += [
        "",
        "5. REQUESTED ACTION",
        "   Nearest unit: verify by surface or air sortie, photograph and sample"
        " if slick is confirmed.",
        "   Regional HQ: assess response tier and mobilise pollution response"
        " equipment as required.",
        "",
        f"ATTACHMENT: dossier_{inc['incident_id']}.pdf (evidence dossier,"
        " SHA-256 hash-chained)",
        "",
        "// SAMUDRA is a decision-support system. Operational action remains the"
        " responsibility of the",
        "// coordinating authority. //",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- build


def build(
    incident_id: str, root: Path = Path("artifacts"), write: bool = True
) -> dict[str, Any]:
    """Assemble the alert for an analysed incident.

    Writes artifacts/<id>/alert.json unless `write` is False. The dossier passes
    write=False: a report that mutates the directory it is attesting to is the
    same circularity the audit log excludes dossier PDFs for.
    """
    d = root / incident_id
    inc = json.loads((d / "incident.json").read_text())

    forecasts = {}
    for h in (24, 48, 72):
        p = d / f"forecast_{h}h.geojson"
        if p.exists():
            forecasts[f"forecast_{h}h"] = json.loads(p.read_text())

    cfg = _tiering()
    cls = classify(inc, forecasts, cfg)

    slick = inc["observed_slick"]
    lat, lon = slick["centroid_lat"], slick["centroid_lon"]
    data = load_stations()
    recipients = nearest_stations(
        lat, lon, cls["nosdcp_tier"], int(cfg["max_recipients"]), data["stations"]
    )

    suspect = inc["suspects"][0] if inc.get("suspects") else None
    issued = datetime.now(timezone.utc)
    acq = datetime.fromisoformat(inc["acquisition_at"])

    alert = {
        "alert_id": f"SAMUDRA-{incident_id}-{acq.strftime('%Y%m%dT%H%MZ')}",
        "incident_id": incident_id,
        "issued_at": issued.isoformat(),
        "acquisition_at": inc["acquisition_at"],
        "severity": cls["severity"],
        "nosdcp_tier": cls["nosdcp_tier"],
        "tier_label": cls["tier_label"],
        "tier_reasons": cls["reasons"],
        "position": {"lat": lat, "lon": lon,
                     "dms": f"{dms(lat, True)} {dms(lon, False)}"},
        "slick": {
            "area_km2": slick["area_km2"],
            "major_axis_deg": slick["major_axis_deg"],
            "age_hours": (inc.get("slick_age") or {}).get("best_hours"),
        },
        "shore": {
            "contact_forecast": cls["shore_contact"],
            "eta_hours": cls["shore_eta_hours"],
            "affected_shoreline_km": cls["affected_shoreline_km"],
            "closest_approach_km": cls["nearest_coast_km"],
        },
        "top_suspect": None if not suspect else {
            "mmsi": suspect["mmsi"],
            "vessel_name": suspect.get("vessel_name"),
            "vessel_type": suspect.get("vessel_type"),
            "flag": suspect.get("flag"),
            "posterior": suspect["posterior"],
            "classification": suspect.get("classification"),
        },
        "recipients": recipients,
        "message": compose_message(inc, cls, recipients, suspect),
        "attachment": f"dossier_{incident_id}.pdf",
        "legal_basis": "MARPOL 73/78 Annex I; Merchant Shipping Act 1958 Part XB; "
                       "response under NOS-DCP with the Indian Coast Guard as Central "
                       "Coordinating Authority.",
        "transmitted": False,
        "transmission_note": (
            "SAMUDRA composes and addresses the alert but does not transmit it. No "
            "message has left this machine. Wiring a transport requires real, verified "
            "recipient addresses and an authority to send, which is a deployment "
            "decision outside this system."
        ),
        "station_source": data["source"],
    }
    if write:
        (d / "alert.json").write_text(json.dumps(alert, indent=2), encoding="utf-8")
    return alert


def stations_geojson(alert: dict | None = None) -> dict[str, Any]:
    """Station layer for the dashboard and for the agency GeoJSON export."""
    data = load_stations()
    addressed = {r["id"] for r in (alert or {}).get("recipients", [])}
    by_id = {r["id"]: r for r in (alert or {}).get("recipients", [])}
    feats = []
    for s in data["stations"]:
        r = by_id.get(s["id"])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "properties": {
                **{k: v for k, v in s.items() if k not in ("lat", "lon")},
                "addressed": s["id"] in addressed,
                "distance_km": r["distance_km"] if r else None,
                "bearing_deg": r["bearing_deg"] if r else None,
                "action": r["action"] if r else None,
            },
        })
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"source": data["source"], "authority": data["authority"]}}
