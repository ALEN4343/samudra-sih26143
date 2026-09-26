"""Continuous monitoring service — CLAUDE.md layer 1, the unattended half.

    python -m samudra.monitor --interval 300
    python -m samudra.monitor --once

WHAT "CONTINUOUS" HONESTLY MEANS HERE, because this is the single easiest
thing in the project to overclaim.

The SERVICE runs continuously. The SATELLITE does not. EOS-04 passes over a
given point once every 5-11 days, so there is no stream of SAR to watch; there
is a loop that checks, finds nothing most of the time, says so, and processes
the moment something lands. That is how every operational oil-spill service
works, EMSA CleanSeaNet included. This module will never print "LIVE" over a
scene that arrived days ago, and `last_new_data_at` is separate from
`last_check_at` precisely so the difference is visible rather than blurred.

WHAT IT WATCHES, and what each source is actually for:

  drop-in folder   data/satellite/incoming/. This is the ONLY path that yields
                   new oil-spill detections. Bhoonidhi publishes no API — 404
                   on /opensearch, /api/ and /services, measured — so an EOS-04
                   product arrives because a human downloaded it. The instant
                   one appears here, the full chain runs unattended.

  MOSDAC catalogue live, unauthenticated, verified against the real service.
                   Returns INSAT-3DS SST and EOS-06 wind / ocean colour. These
                   are ENVIRONMENTAL CONTEXT, not detections: at 1-4 km
                   INSAT-3DS cannot resolve a routine 0.05-5 km2 discharge, and
                   EOS-06's optical channels need daylight and clear sky, which
                   the Arabian Sea does not offer for months at a time. They
                   feed the drift model and the 3-10 m/s detection gate. The
                   status this module reports says "environment", never
                   "detection", for exactly that reason.

So: continuous watch, intermittent imagery, honest labels on both.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MONITOR_STATUS_PATH = Path("artifacts/monitor_status.json")

#: Default AOI for the environmental poll: the Indian EEZ box.
DEFAULT_AOI = (66.0, 6.0, 94.0, 25.0)

#: MOSDAC products worth polling, and the layer each one feeds. Kept small on
#: purpose — a monitor that pulls everything is a monitor nobody reads.
ENV_DATASETS: tuple[tuple[str, str], ...] = (
    ("E06SCT_L2B_WV12", "wind vectors — drift leeway and the 3-10 m/s gate"),
    ("3SIMG_L2B_SST", "sea surface temperature — thermal-front look-alikes"),
    ("E06OCM_L2C_LAC_OC", "ocean colour — biogenic slick discrimination"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_status(path: Path | str = MONITOR_STATUS_PATH) -> dict[str, Any]:
    """Last published status, or a declared-empty one. Never invents a check."""
    p = Path(path)
    if not p.exists():
        return {
            "running": False,
            "checks": 0,
            "last_check_at": None,
            "last_new_data_at": None,
            "detail": "Monitor has never run on this machine.",
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt status is not a crash
        return {"running": False, "checks": 0, "last_check_at": None,
                "last_new_data_at": None,
                "detail": f"status file unreadable: {exc}"}


def _poll_environment(aoi, days: int, timeout: int) -> dict[str, Any]:
    """Ask MOSDAC what environmental data exists over the AOI. No credentials."""
    from samudra.satellite import live

    out: list[dict[str, Any]] = []
    m = live.Mosdac()
    start = (datetime.now(timezone.utc).date().toordinal() - days)
    t0 = datetime.fromordinal(start).strftime("%Y-%m-%d")
    t1 = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for ds, purpose in ENV_DATASETS:
        try:
            r = m.search(ds, start=t0, end=t1, bbox=tuple(aoi), count=1,
                         timeout=timeout)
            out.append({
                "dataset_id": ds, "purpose": purpose,
                "satellite": r.get("satellite"), "sensor": r.get("sensor"),
                "granules": r.get("total_results"),
                "size_mb": r.get("total_size_mb"),
                "result": "OK",
            })
        except Exception as exc:  # noqa: BLE001 - one bad dataset is not an outage
            # MOSDAC answers HTTP 500 for "no data for these parameters", which
            # is an empty result, not a fault. Reporting it as an outage would
            # make a quiet day look like a broken integration.
            msg = str(exc)
            empty = "unavailable for given parameters" in msg.lower()
            out.append({
                "dataset_id": ds, "purpose": purpose,
                "granules": 0 if empty else None,
                "result": "NO DATA" if empty else "ERROR",
                "detail": msg[:200],
            })
    return {
        "provider": "mosdac",
        "authenticated": False,
        "window_days": days,
        "aoi": list(aoi),
        "datasets": out,
        "note": ("Environmental context only. INSAT-3DS at 1-4 km and EOS-06 "
                 "optical cannot resolve a routine 0.05-5 km2 discharge; these "
                 "feed the drift model and the detection gate."),
    }


def scan_once(
    root: Path | str = "artifacts",
    aoi=DEFAULT_AOI,
    days: int = 3,
    process: bool = True,
    base_incident: str = "demo-001",
    timeout: int = 45,
    poll_environment: bool = True,
) -> dict[str, Any]:
    """One monitoring cycle: look for new products, process anything new.

    Returns a report. Never raises for an expected condition — a provider being
    unreachable, or there being nothing new, are normal outcomes of monitoring
    and must not take the service down.
    """
    from samudra.satellite import pipeline, sources

    started = time.perf_counter()
    prev = read_status()
    seen: set[str] = set(prev.get("known_observations") or [])

    report: dict[str, Any] = {
        "checked_at": _now(),
        "new_observations": [],
        "processed": [],
        "errors": [],
    }

    # --- 1. SAR products: the only source of new detections ----------------
    try:
        dropin = sources.list_dropin()
    except Exception as exc:  # noqa: BLE001
        dropin = []
        report["errors"].append({"stage": "list_dropin", "detail": str(exc)[:300]})

    current = {o.observation_id for o in dropin}
    fresh = [o for o in dropin if o.observation_id not in seen]

    for obs in fresh:
        entry = {
            "observation_id": obs.observation_id,
            "satellite": obs.satellite,
            "sensor": obs.sensor,
            "acquired_at": obs.acquired_at,
            "acquisition_time_known": obs.acquisition_time_known,
            "source": obs.source,
        }
        report["new_observations"].append(entry)
        if not process:
            continue
        try:
            out = pipeline.process(obs, base_incident=base_incident,
                                   root=Path(root))
            inf = out.get("inference", {})
            att = out.get("attribution")
            entry_done = {
                **entry,
                "incident_id": out["incident"]["incident_id"],
                "regions": len(inf.get("regions", [])),
                "anomaly_pixel_fraction": inf.get("anomaly_pixel_fraction"),
                "attributed": bool(att),
                "top_suspect": (att["suspects"][0]["mmsi"]
                                if att and att.get("suspects") else None),
                "attribution_unavailable": out.get("attribution_unavailable"),
            }
            report["processed"].append(entry_done)
        except Exception as exc:  # noqa: BLE001 - one bad product, not a crash
            report["errors"].append({
                "stage": "process", "observation_id": obs.observation_id,
                "detail": str(exc)[:300],
                "trace": traceback.format_exc(limit=3)[-600:],
            })

    # --- 2. environment: context, explicitly not detection -----------------
    if poll_environment:
        try:
            report["environment"] = _poll_environment(aoi, days, timeout)
        except Exception as exc:  # noqa: BLE001
            report["environment"] = {"provider": "mosdac", "result": "ERROR",
                                     "detail": str(exc)[:300]}

    report["known_observations"] = sorted(current | seen)
    report["elapsed_s"] = round(time.perf_counter() - started, 2)
    report["result"] = (
        "PROCESSED NEW DATA" if report["processed"] else
        "NEW DATA FOUND" if report["new_observations"] else "NO NEW DATA"
    )
    return report


def _publish(report: dict, checks: int, started_at: str, running: bool,
             last_new: str | None, path: Path) -> dict:
    """Write the status the API and UI read. Separates 'checked' from 'found'."""
    status = {
        "running": running,
        "started_at": started_at,
        "checks": checks,
        "last_check_at": report["checked_at"],
        # Kept deliberately separate from last_check_at. A service that checked
        # 30 seconds ago and last saw a satellite 6 days ago is the normal
        # state, and collapsing the two would read as a live feed.
        "last_new_data_at": last_new,
        "last_result": report["result"],
        "new_observations": report.get("new_observations", []),
        "processed": report.get("processed", []),
        "environment": report.get("environment"),
        "errors": report.get("errors", []),
        "known_observations": report.get("known_observations", []),
        "elapsed_s": report.get("elapsed_s"),
        "detail": (
            "The SERVICE runs continuously; the SATELLITE does not. EOS-04 "
            "revisit over a given point is 5-11 days, so 'NO NEW DATA' is the "
            "expected result on almost every check."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return status


def run(
    interval: float = 300.0,
    once: bool = False,
    root: Path | str = "artifacts",
    aoi=DEFAULT_AOI,
    days: int = 3,
    process: bool = True,
    base_incident: str = "demo-001",
    status_path: Path | str = MONITOR_STATUS_PATH,
    poll_environment: bool = True,
    quiet: bool = False,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    """Watch until interrupted. Returns the final status."""
    path = Path(status_path)
    started_at = _now()
    checks = 0
    last_new = read_status(path).get("last_new_data_at")
    status: dict[str, Any] = {}

    def say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    say("=" * 74)
    say("SAMUDRA MONITOR — continuous watch")
    say("=" * 74)
    say(f"  interval        {interval:g}s" + ("  (single pass)" if once else ""))
    say(f"  watching        data/satellite/incoming/  (new SAR products)")
    say(f"  environment     MOSDAC, unauthenticated, AOI {tuple(aoi)}")
    say(f"  status file     {path}")
    say("")
    say("  EOS-04 revisit is 5-11 days. 'NO NEW DATA' is the expected result")
    say("  on almost every check — the service is continuous, the satellite is")
    say("  not, and this tool will not blur the two.")
    say("")

    try:
        while True:
            checks += 1
            report = scan_once(root=root, aoi=aoi, days=days, process=process,
                               base_incident=base_incident,
                               poll_environment=poll_environment)
            if report["new_observations"]:
                last_new = report["checked_at"]
            status = _publish(report, checks, started_at, not once, last_new, path)

            say(f"[{report['checked_at']}] check {checks}: {report['result']}"
                f"  ({report['elapsed_s']}s)")
            for p in report["processed"]:
                say(f"    PROCESSED {p['observation_id']} -> {p['incident_id']}"
                    f"  {p['regions']} region(s)"
                    + (f", top suspect {p['top_suspect']}" if p["top_suspect"]
                       else ", no attribution (no AIS for this water/date)"))
            env = report.get("environment") or {}
            for d in env.get("datasets", []):
                say(f"    env  {d['dataset_id']:<22} {d['result']:<8} "
                    f"{d.get('granules')} granule(s)")
            for e in report["errors"]:
                say(f"    ERROR {e['stage']}: {e['detail'][:120]}")

            if once or (max_cycles is not None and checks >= max_cycles):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        say("\n  stopped by operator")
    finally:
        if status:
            status["running"] = False
            path.write_text(json.dumps(status, indent=2, default=str),
                            encoding="utf-8")
    return status


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="SAMUDRA continuous monitor.")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="seconds between checks (default 300)")
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    ap.add_argument("--no-process", action="store_true",
                    help="detect new products but do not run the chain")
    ap.add_argument("--no-environment", action="store_true",
                    help="skip the MOSDAC poll (offline demo)")
    ap.add_argument("--base-incident", default="demo-001",
                    help="incident supplying env field and AIS context")
    ap.add_argument("--aoi", help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--status", default=str(MONITOR_STATUS_PATH))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    aoi = tuple(float(v) for v in a.aoi.split(",")) if a.aoi else DEFAULT_AOI
    if len(aoi) != 4:
        ap.error("--aoi needs minlon,minlat,maxlon,maxlat")

    st = run(interval=a.interval, once=a.once, root=a.root, aoi=aoi,
             days=a.days, process=not a.no_process,
             base_incident=a.base_incident, status_path=a.status,
             poll_environment=not a.no_environment, quiet=a.json)
    if a.json:
        print(json.dumps(st, indent=2, default=str))


if __name__ == "__main__":
    main()
