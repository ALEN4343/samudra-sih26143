"""CLI: python -m samudra.dissemination --incident demo-001

Reads artifacts/<id>/incident.json plus the forecast files and writes
artifacts/<id>/alert.json. Runs no analysis of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from samudra.dissemination.alert import build, stations_geojson


def main() -> None:
    # The message and the tier reasons carry degree signs and >= ; a cp1252
    # console kills the process on the first one.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Compose the coastguard alert for an incident.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", type=Path, default=Path("artifacts"))
    ap.add_argument("--print-message", action="store_true",
                    help="print the transmit-ready message text to stdout")
    ap.add_argument("--geojson", action="store_true",
                    help="also write stations.geojson for response agencies")
    a = ap.parse_args()

    d = a.root / a.incident
    if not (d / "incident.json").exists():
        raise SystemExit(
            f"{a.incident} has no incident.json — run "
            f"'python -m samudra.attribution --incident {a.incident}' first"
        )

    alert = build(a.incident, a.root)

    if a.geojson:
        p = d / "coastguard_stations.geojson"
        p.write_text(json.dumps(stations_geojson(alert), indent=2), encoding="utf-8")
        print(f"wrote {p}")

    print(f"wrote {d / 'alert.json'}")
    print(f"  {alert['alert_id']}")
    print(f"  severity      {alert['severity']}  ({alert['tier_label']})")
    for r in alert["tier_reasons"]:
        print(f"                - {r}")
    print(f"  addressed to  {len(alert['recipients'])} unit(s):")
    for r in alert["recipients"]:
        print(f"                - {r['name']}  {r['distance_km']:.0f} km  "
              f"{r['bearing_deg']:.0f}°  [{r['role']}]")
    print("  transmitted   False (compose-only; nothing left this machine)")

    if a.print_message:
        print("\n" + "-" * 72)
        print(alert["message"])
        print("-" * 72)


if __name__ == "__main__":
    main()
