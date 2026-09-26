"""Fetch a real Natural Earth coastline into data/raw/.

    python scripts/fetch_coastline.py

Why this exists as a script rather than a note in the README: the download used
to fail on this machine and the failure was recorded as a permanent limitation.
It is not. This environment sits behind TLS interception, so Python's bundled CA
set rejects a handshake the operating system trusts. `truststore` routes
verification through the Windows certificate store and the download succeeds.

**Verification stays ON.** Nothing here passes `verify=False`, and nothing here
disables certificate checking. If the fetch fails, it reports why and leaves the
bundled west-coast fallback in place rather than reaching for an unverified
connection.

What the fallback costs, measured, is why this is worth running: without a real
coastline the bundled outline covers the Indian **west coast only**, so a
shoreline-impact ETA on the east coast, the Andamans or Lakshadweep has no coast
to intersect.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEST = Path("data/raw")

#: Most detailed first — this is the order `impact/coastline.py` prefers.
#: Natural Earth's "10m" means 1:10,000,000 and is FINER than "50m".
SOURCES = [
    ("ne_10m_coastline.geojson",
     "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
     "geojson/ne_10m_coastline.geojson"),
    ("ne_50m_coastline.geojson",
     "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
     "geojson/ne_50m_coastline.geojson"),
]

LICENCE = ("Natural Earth is public domain. https://www.naturalearthdata.com/ "
           "Vector mirror: github.com/nvkelso/natural-earth-vector")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        import truststore

        truststore.inject_into_ssl()
        trust = "system trust store (truststore)"
    except ImportError:
        trust = "certifi bundle — may fail behind TLS interception"

    try:
        import requests
    except ImportError:
        print("requests is not installed; pip install requests")
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"TLS verification : ON, via {trust}")
    print(f"destination      : {DEST.resolve()}")
    print()

    got = 0
    for name, url in SOURCES:
        out = DEST / name
        if out.exists() and out.stat().st_size > 100_000:
            print(f"{name:<32} already present ({out.stat().st_size/1e6:.2f} MB)")
            got += 1
            continue
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<32} FAILED  {type(exc).__name__}: {str(exc)[:120]}")
            continue
        out.write_bytes(r.content)
        print(f"{name:<32} OK  {len(r.content)/1e6:.2f} MB")
        got += 1

    print()
    if not got:
        print("No coastline fetched. SAMUDRA falls back to the bundled")
        print("west-coast-only outline, which cannot serve the east coast,")
        print("the Andamans or Lakshadweep. Shoreline ETA there will report")
        print("its source as the fallback rather than silently guessing.")
        return 1

    from samudra.impact.coastline import load_coastline

    c = load_coastline()
    print(f"loaded: {c['source']}")
    print(LICENCE)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
