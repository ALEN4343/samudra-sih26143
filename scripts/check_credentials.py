"""Check INDIAN satellite provider credentials without ever printing them.

    python scripts/check_credentials.py

Reports, per provider: whether the variables are set, whether the host is
reachable, and — where the provider supports it — whether the credentials are
actually accepted. It prints the username masked and never the password, so the
output is safe to paste into a chat or a slide.

Exit code 0 if at least one provider can do something useful, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samudra.satellite import live  # noqa: E402


def mask(v: str | None) -> str:
    if not v:
        return "(not set)"
    if "@" in v:
        user, _, dom = v.partition("@")
        return f"{user[:2]}{'*' * max(1, len(user) - 2)}@{dom}"
    return f"{v[:2]}{'*' * max(1, len(v) - 2)}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("SAMUDRA — satellite provider credential check")
    print("=" * 70)
    print(f"system trust store in use : {live._TRUSTSTORE}")
    print()

    usable = False
    # ---- Indian providers: gated, adapters unverified ----------------------
    for cls in (live.Bhoonidhi, live.Mosdac):
        p = cls()
        if p.has_credentials:
            usable = True
        print()
        print(f"{p.name.upper()}  ({p.satellite})")
        print(f"  role                    : {p.role}")
        print(f"  instruments             : {p.instruments}")
        print(f"  {p.env_user:<24}: {mask(os.environ.get(p.env_user))}")
        print(f"  {p.env_pass:<24}: {'set' if os.environ.get(p.env_pass) else '(not set)'}")
        probe = p.probe()
        print(f"  host reachable          : {probe['reachable']}  ({probe['detail'][:60]})")
        print(f"  adapter verified        : {p.verified}  — the catalogue query has "
              f"never been run against a real account")
        if not p.has_credentials:
            print(f"  register                : {p.register_url}")

    print()
    print("-" * 70)
    if usable:
        print("Credentials present. The adapter is still UNVERIFIED against a real")
        print("account — download the product from the portal and drop it into")
        print("data/satellite/incoming/ with a sidecar .json.")
    else:
        print("No Indian provider is configured, so LIVE/NRT reports UNAVAILABLE.")
        print("That is the honest state, not a failure. Register at:")
        print("  EOS-04 SAR               https://bhoonidhi.nrsc.gov.in")
        print("  INSAT-3DS SST / EOS-06   https://www.mosdac.gov.in")
    print("No password was printed by this script.")
    sys.exit(0 if usable else 1)


if __name__ == "__main__":
    main()
