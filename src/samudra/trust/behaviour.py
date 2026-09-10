"""Behavioural analysis - CLAUDE.md section 4.4 (layer 5).

Separate from score.py on purpose. Trust asks whether the AIS *report* can be
believed; behaviour asks whether the *vessel* was acting like one about to
discharge. A vessel can have impeccable data and damning behaviour, or the
reverse, and collapsing the two loses exactly the distinction an investigator
needs.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from samudra.attribution.prune import Track
from samudra.geo import Projector

# A vessel is "under way" above this speed. Below it, and for less than
# MIN_UNDERWAY_FRACTION of its reports, it is berthed or at anchor.
UNDERWAY_KN = 3.0
MIN_UNDERWAY_FRACTION = 0.05


def _flag(code: str, severity: str, detail: str, at=None, lat=None, lon=None) -> dict:
    return {
        "code": code,
        "severity": severity,
        "detail": detail,
        "at": at.isoformat() if isinstance(at, datetime) else at,
        "lat": lat,
        "lon": lon,
    }


# --------------------------------------------------------------------------
# Behaviour checks
# --------------------------------------------------------------------------


def behaviour_flags(proj: Projector, tr: Track, cfg: dict) -> list[dict]:
    out: list[dict] = []
    if len(tr.t) < 12:
        return out

    # A vessel that never gets under way is MOORED, not loitering, and moored is
    # the normal state in a port. On the Houston feed the naive test flagged 444
    # of 560 vessels, which makes the behaviour prior worse than useless: it
    # promoted every berthed ship into the suspect list. Loitering is only
    # meaningful as an interruption to a passage.
    underway = tr.sog > UNDERWAY_KN
    if underway.mean() < MIN_UNDERWAY_FRACTION:
        return out

    x, y = proj.to_m(tr.lon, tr.lat)

    # Loitering: low speed with small net displacement over a window.
    # 60 minutes is the conventional loitering threshold; a 90-minute window
    # cannot see a two-hour stop reported at one-minute intervals.
    win_s = 60 * 60.0
    i = 0
    best = None
    while i < len(tr.t):
        j = int(np.searchsorted(tr.t, tr.t[i] + win_s))
        if j >= len(tr.t):
            break
        net_km = math.hypot(x[j] - x[i], y[j] - y[i]) / 1000.0
        mean_sog = float(np.mean(tr.sog[i:j])) if j > i else 99.0
        span_h = (tr.t[j] - tr.t[i]) / 3600.0
        # Require the stop to be bracketed by movement, so it is a pause in a
        # transit rather than the beginning or end of a berth period.
        moved_before = underway[:i].any()
        moved_after = underway[j:].any()
        if mean_sog < 2.0 and net_km < 3.0 and moved_before and moved_after:
            if best is None or span_h > best[0]:
                best = (span_h, mean_sog, net_km, i)
        i = j if j > i else i + 1
    if best:
        span_h, mean_sog, net_km, i = best
        out.append(
            _flag(
                "LOITERING", "WARN",
                f"{mean_sog:.1f} kn average, {net_km:.1f} km net displacement over {span_h:.1f} h",
                at=datetime.fromtimestamp(tr.t[i], tz=timezone.utc),
                lat=float(tr.lat[i]), lon=float(tr.lon[i]),
            )
        )

    # Unexplained speed reduction: sustained drop against the vessel's own norm.
    med = float(np.median(tr.sog))
    if med > 6.0:
        slow = tr.sog < med * 0.45
        runs = _runs(slow)
        for a, b in runs:
            dur_h = (tr.t[b] - tr.t[a]) / 3600.0
            if dur_h >= 1.0:
                out.append(
                    _flag(
                        "SPEED_REDUCTION", "WARN",
                        f"dropped to {float(np.mean(tr.sog[a:b])):.1f} kn from a "
                        f"{med:.1f} kn norm for {dur_h:.1f} h",
                        at=datetime.fromtimestamp(tr.t[a], tz=timezone.utc),
                        lat=float(tr.lat[a]), lon=float(tr.lon[a]),
                    )
                )
                break

    # Course deviation from the vessel's own prevailing heading.
    if len(tr.cog) > 20 and np.any(tr.cog != 0):
        rad = np.radians(tr.cog)
        prevailing = math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0
        dev = np.abs((tr.cog - prevailing + 180.0) % 360.0 - 180.0)
        big = _runs(dev > 60.0)
        for a, b in big:
            if (tr.t[b] - tr.t[a]) / 3600.0 >= 0.75:
                out.append(
                    _flag(
                        "COURSE_DEVIATION", "INFO",
                        f"held {float(np.mean(dev[a:b])):.0f} deg off its {prevailing:.0f} deg "
                        f"prevailing course for {(tr.t[b] - tr.t[a]) / 3600.0:.1f} h",
                        at=datetime.fromtimestamp(tr.t[a], tz=timezone.utc),
                    )
                )
                break

    # Night-time manoeuvring: course changes in darkness.
    hours = np.array(
        [datetime.fromtimestamp(v, tz=timezone.utc).hour for v in tr.t[:: max(1, len(tr.t) // 200)]]
    )
    night = ((hours >= 20) | (hours <= 3)).mean() if len(hours) else 0.0
    if night > 0.25 and any(f["code"] in ("LOITERING", "COURSE_DEVIATION") for f in out):
        out.append(
            _flag("NIGHT_MANOEUVRING", "INFO",
                  f"{night * 100:.0f}% of the track falls between 20:00 and 03:00 UTC")
        )

    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end) index pairs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out
