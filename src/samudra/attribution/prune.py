"""Candidate pruning — CLAUDE.md layer 6.

Reduces every vessel in the scene to the few that could physically have been at
the origin. A vessel survives only if it was inside the reverse-advected envelope
in *space and time* together: being in the right place four hours late is not a
match, and neither is being in the right place at the wrong end of the AOI.

The funnel (total_in_scene -> in_envelope -> scored -> ranked) is a first-class
output, not a log line. "40 vessels became 6 became 1" is the argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

from samudra.timeutil import epoch_seconds


@dataclass
class Track:
    """One vessel's AIS history, in arrays for fast interpolation."""

    mmsi: int
    t: np.ndarray  # epoch seconds, ascending
    lat: np.ndarray
    lon: np.ndarray
    sog: np.ndarray
    cog: np.ndarray
    meta: dict = field(default_factory=dict)

    def position_at(self, ts: float) -> tuple[float, float] | None:
        """Linear interpolation. None if the vessel was not reporting then."""
        if ts < self.t[0] or ts > self.t[-1]:
            return None
        return (
            float(np.interp(ts, self.t, self.lat)),
            float(np.interp(ts, self.t, self.lon)),
        )

    def xyt(self) -> np.ndarray:
        """(epoch, lat, lon) — the array shape drift.advect_forward expects."""
        return np.column_stack([self.t, self.lat, self.lon])

    @property
    def name(self) -> str | None:
        return self.meta.get("vessel_name")


def load_tracks(path: Path | str) -> dict[int, Track]:
    """ais.parquet -> {mmsi: Track}. Schema is shared with the real-AIS loader."""
    df = pd.read_parquet(path)
    required = {"mmsi", "t", "lat", "lon", "sog"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.sort_values(["mmsi", "t"])
    ts = epoch_seconds(df.t)
    out: dict[int, Track] = {}
    for mmsi, idx in df.groupby("mmsi", sort=False).indices.items():
        g = df.iloc[idx]
        meta = {}
        for col in ("vessel_name", "imo", "vessel_type", "length_m", "flag"):
            if col in g.columns:
                v = g[col].iloc[0]
                meta[col] = None if pd.isna(v) else v
        out[int(mmsi)] = Track(
            mmsi=int(mmsi),
            t=ts[idx],
            lat=g.lat.to_numpy(dtype=float),
            lon=g.lon.to_numpy(dtype=float),
            sog=g.sog.to_numpy(dtype=float),
            cog=g.cog.to_numpy(dtype=float) if "cog" in g.columns else np.zeros(len(g)),
            meta=meta,
        )
    return out


@dataclass
class Candidate:
    """A vessel that was inside the origin envelope, and when."""

    mmsi: int
    track: Track
    hit_times: list[float]  # epoch seconds of the reverse steps it matched
    hit_hours_back: list[float]
    hit_sog: list[float] = field(default_factory=list)  # SOG at each hit

    @property
    def window(self) -> tuple[float, float]:
        return min(self.hit_times), max(self.hit_times)

    def underway_hits(self, min_kn: float) -> int:
        """Envelope intersections where the vessel was actually making way."""
        return sum(1 for v in self.hit_sog if v >= min_kn)


def prune(
    tracks: dict[int, Track],
    steps: list[dict],
    aoi_bounds: tuple[float, float, float, float] | None = None,
    search_window: tuple[float, float] | None = None,
    min_underway_kn: float = 1.0,
) -> tuple[list[Candidate], dict]:
    """Keep vessels intersecting the reverse-advected envelope in space and time.

    `steps` comes from drift.advect_reverse: one polygon per hour back, each with
    the time it corresponds to. A vessel matches a step only if it was inside
    *that* polygon at *that* time.
    """
    if not steps:
        raise ValueError("no reverse-advection steps supplied — cannot prune")

    # total_in_scene: reporting inside the AOI during the search window.
    lo_t, hi_t = search_window if search_window else (-np.inf, np.inf)
    in_scene = 0
    in_scene_mmsis: list[int] = []
    for tr in tracks.values():
        m = (tr.t >= lo_t) & (tr.t <= hi_t)
        if not m.any():
            continue
        if aoi_bounds is not None:
            min_lon, min_lat, max_lon, max_lat = aoi_bounds
            m &= (
                (tr.lon >= min_lon) & (tr.lon <= max_lon)
                & (tr.lat >= min_lat) & (tr.lat <= max_lat)
            )
        if m.any():
            in_scene += 1
            in_scene_mmsis.append(tr.mmsi)

    # The envelope is sampled hourly, but drift is continuous. Test each vessel
    # across the half-hour either side of a step so the checks tile the timeline
    # instead of probing isolated instants; a vessel crossing the envelope 20
    # minutes off an exact snapshot is still a match.
    offsets = np.array([-0.5, -0.25, 0.0, 0.25, 0.5]) * 3600.0

    hits: dict[int, Candidate] = {}
    for s in steps:
        ts = s["time"].timestamp()
        poly = s["polygon"]
        minx, miny, maxx, maxy = poly.bounds
        for tr in tracks.values():
            matched_at = None
            for off in offsets:
                pos = tr.position_at(ts + off)
                if pos is None:
                    continue
                lat, lon = pos
                # Cheap bbox reject before the exact containment test.
                if not (minx <= lon <= maxx and miny <= lat <= maxy):
                    continue
                if poly.contains(Point(lon, lat)):
                    matched_at = ts + off
                    break
            if matched_at is None:
                continue
            c = hits.get(tr.mmsi)
            if c is None:
                c = hits[tr.mmsi] = Candidate(tr.mmsi, tr, [], [])
            c.hit_times.append(matched_at)
            c.hit_hours_back.append(s["hours_back"])
            c.hit_sog.append(float(np.interp(matched_at, tr.t, tr.sog)))

    # Order candidates by how often they were inside the envelope WHILE UNDER WAY,
    # not by raw intersection count.
    #
    # Raw count is actively backwards on real data. A moored vessel sits inside
    # the envelope at every single time step and scores the maximum; a vessel
    # transiting through passes in one or two. On the Houston feed that put all
    # twelve top candidates at 0.00 kn median speed and pushed the real culprit
    # to 37th, where truncation dropped it before it was ever scored.
    #
    # Being under way is also the physically meaningful criterion: MARPOL Annex I
    # regulations 15 and 34 both condition any lawful discharge on the ship
    # "proceeding en route", and only a moving vessel lays the elongated,
    # course-aligned slick the orientation term is built to recognise. Stationary
    # vessels are still kept and still scored - they simply lose priority, so a
    # genuine culprit that stopped is never silently discarded.
    candidates = sorted(
        hits.values(),
        key=lambda c: (-c.underway_hits(min_underway_kn), -len(c.hit_times)),
    )
    underway = sum(1 for c in candidates if c.underway_hits(min_underway_kn) > 0)

    funnel = {
        "total_in_scene": in_scene,
        "in_envelope": len(candidates),
        "in_envelope_underway": underway,
        "scored": 0,
        "ranked": 0,
        "in_scene_mmsis": in_scene_mmsis,
        "in_envelope_mmsis": [c.mmsi for c in candidates],
    }
    return candidates, funnel
