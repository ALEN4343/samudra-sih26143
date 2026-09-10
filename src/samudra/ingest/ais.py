"""Real AIS ingestion — CLAUDE.md layer 1, section 5.3.

Loads a NOAA MarineCadastre bulk broadcast-points file and normalises it to the
same parquet schema `synth/generate.py` produces, so every downstream module runs
unchanged on real traffic.

The column names here were read off the actual file, not assumed. This file uses
lowercase snake_case (`mmsi`, `base_date_time`, `longitude`, `latitude`), NOT the
`MMSI`/`BaseDateTime`/`LAT`/`LON` of the classic MarineCadastre export. CLAUDE.md
section 8 requires inspecting before writing a loader, and this is why: assuming
the documented names raises AttributeError on the first chunk.

Real data is also missing things the synthetic generator never omits — heading is
null on 46% of rows, IMO on 67%, COG on 18%. Nothing here fabricates a value to
fill a hole.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.timeutil import epoch_seconds

# Columns this loader needs, as they actually appear in the file.
REQUIRED = ("mmsi", "base_date_time", "longitude", "latitude", "sog")

# ITU-R M.1371 ship-type codes. Mapped to descriptive labels because the
# type_risk prior in config/weights.yaml matches on substrings ("tanker",
# "cargo", "tug", "fishing"), so a bare integer would silently score 1.0.
_TYPE_RANGES: list[tuple[int, int, str]] = [
    (20, 29, "Wing In Ground"),
    (40, 49, "High Speed Craft"),
    (60, 69, "Passenger"),
    (70, 79, "Cargo"),
    (80, 89, "Tanker"),
    (90, 99, "Other"),
]
_TYPE_EXACT: dict[int, str] = {
    30: "Fishing Vessel",
    31: "Towing",
    32: "Towing Large",
    33: "Dredger",
    34: "Diving Support",
    35: "Military",
    36: "Sailing Vessel",
    37: "Pleasure Craft",
    50: "Pilot Vessel",
    51: "Search And Rescue",
    52: "Tug",
    53: "Port Tender",
    54: "Anti-Pollution",
    55: "Law Enforcement",
    58: "Medical Transport",
    59: "Non-Combatant",
}

# MMSI MID prefix -> ISO country code. Enough of the register to cover US coastal
# traffic plus the common foreign flags; unknown prefixes stay None rather than
# being guessed.
MID_TO_FLAG: dict[int, str] = {
    338: "US", 366: "US", 367: "US", 368: "US", 369: "US",
    316: "CA", 345: "MX", 341: "KN", 339: "KY", 305: "AG", 306: "CW",
    370: "PA", 371: "PA", 372: "PA", 373: "PA", 374: "PA",
    636: "LR", 538: "MH", 563: "SG", 477: "HK", 412: "CN", 431: "JP",
    440: "KR", 249: "MT", 248: "MT", 240: "GR", 241: "GR", 232: "GB",
    235: "GB", 236: "GI", 244: "NL", 245: "NL", 246: "NL", 205: "BE",
    211: "DE", 218: "DE", 227: "FR", 228: "FR", 247: "IT", 224: "ES",
    257: "NO", 258: "NO", 259: "NO", 265: "SE", 266: "SE", 219: "DK",
    220: "DK", 273: "RU", 419: "IN", 422: "IR", 463: "PK", 461: "OM",
    470: "AE", 403: "SA", 525: "ID", 574: "VN", 567: "TH", 533: "MY",
    548: "PH", 710: "BR", 725: "CL", 761: "VE",
}


def vessel_type_label(code) -> str | None:
    """Numeric AIS ship-type code -> descriptive label. None when absent."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    if c in _TYPE_EXACT:
        return _TYPE_EXACT[c]
    for lo, hi, label in _TYPE_RANGES:
        if lo <= c <= hi:
            return label
    return None


def flag_from_mmsi(mmsi) -> str | None:
    try:
        s = str(int(mmsi))
    except (TypeError, ValueError):
        return None
    return MID_TO_FLAG.get(int(s[:3])) if len(s) >= 9 else None


def _imo_to_int(v) -> int | None:
    """'IMO8941145' -> 8941145. Real files carry the prefix; ours is an int."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper().removeprefix("IMO").strip()
    return int(s) if s.isdigit() else None


def load_csv(
    path: Path | str,
    bbox: tuple[float, float, float, float] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read a MarineCadastre CSV (optionally .zst) into the synthetic schema.

    `bbox` is (min_lon, min_lat, max_lon, max_lat) and filters while streaming, so
    a nationwide file never needs to fit in memory.
    """
    path = Path(path)

    def _frames():
        if path.suffix == ".zst":
            import zstandard as zstd

            with path.open("rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as r:
                yield from pd.read_csv(
                    io.TextIOWrapper(r, encoding="utf-8", errors="replace"),
                    chunksize=chunksize, low_memory=False,
                )
        else:
            yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)

    keep: list[pd.DataFrame] = []
    total = 0
    for chunk in _frames():
        if total == 0:
            missing = [c for c in REQUIRED if c not in chunk.columns]
            if missing:
                raise ValueError(
                    f"{path} is missing required columns {missing}. "
                    f"Found: {list(chunk.columns)}"
                )
        total += len(chunk)
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            chunk = chunk[
                chunk.longitude.between(min_lon, max_lon)
                & chunk.latitude.between(min_lat, max_lat)
            ]
        if len(chunk):
            keep.append(chunk)

    if not keep:
        raise ValueError(f"no rows survived the bbox filter over {total:,} rows")

    df = pd.concat(keep, ignore_index=True)

    out = pd.DataFrame(
        {
            "mmsi": df.mmsi.astype("int64"),
            "t": pd.to_datetime(df.base_date_time, utc=True),
            "lat": df.latitude.astype(float),
            "lon": df.longitude.astype(float),
            "sog": df.sog.astype(float),
            "cog": df.cog.astype(float) if "cog" in df else np.nan,
            "heading": df.heading.astype(float) if "heading" in df else np.nan,
            "vessel_name": df.vessel_name if "vessel_name" in df else None,
            "imo": df.imo.map(_imo_to_int) if "imo" in df else None,
            "vessel_type": (
                df.vessel_type.map(vessel_type_label) if "vessel_type" in df else None
            ),
            "length_m": df.length.astype(float) if "length" in df else np.nan,
            "flag": df.mmsi.map(flag_from_mmsi),
        }
    )

    # A vessel reporting twice at the same instant is a duplicate broadcast, not
    # two positions. Left in, it reads to the trust layer as a duplicate-MMSI
    # transmission from two places.
    before = len(out)
    out = out.drop_duplicates(subset=["mmsi", "t"], keep="first")
    out = out.sort_values(["mmsi", "t"]).reset_index(drop=True)
    out.attrs["national_rows"] = total
    out.attrs["duplicates_dropped"] = before - len(out)
    return out


def find_gaps(df: pd.DataFrame, gap_min: float = 15.0) -> pd.DataFrame:
    """Per-vessel reporting gaps longer than `gap_min` minutes."""
    rows = []
    for mmsi, g in df.groupby("mmsi", sort=False):
        if len(g) < 2:
            continue
        ts = epoch_seconds(g.t)
        dt = np.diff(ts)
        for i in np.flatnonzero(dt > gap_min * 60.0):
            i = int(i)
            rows.append(
                {
                    "mmsi": int(mmsi),
                    "start_t": g.t.iloc[i],
                    "end_t": g.t.iloc[i + 1],
                    "duration_min": float(dt[i] / 60.0),
                    "start_lat": float(g.lat.iloc[i]),
                    "start_lon": float(g.lon.iloc[i]),
                    "end_lat": float(g.lat.iloc[i + 1]),
                    "end_lon": float(g.lon.iloc[i + 1]),
                }
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    g = df.groupby("mmsi")
    moving = df[df.sog > 1.0]
    return {
        "rows": len(df),
        "vessels": int(df.mmsi.nunique()),
        "t_min": df.t.min(),
        "t_max": df.t.max(),
        "bbox": [float(df.lon.min()), float(df.lat.min()),
                 float(df.lon.max()), float(df.lat.max())],
        "median_reports_per_vessel": float(g.size().median()),
        "vessels_over_200_reports": int((g.size() >= 200).sum()),
        "vessels_underway": int(moving.mmsi.nunique()),
        "pct_rows_underway": 100.0 * len(moving) / max(len(df), 1),
        "null_heading_pct": 100.0 * df.heading.isna().mean(),
        "null_imo_pct": 100.0 * df.imo.isna().mean(),
        "null_type_pct": 100.0 * df.vessel_type.isna().mean(),
        "gaps": len(gaps),
        "vessels_with_gaps": int(gaps.mmsi.nunique()) if len(gaps) else 0,
        "types": df.drop_duplicates("mmsi").vessel_type.value_counts().head(8).to_dict(),
        "flags": df.drop_duplicates("mmsi").flag.value_counts().head(5).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest real AIS into the pipeline schema.")
    ap.add_argument("--input", default="data/raw/ais/houston_filtered.csv")
    ap.add_argument("--out", default="data/raw/ais/houston.parquet")
    ap.add_argument("--bbox", type=float, nargs=4, default=None,
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--gap-min", type=float, default=15.0)
    a = ap.parse_args()

    df = load_csv(a.input, bbox=tuple(a.bbox) if a.bbox else None)
    gaps = find_gaps(df, a.gap_min)
    s = summarise(df, gaps)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if len(gaps):
        gaps.to_parquet(out.with_name(out.stem + "_gaps.parquet"), index=False)

    print(f"source            : {a.input}")
    if df.attrs.get("national_rows"):
        print(f"rows read         : {df.attrs['national_rows']:,}")
    print(f"rows kept         : {s['rows']:,}   (duplicates dropped "
          f"{df.attrs.get('duplicates_dropped', 0):,})")
    print(f"vessels           : {s['vessels']:,}")
    print(f"time span         : {s['t_min']}  ->  {s['t_max']}")
    print(f"bbox              : {[round(v, 3) for v in s['bbox']]}")
    print(f"reports/vessel    : median {s['median_reports_per_vessel']:.0f}, "
          f"{s['vessels_over_200_reports']} vessels with 200+")
    print(f"underway (>1 kn)  : {s['vessels_underway']} vessels, "
          f"{s['pct_rows_underway']:.1f}% of rows")
    print(f"AIS gaps >{a.gap_min:.0f} min : {s['gaps']:,} across {s['vessels_with_gaps']} vessels")
    print(f"nulls             : heading {s['null_heading_pct']:.1f}%, "
          f"imo {s['null_imo_pct']:.1f}%, type {s['null_type_pct']:.1f}%")
    print(f"types             : {s['types']}")
    print(f"flags             : {s['flags']}")
    print(f"written           : {out}")


if __name__ == "__main__":
    main()
