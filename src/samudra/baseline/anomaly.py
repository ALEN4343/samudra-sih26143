"""Per-cell backscatter baseline and anomaly scoring — CLAUDE.md 5.1 (layer 2).

This is the architectural commitment that makes the system stateless-per-scene no
longer: an H3 resolution-7 grid holds running sigma0 statistics per cell,
**bucketed by wind speed**, so a new scene is z-scored against what that
particular patch of sea normally looks like under those particular conditions.

Wind conditioning is the whole point. A cell is legitimately dark at 2 m/s and
legitimately bright at 12 m/s; pooling those produces a standard deviation so
wide that nothing is ever anomalous.

A chronic seep is not anomalous against its own history — which is the feature,
not a bug. It stops a known natural seep being reported as a fresh discharge
every revisit.

Bootstrapping needs >= 15 scenes per cell. Below that the local statistics are
noise, so it falls back to a global background estimate and says so loudly rather
than quietly reporting confident nonsense.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

H3_RESOLUTION = 7
MIN_SCENES_FOR_LOCAL = 15
WIND_BINS = [0.0, 3.0, 5.0, 7.0, 10.0, np.inf]
WIND_BIN_LABELS = ["0-3", "3-5", "5-7", "7-10", "10+"]


def wind_bin(speed_ms: float) -> str:
    """Label the wind bucket a speed falls into."""
    for i in range(len(WIND_BINS) - 1):
        if WIND_BINS[i] <= speed_ms < WIND_BINS[i + 1]:
            return WIND_BIN_LABELS[i]
    return WIND_BIN_LABELS[-1]


def cell_index(lats: np.ndarray, lons: np.ndarray, resolution: int = H3_RESOLUTION):
    """H3 cell id per position. ~5 km^2 cells at resolution 7."""
    import h3

    return np.array(
        [h3.latlng_to_cell(float(la), float(lo), resolution) for la, lo in zip(lats, lons)]
    )


def accumulate(
    store: Path,
    db: np.ndarray,
    transform,
    bounds,
    wind_speed_ms: float,
    scene_id: str,
    sample_step: int = 16,
    resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    """Fold one scene into the running per-(cell, wind_bin) statistics.

    Pixels are sampled on a stride rather than used wholesale: adjacent SAR
    pixels are not independent samples of a cell's backscatter, and a 13 MP scene
    would otherwise make the counts meaningless as a measure of evidence.
    """
    ny, nx = db.shape
    ys = np.arange(0, ny, sample_step)
    xs = np.arange(0, nx, sample_step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")

    lat = bounds.top + (gy + 0.5) * (bounds.bottom - bounds.top) / ny
    lon = bounds.left + (gx + 0.5) * (bounds.right - bounds.left) / nx
    vals = db[gy, gx].ravel()

    cells = cell_index(lat.ravel(), lon.ravel(), resolution)
    wb = wind_bin(wind_speed_ms)

    fresh = pd.DataFrame({"cell": cells, "wind_bin": wb, "v": vals})
    agg = (
        fresh.groupby(["cell", "wind_bin"])
        .agg(n=("v", "size"), s=("v", "sum"), ss=("v", lambda x: float((x * x).sum())))
        .reset_index()
    )
    agg["scenes"] = 1
    agg["scene_ids"] = scene_id

    store = Path(store)
    if store.exists():
        prev = pd.read_parquet(store)
        if scene_id in set(prev.scene_ids.str.split(",").explode()):
            # Folding the same scene twice would inflate the evidence count and
            # shrink the variance for free.
            return prev
        merged = pd.concat([prev, agg], ignore_index=True)
        out = (
            merged.groupby(["cell", "wind_bin"])
            .agg(
                n=("n", "sum"), s=("s", "sum"), ss=("ss", "sum"),
                scenes=("scenes", "sum"),
                scene_ids=("scene_ids", lambda x: ",".join(sorted(set(",".join(x).split(","))))),
            )
            .reset_index()
        )
    else:
        out = agg

    store.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(store, index=False)
    return out


def statistics(store: Path) -> pd.DataFrame:
    """Per-(cell, wind_bin) mean and standard deviation."""
    df = pd.read_parquet(store)
    df["mean"] = df.s / df.n
    df["var"] = np.maximum(df.ss / df.n - df["mean"] ** 2, 1e-9)
    df["std"] = np.sqrt(df["var"])
    return df


def anomaly_z(
    db: np.ndarray,
    transform,
    bounds,
    wind_speed_ms: float,
    store: Path | None = None,
    sample_step: int = 16,
    resolution: int = H3_RESOLUTION,
    min_scenes: int = MIN_SCENES_FOR_LOCAL,
) -> tuple[np.ndarray, dict]:
    """Z-score the scene against the per-cell baseline. Negative = darker.

    Returns (z image, info). `info["mode"]` is "local" or "global_fallback"; the
    caller is expected to surface which one ran.
    """
    ny, nx = db.shape
    info: dict = {
        "resolution": resolution,
        "wind_bin": wind_bin(wind_speed_ms),
        "min_scenes_for_local": min_scenes,
    }

    stats = None
    if store is not None and Path(store).exists():
        s = statistics(Path(store))
        s = s[(s.wind_bin == info["wind_bin"]) & (s.scenes >= min_scenes)]
        if len(s):
            stats = s.set_index("cell")[["mean", "std"]].to_dict("index")

    if not stats:
        # CLAUDE.md 5.1: fall back to a global background estimate, and log it.
        med = float(np.median(db))
        mad = float(np.median(np.abs(db - med))) * 1.4826  # robust sigma
        z = (db - med) / max(mad, 1e-6)
        info.update(
            mode="global_fallback",
            reason=(
                f"fewer than {min_scenes} scenes in the {info['wind_bin']} m/s wind bin "
                f"for these cells — local baseline not established"
            ),
            global_median_db=med,
            global_sigma_db=mad,
            cells_with_local_stats=0,
        )
        return z, info

    # Local path: z-score each pixel against its own cell's conditional stats.
    ys = np.arange(0, ny, sample_step)
    xs = np.arange(0, nx, sample_step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    lat = bounds.top + (gy + 0.5) * (bounds.bottom - bounds.top) / ny
    lon = bounds.left + (gx + 0.5) * (bounds.right - bounds.left) / nx
    cells = cell_index(lat.ravel(), lon.ravel(), resolution).reshape(gy.shape)

    med = float(np.median(db))
    mad = max(float(np.median(np.abs(db - med))) * 1.4826, 1e-6)
    mu_c = np.full(gy.shape, med, dtype=np.float32)
    sd_c = np.full(gy.shape, mad, dtype=np.float32)
    hits = 0
    for idx, c in np.ndenumerate(cells):
        st = stats.get(c)
        if st:
            mu_c[idx] = st["mean"]
            sd_c[idx] = max(st["std"], 1e-6)
            hits += 1

    from scipy import ndimage

    mu = ndimage.zoom(mu_c, (ny / mu_c.shape[0], nx / mu_c.shape[1]), order=1)[:ny, :nx]
    sd = ndimage.zoom(sd_c, (ny / sd_c.shape[0], nx / sd_c.shape[1]), order=1)[:ny, :nx]
    if mu.shape != db.shape:
        mu = np.resize(mu, db.shape)
        sd = np.resize(sd, db.shape)

    info.update(
        mode="local",
        cells_with_local_stats=int(hits),
        cells_sampled=int(cells.size),
    )
    return (db - mu) / np.maximum(sd, 1e-6), info


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline accumulation and anomaly scoring.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--store", default="data/baseline/sigma0_cells.parquet")
    ap.add_argument("--accumulate", action="store_true",
                    help="fold this scene into the baseline store")
    a = ap.parse_args()

    from datetime import datetime

    from samudra.attribution import drift
    from samudra.detection.preprocess import scene_db

    d = Path(a.root) / a.incident
    db, transform, bounds = scene_db(d / "scene.tif")
    gj = json.loads((d / "observed_slick.geojson").read_text())
    acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])
    env = drift.EnvField.load(d / "env.npz")
    wind = env.summary(acq.timestamp())["mean_wind_speed_ms"]

    if a.accumulate:
        out = accumulate(Path(a.store), db, transform, bounds, wind, a.incident)
        print(f"baseline store    : {a.store}")
        print(f"  cells x bins    : {len(out)}")
        print(f"  scenes folded   : {int(out.scenes.max())} max per cell")

    z, info = anomaly_z(db, transform, bounds, wind, Path(a.store))
    print(f"wind              : {wind:.1f} m/s  -> bin {info['wind_bin']} m/s")
    print(f"anomaly mode      : {info['mode'].upper()}")
    if info["mode"] == "global_fallback":
        print(f"  !! {info['reason']}")
        print(f"  global background {info['global_median_db']:.1f} dB "
              f"sigma {info['global_sigma_db']:.2f} dB")
    else:
        print(f"  cells with local statistics: {info['cells_with_local_stats']} "
              f"of {info['cells_sampled']} sampled")
    print(f"z range           : {z.min():.1f} to {z.max():.1f}")
    print(f"strongly dark     : {(z < -1.5).mean() * 100:.3f}% of pixels below z=-1.5")


if __name__ == "__main__":
    main()
