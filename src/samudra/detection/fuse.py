"""Confidence fusion — CLAUDE.md section 5.2.

    confidence = cnn_oil_prob * sigmoid(baseline_anomaly_z - 1.5) * wind_gate_pass

A dark patch is only oil if the CNN says so **and** it is anomalous against that
cell's own history **and** the wind was in the detectable band. A look-alike
fails at least one of the three: a low-wind glassy patch fails the gate, a
chronic seep fails the anomaly term because it is normal for that cell, and a
CNN false positive fails the first.

Note the sign. `anomaly_z` is signed with negative meaning darker than baseline,
and oil is dark, so the term uses the *magnitude of darkness* `-z`. Feeding the
raw z in would score bright anomalies as oil and suppress the actual slicks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANOMALY_OFFSET = 1.5


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fuse(
    cnn_oil_prob: np.ndarray | float,
    anomaly_z: np.ndarray | float,
    wind_gate_pass: np.ndarray | float | bool,
    offset: float = ANOMALY_OFFSET,
) -> np.ndarray | float:
    """Fused, calibrated detection confidence in [0, 1]."""
    darkness = -np.asarray(anomaly_z, dtype=np.float64)
    return (
        np.asarray(cnn_oil_prob, dtype=np.float64)
        * sigmoid(darkness - offset)
        * np.asarray(wind_gate_pass, dtype=np.float64)
    )


def fuse_detection(rec: dict, anomaly_z: float, wind_gate_pass: bool, offset: float = ANOMALY_OFFSET) -> dict:
    """Attach the fusion terms to one SlickDetection record."""
    p = float(rec.get("cnn_oil_prob", 0.0))
    conf = float(fuse(p, anomaly_z, bool(wind_gate_pass), offset))
    out = dict(rec)
    out.update(
        baseline_anomaly_z=float(anomaly_z),
        wind_gate_pass=bool(wind_gate_pass),
        confidence=conf,
        # A look-alike set is not labelled in either training set, so the model
        # has no look-alike class. Reporting the complement would imply a
        # discrimination the model was never trained to make.
        cnn_lookalike_prob=0.0,
        fusion={
            "cnn_oil_prob": p,
            "anomaly_term": float(sigmoid(-anomaly_z - offset)),
            "wind_gate_pass": bool(wind_gate_pass),
            "offset": offset,
        },
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse CNN, anomaly and wind gate into confidence.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--store", default="data/baseline/sigma0_cells.parquet")
    a = ap.parse_args()

    from datetime import datetime

    import rasterio

    from samudra.attribution import drift
    from samudra.baseline.anomaly import anomaly_z as compute_z
    from samudra.detection.preprocess import scene_db

    d = Path(a.root) / a.incident
    p = d / "detected_slicks.geojson"
    if not p.exists():
        raise SystemExit(
            f"{p} not found — run the segmenter and polygonize stages first"
        )

    db, transform, bounds = scene_db(d / "scene.tif")
    gj = json.loads((d / "observed_slick.geojson").read_text())
    acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])
    env = drift.EnvField.load(d / "env.npz")
    summ = env.summary(acq.timestamp())

    z_img, info = compute_z(db, transform, bounds, summ["mean_wind_speed_ms"], Path(a.store))

    fc = json.loads(p.read_text())
    out_feats = []
    print(f"anomaly mode      : {info['mode'].upper()}"
          + ("  (local baseline not established)" if info["mode"] != "local" else ""))
    print(f"wind gate         : {'PASS' if summ['wind_gate_pass'] else 'FAIL'} "
          f"({summ['mean_wind_speed_ms']:.1f} m/s)")
    print()
    print(f"{'slick':<12}{'area km2':>10}{'cnn':>7}{'z':>8}{'anom':>7}{'conf':>7}")
    print("-" * 51)

    for f in fc["features"]:
        props = f["properties"]
        inside = rasterio.features.geometry_mask(
            [f["geometry"]], out_shape=db.shape, transform=transform, invert=True
        )
        z = float(np.median(z_img[inside])) if inside.any() else 0.0
        merged = fuse_detection(props, z, summ["wind_gate_pass"])
        out_feats.append({"type": "Feature", "geometry": f["geometry"], "properties": merged})
        print(f"{merged['slick_id']:<12}{merged['area_km2']:>10.2f}"
              f"{merged.get('cnn_oil_prob', 0):>7.3f}{z:>8.2f}"
              f"{merged['fusion']['anomaly_term']:>7.3f}{merged['confidence']:>7.3f}")

    (d / "detected_slicks.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "properties": {"anomaly": info},
                    "features": out_feats}, indent=2)
    )
    print(f"\nwritten           : {p}")


if __name__ == "__main__":
    main()
