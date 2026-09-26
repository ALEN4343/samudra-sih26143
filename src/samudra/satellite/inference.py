"""ML inference on a satellite observation — CLAUDE.md layer 3, from the console.

What runs here is the trained DeepLabv3+ checkpoint from `data/models/seg.pt`,
through `detection/segmenter.infer`, over the actual pixels of the selected
observation. There is no second model, no cached mask, no lookup table.

THINGS THIS MODULE MUST NEVER DO, and the reasons:

* Read `ground_truth.json`, or a `label/` image, or the CSIRO `Class_*` folder
  name, at any point on the prediction path. The whole claim of the Satellite
  tab is that the mask came out of the model. `assert_no_ground_truth()` below
  is a runtime guard, and `tests/test_satellite.py` asserts it stays true.
* Report a confidence that is not derived from the model's own output.
* Present a dark patch as oil. SAR cannot distinguish an oil slick from a
  biogenic slick, a low-wind cell, rain cells or a wind shadow. The output is
  named OIL-LIKE SURFACE ANOMALY everywhere it surfaces.

Preprocessing matches training exactly. Training fed `uint8/255` into the
ImageNet normalisation; `segmenter.prepare_tile` maps dB [-35, 0] onto [0, 1]
and then normalises. Mapping an 8-bit chip through the same linear ramp makes
the two paths algebraically identical, so a chip is scored the way it was
trained rather than through a second, subtly different pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from samudra.detection import segmenter
from samudra.detection.segmenter import DB_CLIP_HI, DB_CLIP_LO

FORBIDDEN_ON_INFERENCE_PATH = ("ground_truth.json", "/label/", "\\label\\")


def assert_no_ground_truth(*paths: str | Path) -> None:
    """Refuse to proceed if anything label-shaped is on the inference path."""
    for p in paths:
        s = str(p).replace("\\", "/").lower()
        if "ground_truth.json" in s or "/label/" in s:
            raise RuntimeError(
                f"inference path touched a label source: {p}. The satellite "
                f"detection must come from the image alone."
            )


def load_grey(path: Path | str) -> np.ndarray:
    """Observation raster -> HxW uint8 greyscale, whatever the container."""
    p = Path(path)
    assert_no_ground_truth(p)
    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(p) as ds:
                a = ds.read(1).astype(np.float32)
            finite = np.isfinite(a)
            if finite.any():
                lo, hi = np.percentile(a[finite], [2, 98])
                a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
            return a.astype(np.uint8)
        except Exception:  # noqa: BLE001 - fall through to PIL
            pass
    from PIL import Image

    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def _u8_to_db(img: np.ndarray) -> np.ndarray:
    """8-bit chip onto the dB ramp the segmenter expects. Exactly invertible."""
    return DB_CLIP_LO + (img.astype(np.float32) / 255.0) * (DB_CLIP_HI - DB_CLIP_LO)


def load_scene_db(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Observation raster -> (sigma0 dB, valid mask, how it got there).

    Two genuinely different inputs arrive here and they must not be treated the
    same way:

    CALIBRATED   a real product run through `ingest/eos04.py`, carrying absolute
                 sigma-nought in dB and a SAMUDRA_CALIBRATED tag. Its dB values
                 are physically meaningful, so they go straight onto the
                 [-35, 0] ramp the model was trained against. A dark pixel here
                 is dark in an absolute sense.

    UNCALIBRATED a research chip with no radiometric scale. The only thing
                 available is a per-image percentile stretch, which makes the
                 result *relative to that chip* — the darkest thing present
                 becomes dark regardless of what it actually backscattered.
                 That is a real limitation of chip replay and is reported as
                 such rather than smoothed over.

    Feeding a calibrated product through the percentile path would discard the
    absolute scale that makes it worth having, so the two are kept apart.
    """
    p = Path(path)
    assert_no_ground_truth(p)

    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(p) as ds:
                tags = ds.tags()
                if str(tags.get("SAMUDRA_CALIBRATED", "")).lower() == "true":
                    db = ds.read(1).astype(np.float32)
                    valid = np.isfinite(db)
                    if not valid.any():
                        raise ValueError(f"{p} has no valid pixels")
                    # The network cannot take NaN. Filling with the scene median
                    # makes absent data look like ordinary sea, so it neither
                    # invents a dark anomaly nor a bright one; predictions there
                    # are discarded below regardless.
                    fill = float(np.median(db[valid]))
                    db = np.where(valid, db, fill).astype(np.float32)
                    return db, valid, {
                        "radiometry": "absolute",
                        "units": "sigma0 dB",
                        "equation": tags.get("SAMUDRA_EQUATION"),
                        "satellite": tags.get("SAMUDRA_SATELLITE"),
                        "acquired_at": tags.get("SAMUDRA_ACQUIRED_AT"),
                        "invalid_fraction": round(float((~valid).mean()), 4),
                        "detail": "calibrated sigma0 dB used directly; no "
                                  "per-image stretch applied",
                    }
        except ImportError:
            pass

    grey = load_grey(p)
    return _u8_to_db(grey), np.ones(grey.shape, dtype=bool), {
        "radiometry": "relative",
        "units": "arbitrary (per-image 2-98 percentile stretch)",
        "detail": "Uncalibrated chip. Contrast is relative to this image only, "
                  "so dB values are not physical and must not be quoted.",
    }


def checkpoint_status(checkpoint: Path | str = segmenter.DEFAULT_CHECKPOINT) -> dict:
    """What model is installed, and is it fit to quote."""
    ck = Path(checkpoint)
    if not ck.exists():
        return {"available": False, "path": str(ck),
                "detail": "No checkpoint. Train with scripts/train_segmenter.py."}
    try:
        import torch

        blob = torch.load(ck, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(ck), "detail": f"unreadable: {exc}"}
    meta = {k: v for k, v in blob.items() if k != "model"}
    return {
        "available": True,
        "path": str(ck),
        "sha256_prefix": None,
        "arch": meta.get("arch"),
        "classes": meta.get("classes"),
        "input_size": meta.get("input_size"),
        "epoch": meta.get("epoch"),
        "val_oil_iou": meta.get("val_oil_iou", meta.get("oil_iou")),
        "selected_on": meta.get("selected_on", "unrecorded"),
        "representative": bool(meta.get("representative", False)),
        "detail": (
            "Checkpoint is from a full run and its metrics are quotable."
            if meta.get("representative")
            else "Checkpoint is marked NOT REPRESENTATIVE (reduced or CPU run). "
                 "Predictions still come from the real model, but do not quote "
                 "its accuracy."
        ),
    }


def run(
    raster_path: Path | str,
    checkpoint: Path | str = segmenter.DEFAULT_CHECKPOINT,
    threshold: float | None = None,
    min_area_px: int = 64,
) -> dict[str, Any]:
    """Segment one observation. Returns probability, mask and honest statistics."""
    p = Path(raster_path)
    assert_no_ground_truth(p)

    stages: list[dict] = []

    t0 = time.perf_counter()
    db, valid, radiometry = load_scene_db(p)
    stages.append({"stage": "observation received", "ms": (time.perf_counter() - t0) * 1e3,
                   "detail": f"{db.shape[1]}x{db.shape[0]} px, "
                             f"{radiometry['radiometry']} radiometry "
                             f"({radiometry['units']})"})

    t0 = time.perf_counter()
    stages.append({"stage": "preprocessing", "ms": (time.perf_counter() - t0) * 1e3,
                   "detail": f"clipped to dB ramp [{DB_CLIP_LO:g}, {DB_CLIP_HI:g}], "
                             f"ImageNet normalisation — identical to training; "
                             f"{radiometry['detail']}"})

    t0 = time.perf_counter()
    model, device, meta = segmenter.load_model(checkpoint)
    thr = float(threshold if threshold is not None else _default_threshold())
    prob = segmenter.infer(db, model, device)
    # No data means no detection. Filled pixels were never observed, so anything
    # predicted over them is an artefact of the fill, not of the sensor.
    prob = np.where(valid, prob, 0.0).astype(np.float32)
    mask = prob >= thr
    infer_ms = (time.perf_counter() - t0) * 1e3
    stages.append({"stage": "ML inference", "ms": infer_ms,
                   "detail": f"DeepLabv3+/ResNet-50 on {device.type}, "
                             f"threshold {thr:.2f}"})

    # Drop specks: a handful of pixels is not a slick, and letting them through
    # inflates the anomaly count with noise.
    t0 = time.perf_counter()
    labelled, regions = _components(mask, min_area_px)
    stages.append({"stage": "anomaly characterisation", "ms": (time.perf_counter() - t0) * 1e3,
                   "detail": f"{len(regions)} connected region(s) >= {min_area_px} px"})

    # Over observed pixels only. Counting the nodata border in the denominator
    # would quietly shrink the reported fraction on any windowed product.
    oil_frac = float(mask[valid].mean()) if valid.any() else 0.0
    conf = float(prob[mask].mean()) if mask.any() else 0.0

    # 8-bit render for display. Derived from the dB the model actually saw, so
    # what the operator looks at is what was scored.
    grey = np.clip((db - DB_CLIP_LO) / (DB_CLIP_HI - DB_CLIP_LO), 0, 1)
    grey = (grey * 255).astype(np.uint8)

    return {
        "prob": prob,
        "mask": mask,
        "db": db,
        "valid": valid,
        "radiometry": radiometry,
        "grey": grey,
        "labelled": labelled,
        "regions": regions,
        "threshold": thr,
        "anomaly_pixel_fraction": oil_frac,
        "mean_confidence_in_mask": conf,
        "max_probability": float(prob.max()),
        "device": device.type,
        "checkpoint_meta": meta,
        "stages": stages,
        "class_label": "OIL-LIKE SURFACE ANOMALY",
        "class_caveat": (
            "SAR measures surface roughness. Oil, biogenic slicks, low-wind "
            "cells, rain cells and wind shadows all darken it. This is an "
            "anomaly consistent with oil, not a confirmed oil detection."
        ),
    }


def _default_threshold() -> float:
    """The operating point chosen on validation during training, if recorded."""
    import json

    m = Path("artifacts/model_metrics.json")
    if m.exists():
        try:
            return float(json.loads(m.read_text()).get("decision_threshold", 0.5))
        except Exception:  # noqa: BLE001
            pass
    return 0.5


def _components(mask: np.ndarray, min_area_px: int):
    """Connected components without a SciPy dependency."""
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    regions: list[dict] = []
    cur = 0
    stack: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or lab[y, x]:
                continue
            cur += 1
            stack.append((y, x))
            lab[y, x] = cur
            pix: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pix.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
            if len(pix) < min_area_px:
                for cy, cx in pix:
                    lab[cy, cx] = 0
                cur -= 1
                continue
            ys = np.array([q[0] for q in pix], dtype=np.float64)
            xs = np.array([q[1] for q in pix], dtype=np.float64)
            regions.append(_shape_stats(ys, xs, len(pix)))
    return lab, regions


def _shape_stats(ys: np.ndarray, xs: np.ndarray, n: int) -> dict:
    """Second-moment shape descriptors, in pixels — the same family the
    attribution layer consumes once they are scaled to metres."""
    cy, cx = float(ys.mean()), float(xs.mean())
    y0, x0 = ys - cy, xs - cx
    cov = np.cov(np.vstack([x0, y0])) if n > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    major = float(4 * np.sqrt(max(evals[0], 1e-9)))
    minor = float(4 * np.sqrt(max(evals[1], 1e-9)))
    ecc = float(np.sqrt(max(0.0, 1 - (evals[1] / max(evals[0], 1e-9)))))
    # image y grows downward, so negate to get a map-style bearing
    ang = float(np.degrees(np.arctan2(evecs[0, 0], -evecs[1, 0])) % 180.0)
    return {
        "area_px": int(n),
        "centroid_px": [cx, cy],
        "major_axis_px": major,
        "minor_axis_px": minor,
        "eccentricity": ecc,
        "orientation_deg": ang,
    }
