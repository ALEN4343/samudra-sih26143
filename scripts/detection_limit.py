"""Measure the smallest / faintest slick the real model finds in real EOS-04 pixels.

    python scripts/detection_limit.py

WHY THIS EXISTS. "Can it detect small slicks?" is the first question a judge
asks, and the honest answer is a curve, not a claim. This plants synthetic
slicks of known area and known contrast into REAL ocean backscatter from the
NRSC EOS-04 scene, runs the REAL checkpoint over them, and reports the
detection rate per cell. Every serious remote-sensing paper carries this figure;
without it, any statement about small-slick capability is an assertion.

WHAT IS REAL AND WHAT IS PLANTED
    real      the ocean: speckle statistics, wind streaks, texture, radiometry,
              all from the calibrated EOS-04 HH product
    real      the model and its decision threshold, unchanged
    planted   the slicks — elliptical, at a controlled area and contrast

Planting is done in dB by subtraction. SAR speckle is multiplicative, so a
constant damping ratio is a constant dB offset, and subtracting leaves the real
speckle texture intact underneath. Replacing the patch with smooth synthetic
values instead would make it trivially detectable and the whole measurement
worthless.

WHAT THE RESULT MEANS. The floor is set by CONTRAST, not by area. A slick
smaller than the connected-component minimum cannot be reported at all; a slick
fainter than roughly the speckle standard deviation cannot be separated from the
sea no matter how large it is.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHIP = 512
#: Areas in km2, spanning the connected-component floor up to a large spill.
AREAS_KM2 = (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
#: Damping in dB. Real oil damps capillary waves by roughly 5-15 dB; 1-3 dB is
#: the regime where a slick is indistinguishable from a wind shadow or a wake.
CONTRASTS_DB = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


def ocean_chips(raster: Path, n: int, rng, max_std_db: float = 3.2,
                clear_half: int = 96):
    """Ocean chips from the real scene, with a clear centre to plant into.

    Ships are deliberately NOT excluded. This AOI holds 195 vessels, so every
    512 px chip contains at least one 25-33 dB point target, and a scene with
    the ships removed would not be the scene the detector actually faces. What
    must be clear is only the centre, where the slick is planted, so that a
    ship's return is not counted as part of the planted feature.

    Chips are still rejected on standard deviation, which is the signature of a
    land edge or a strong radiometric gradient rather than open water. Measured
    on this product, clean ocean sits at ~2.5 dB.
    """
    import rasterio

    with rasterio.open(raster) as ds:
        arr = ds.read(1).astype(np.float32)
        gsd = abs(ds.transform.a)

    h, w = arr.shape
    half = CHIP // 2
    chips = []
    tries = 0
    while len(chips) < n and tries < n * 200:
        tries += 1
        r = int(rng.integers(0, max(1, h - CHIP)))
        c = int(rng.integers(0, max(1, w - CHIP)))
        chip = arr[r:r + CHIP, c:c + CHIP]
        if chip.shape != (CHIP, CHIP) or not np.isfinite(chip).all():
            continue
        if chip.std() > max_std_db:        # land edge or strong gradient
            continue
        core = chip[half - clear_half:half + clear_half,
                    half - clear_half:half + clear_half]
        # A ship is many pixels tens of dB above the sea. Speckle alone puts the
        # core maximum 9-12 dB above the median on this product (measured), so
        # testing the maximum against a small offset rejects open water too —
        # it rejected every chip in the AOI on the first attempt. Counting
        # pixels beyond +12 dB separates a real target from speckle cleanly.
        if int((core > np.median(chip) + 12.0).sum()) > 0:
            continue
        chips.append(chip.copy())
    return chips, gsd


def plant(chip: np.ndarray, area_km2: float, contrast_db: float, gsd_m: float,
          rng) -> tuple[np.ndarray, np.ndarray]:
    """Subtract `contrast_db` inside an ellipse of `area_km2`. Returns (chip, mask)."""
    px_area = gsd_m * gsd_m
    n_px = (area_km2 * 1e6) / px_area
    # Elongated, as a slick from a moving vessel is; 4:1 is typical.
    ratio = 4.0
    b = math.sqrt(n_px / (math.pi * ratio))
    a = ratio * b

    cy, cx = CHIP / 2.0, CHIP / 2.0
    th = rng.uniform(0, math.pi)
    yy, xx = np.mgrid[0:CHIP, 0:CHIP]
    dy, dx = yy - cy, xx - cx
    xr = dx * math.cos(th) + dy * math.sin(th)
    yr = -dx * math.sin(th) + dy * math.cos(th)
    mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

    out = chip.copy()
    out[mask] -= contrast_db
    return out, mask


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Minimum detectable slick, measured.")
    ap.add_argument("--raster",
                    default="data/satellite/incoming/eos04_kutch_water_hh.tif")
    ap.add_argument("--trials", type=int, default=6,
                    help="independent ocean chips per (area, contrast) cell")
    ap.add_argument("--min-area-px", type=int, default=64,
                    help="connected-component floor, as the pipeline uses")
    ap.add_argument("--detect-frac", type=float, default=0.25,
                    help="fraction of planted pixels that must be predicted oil")
    ap.add_argument("--out", default="artifacts/detection_limit.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--checkpoint", default=None,
                    help="compare a candidate model without installing it")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override; defaults to the one chosen on validation")
    a = ap.parse_args()

    raster = Path(a.raster)
    if not raster.exists():
        print(f"no raster at {raster} — run `python -m samudra.ingest.eos04` first")
        return 1

    from samudra.detection import segmenter
    from samudra.satellite import inference

    rng = np.random.default_rng(a.seed)
    chips, gsd = ocean_chips(raster, a.trials, rng)
    if not chips:
        print("could not find clean ocean chips in this raster")
        return 1

    ckpt = a.checkpoint or segmenter.DEFAULT_CHECKPOINT
    model, device, meta = segmenter.load_model(ckpt)
    # A candidate checkpoint carries its own validation-chosen threshold. Using
    # the installed model's threshold for it would compare two models at an
    # operating point only one of them was tuned for.
    if a.threshold is not None:
        thr = float(a.threshold)
    elif a.checkpoint:
        thr = float(meta.get("decision_threshold") or inference._default_threshold())
    else:
        thr = inference._default_threshold()

    print("=" * 78)
    print("MINIMUM DETECTABLE SLICK — real EOS-04 ocean, real checkpoint")
    print("=" * 78)
    print(f"  raster            {raster}")
    print(f"  checkpoint file   {ckpt}")
    print(f"  ground sample     {gsd:g} m/px   ({gsd*gsd:.0f} m2 per pixel)")
    print(f"  clean ocean chips {len(chips)}  ({CHIP}x{CHIP})")
    print(f"  ocean sigma0      mean {np.mean([c.mean() for c in chips]):.2f} dB   "
          f"speckle sd {np.mean([c.std() for c in chips]):.2f} dB")
    print(f"  checkpoint        {meta.get('arch')}  "
          f"representative={meta.get('representative')}")
    print(f"  threshold         {thr:.2f}   device {device.type}")
    print(f"  detected when     >= {a.detect_frac:.0%} of planted pixels predicted oil")
    print(f"  component floor   {a.min_area_px} px = "
          f"{a.min_area_px * gsd * gsd / 1e6:.4f} km2")
    print()

    cells = {}
    for area in AREAS_KM2:
        n_px = (area * 1e6) / (gsd * gsd)
        for cdb in CONTRASTS_DB:
            hits = 0
            ious = []
            for chip in chips:
                planted, mask = plant(chip, area, cdb, gsd, rng)
                prob = segmenter.infer(planted, model, device)
                pred = prob >= thr
                # Apply the same speck rejection the pipeline uses.
                lbl, regions = inference._components(pred, a.min_area_px)
                pred = lbl > 0
                inter = float((pred & mask).sum())
                frac = inter / max(mask.sum(), 1)
                union = float((pred | mask).sum())
                ious.append(inter / union if union else 0.0)
                hits += frac >= a.detect_frac
            cells[(area, cdb)] = {
                "area_km2": area, "contrast_db": cdb,
                "planted_px": int(round(n_px)),
                "detection_rate": hits / len(chips),
                "mean_iou": float(np.mean(ious)),
            }

    # ---- table ----------------------------------------------------------
    print("DETECTION RATE  (rows: slick area, cols: contrast in dB)")
    hdr = "  area km2   px  " + "".join(f"{c:>7.0f}dB" for c in CONTRASTS_DB)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for area in AREAS_KM2:
        n_px = int(round((area * 1e6) / (gsd * gsd)))
        row = f"  {area:>8.3f} {n_px:>5d}  "
        for cdb in CONTRASTS_DB:
            r = cells[(area, cdb)]["detection_rate"]
            row += f"{r * 100:>6.0f}% "
        print(row)
    print()

    # ---- the two numbers the pitch needs --------------------------------
    solid = [c for c in cells.values() if c["detection_rate"] >= 0.8]
    smallest = min((c["area_km2"] for c in solid), default=None)
    faintest = min((c["contrast_db"] for c in solid), default=None)
    best_at_smallest = (
        min((c["contrast_db"] for c in solid if c["area_km2"] == smallest))
        if smallest is not None else None)

    print("HEADLINE (>=80% detection rate)")
    if smallest is None:
        print("  nothing reached 80% — the checkpoint or threshold is the limit,")
        print("  not the sensor. Do not quote a small-slick capability.")
    else:
        print(f"  smallest slick detected   {smallest:g} km2 "
              f"(at >= {best_at_smallest:g} dB contrast)")
        print(f"  faintest slick detected   {faintest:g} dB contrast")
        print(f"  = {smallest * 100:g} hectares, "
              f"{10.0 / smallest:,.0f}x smaller than the 10 km2 INSAT-3DS floor")
    # ---- what the same result implies at another ground sample -----------
    # The model sees PIXELS, not square kilometres, so the measured floor is
    # most honestly expressed as a pixel count and converted per product mode.
    if smallest is not None:
        floor_px = int(round((smallest * 1e6) / (gsd * gsd)))
        print("PROJECTION TO OTHER EOS-04 MODES  (not measured — see caveat)")
        print(f"  measured floor            {floor_px} px at "
              f">= {best_at_smallest:g} dB")
        for mode, g in (("MRS  (this product)", gsd), ("FRS  9 m", 9.0),
                        ("FRS  6 m", 6.0)):
            print(f"    {mode:<22} {floor_px} px = "
                  f"{floor_px * g * g / 1e6:.4f} km2")
        print("  A finer mode is the single largest available improvement and")
        print("  needs no model change. BUT it is a projection, not a result:")
        print("  a finer product is less multi-looked, so speckle rises and the")
        print("  contrast threshold rises with it. Re-run this script on a real")
        print("  FRS product before quoting an FRS number.")
        print()

    print("READ THIS BEFORE QUOTING IT")
    print("  The floor is set by CONTRAST, not area. Real oil damps capillary")
    print("  waves 5-15 dB; below ~2 dB a dark patch is not distinguishable from")
    print("  a wind shadow or a ship wake, at any size. The slicks are planted,")
    print("  the ocean and the model are real, and the model was trained on")
    print("  Sentinel-1 VV while this is EOS-04 HH — an unquantified domain shift.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "raster": str(raster), "ground_sample_m": gsd, "threshold": thr,
        "min_area_px": a.min_area_px, "detect_frac": a.detect_frac,
        "trials_per_cell": len(chips),
        "ocean_mean_db": float(np.mean([c.mean() for c in chips])),
        "ocean_speckle_sd_db": float(np.mean([c.std() for c in chips])),
        "checkpoint": {k: meta.get(k) for k in ("arch", "representative", "epoch")},
        "cells": list(cells.values()),
        "smallest_km2_at_80pct": smallest,
        "faintest_db_at_80pct": faintest,
        "caveat": ("Slicks are planted into real EOS-04 ocean by dB subtraction; "
                   "ocean, speckle and model are real. Model trained on "
                   "Sentinel-1 VV, evaluated here on EOS-04 HH."),
    }, indent=2))
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
