"""Tiled DeepLabv3+ inference — CLAUDE.md section 5.2 (layer 3).

Loads data/models/seg.pt, runs it over the scene in overlapping tiles, and
stitches the per-pixel oil probability back together with overlap averaging.

The checkpoint is optional by design. When it is absent the pipeline falls back
to the synthetic observed slick, so run_demo.sh works on a machine that has never
trained anything. `available()` is the single place that decides which path runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from samudra.detection.preprocess import (
    TILE_OVERLAP_PX,
    TILE_PX,
    land_mask,
    scene_db,
    stitch,
    tiles,
    wind_gate,
)

DEFAULT_CHECKPOINT = Path("data/models/seg.pt")

# Matches datasets.py: the model was trained on ImageNet-normalised 8-bit chips.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CLAUDE.md 5.2: input is VV in dB clipped to [-35, 0], normalised.
DB_CLIP_LO, DB_CLIP_HI = -35.0, 0.0


def available(checkpoint: Path | str = DEFAULT_CHECKPOINT) -> bool:
    """True when a trained checkpoint exists AND torch can be imported."""
    if not Path(checkpoint).exists():
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def load_model(checkpoint: Path | str = DEFAULT_CHECKPOINT, device=None):
    """Load the checkpoint. Returns (model, device, metadata)."""
    import torch

    from samudra.detection.train import DeepLabV3Plus

    ck = Path(checkpoint)
    if not ck.exists():
        raise FileNotFoundError(
            f"{ck} not found. Train with scripts/train_segmenter.py, or run the "
            f"pipeline without it — detection falls back to the synthetic slick."
        )

    # weights_only=True: this file may have come off a Colab session or a shared
    # drive, and torch.load otherwise executes arbitrary pickled code.
    blob = torch.load(ck, map_location="cpu", weights_only=True)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(num_classes=blob.get("num_classes", 2), pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()

    meta = {k: v for k, v in blob.items() if k != "model"} if isinstance(blob, dict) else {}
    return model, device, meta


def prepare_tile(db_tile: np.ndarray) -> np.ndarray:
    """dB chip -> normalised CHW float array matching training."""
    x = np.clip(db_tile, DB_CLIP_LO, DB_CLIP_HI)
    x = (x - DB_CLIP_LO) / (DB_CLIP_HI - DB_CLIP_LO)  # -> [0, 1]
    x = np.stack([x, x, x], axis=-1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1).astype(np.float32)


def infer(
    db: np.ndarray,
    model,
    device,
    tile_px: int = TILE_PX,
    overlap: int = TILE_OVERLAP_PX,
    batch: int = 4,
) -> np.ndarray:
    """Per-pixel oil probability over the whole scene."""
    import torch

    windows = list(tiles(db.shape, tile_px, overlap))
    pieces = []

    with torch.no_grad():
        for i in range(0, len(windows), batch):
            chunk = windows[i : i + batch]
            arr = np.stack([prepare_tile(db[y0:y1, x0:x1]) for y0, y1, x0, x1 in chunk])
            logits = model(torch.from_numpy(arr).to(device))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for w, p in zip(chunk, prob):
                pieces.append((w, p))

    return stitch(db.shape, pieces)


def segment(
    incident_dir: Path,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    threshold: float = 0.5,
    apply_gates: bool = True,
) -> dict:
    """Run inference over an incident's scene. Returns a result dict."""
    from datetime import datetime

    from samudra.attribution import drift

    db, transform, bounds = scene_db(incident_dir / "scene.tif")
    model, device, meta = load_model(checkpoint)
    prob = infer(db, model, device)

    gates = {}
    if apply_gates:
        gj = json.loads((incident_dir / "observed_slick.geojson").read_text())
        acq = datetime.fromisoformat(gj["features"][0]["properties"]["acquisition_at"])
        land = land_mask(bounds, db.shape, transform)
        env = drift.EnvField.load(incident_dir / "env.npz")
        gate, gates = wind_gate(env, acq.timestamp(), bounds, db.shape)
        # Oil cannot be on land, and outside the wind band the detector's output
        # is not trustworthy — suppress rather than let it vote.
        prob = prob * (~land) * gate
        gates["land_fraction"] = float(land.mean())

    mask = prob >= threshold
    return {
        "prob": prob,
        "mask": mask,
        "db": db,
        "transform": transform,
        "bounds": bounds,
        "checkpoint_meta": meta,
        "gates": gates,
        "threshold": threshold,
        "oil_fraction": float(mask.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the slick segmenter over a scene.")
    ap.add_argument("--incident", required=True)
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    d = Path(a.root) / a.incident
    if not available(a.checkpoint):
        raise SystemExit(
            f"No usable checkpoint at {a.checkpoint}. Train one with "
            f"scripts/train_segmenter.py, or run the pipeline without it."
        )

    r = segment(d, a.checkpoint, a.threshold)
    meta = r["checkpoint_meta"]
    np.save(d / "oil_prob.npy", r["prob"].astype(np.float32))

    # Record whether these weights came from a real training run. The attribution
    # stage refuses to route the demo through a smoke-test checkpoint unless
    # explicitly forced, and this is what it reads.
    (d / "detection_meta.json").write_text(
        json.dumps(
            {
                "checkpoint": str(a.checkpoint),
                "representative": bool(meta.get("representative", False)),
                "epoch": meta.get("epoch"),
                "oil_iou": meta.get("oil_iou"),
                "threshold": a.threshold,
                "oil_fraction": r["oil_fraction"],
                "gates": r["gates"],
            },
            indent=2,
        )
    )

    print(f"checkpoint        : {a.checkpoint}")
    print(f"  trained epoch   : {meta.get('epoch')}   oil IoU {meta.get('oil_iou')}")
    if meta.get("representative") is False:
        print("  !! this checkpoint came from a NON-REPRESENTATIVE run "
              "(CPU smoke test). Its output is not meaningful.")
    print(f"scene             : {r['db'].shape[1]} x {r['db'].shape[0]} px")
    if r["gates"]:
        print(f"wind gate         : {r['gates']['fraction_in_band'] * 100:.0f}% in band; "
              f"land {r['gates']['land_fraction'] * 100:.2f}%")
    print(f"oil pixels        : {r['oil_fraction'] * 100:.3f}% at threshold {a.threshold}")
    print(f"written           : {d / 'oil_prob.npy'}")


if __name__ == "__main__":
    main()
