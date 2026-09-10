"""Dataset loaders for the two open SAR oil-spill sets — CLAUDE.md 5.2, 7.

The two sets are structurally unrelated and are loaded separately. Measured
properties, not assumed ones:

  Deep-SAR SOS      256x256 PNG, greyscale stored as RGB, image/label pairs.
                    train: 3101 palsar + 3354 sentinel;  test: 776 + 839.
                    palsar masks are clean binary {0, 255}.
                    sentinel masks carry intermediate values on 1.44% of pixels,
                    98.3% of which lie on the mask boundary — anti-aliasing, not
                    extra classes. Both threshold cleanly at 128.
                    Oil fraction: palsar 15.7%, sentinel 34.5%.

  Sentinel-1 binary 400x400 JPEG, per-IMAGE labels only, no masks.
                    Class_0 3695 (no oil), Class_1 1843 (oil).

Neither SOS sensor encodes an oil-vs-lookalike distinction, so the confusion
matrix this supports is oil vs background. That was checked, not assumed.

HOW THE BINARY SET IS USED, and why. It supplies an auxiliary image-level
classification signal through a separate head, and is NOT mixed into the
segmentation set as all-background negatives. Class_0 does technically imply a
valid all-zero mask, but the two sets are visibly different domains: intensity
std is 8-12 for the binary set against 31-49 for SOS. Training segmentation
across that gap lets the network satisfy the loss by recognising which dataset
an image came from rather than where the oil is — a shortcut that scores well
and learns nothing. Class_1 is unusable for segmentation in any case: it says
oil is present somewhere, not where, and inventing a mask from that would be
fabricating labels.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet statistics: the ResNet-50 backbone is pretrained, so its input
# distribution is what the first convolutions expect. SAR greyscale is
# replicated to three channels.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MASK_THRESHOLD = 128  # sentinel masks are anti-aliased; palsar is already {0,255}


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxW uint8 greyscale -> normalised 3xHxW float tensor."""
    x = img.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=-1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def _augment(img: np.ndarray, mask: np.ndarray | None, rng: random.Random):
    """Flips and 90-degree rotations.

    SAR backscatter has no canonical orientation — a slick is equally plausible
    at any bearing — so the dihedral group is label-preserving here in a way it
    would not be for natural images.
    """
    if rng.random() < 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask) if mask is not None else None
    if rng.random() < 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask) if mask is not None else None
    k = rng.randint(0, 3)
    if k:
        img = np.rot90(img, k)
        mask = np.rot90(mask, k) if mask is not None else None
    return np.ascontiguousarray(img), (
        np.ascontiguousarray(mask) if mask is not None else None
    )


class SOSSegmentation(Dataset):
    """Deep-SAR SOS image/label pairs. The primary segmentation trainer."""

    SENSORS = ("palsar", "sentinel")

    def __init__(
        self,
        root: Path | str = "data/raw/oilspill/sos",
        split: str = "train",
        sensors: tuple[str, ...] = SENSORS,
        augment: bool = True,
        limit: int | None = None,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.augment = augment
        self.rng = random.Random(seed)

        self.items: list[tuple[Path, Path, str]] = []
        for sensor in sensors:
            imgd = self.root / split / sensor / "image"
            labd = self.root / split / sensor / "label"
            if not imgd.is_dir():
                raise FileNotFoundError(f"{imgd} not found — run the inventory step")
            for f in sorted(imgd.glob("*.png")):
                lab = labd / f.name
                if lab.exists():  # never guess a missing label
                    self.items.append((f, lab, sensor))

        if not self.items:
            raise RuntimeError(f"no image/label pairs under {self.root / split}")

        if limit is not None and limit < len(self.items):
            # Stratified by sensor so a subset keeps both domains.
            per = {s: [] for s in sensors}
            for it in self.items:
                per[it[2]].append(it)
            take = max(1, limit // max(len(sensors), 1))
            picked = []
            r = random.Random(seed)
            for s in sensors:
                r.shuffle(per[s])
                picked += per[s][:take]
            self.items = picked[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def sensor_of(self, i: int) -> str:
        return self.items[i][2]

    def __getitem__(self, i: int):
        img_p, lab_p, sensor = self.items[i]
        img = np.asarray(Image.open(img_p).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(lab_p).convert("L"), dtype=np.uint8)
        mask = (mask >= MASK_THRESHOLD).astype(np.int64)

        if self.augment:
            img, mask = _augment(img, mask, self.rng)

        return {
            "image": _to_tensor(img),
            "mask": torch.from_numpy(mask.copy()).long(),
            "sensor": sensor,
        }

    def oil_pixel_fraction(self, sample: int = 400) -> float:
        """Measured oil fraction, used to weight the segmentation loss."""
        idx = np.linspace(0, len(self.items) - 1, min(sample, len(self.items))).astype(int)
        tot = oil = 0
        for i in idx:
            m = np.asarray(Image.open(self.items[i][1]).convert("L"))
            oil += int((m >= MASK_THRESHOLD).sum())
            tot += m.size
        return oil / max(tot, 1)


class BinaryClassification(Dataset):
    """Sentinel-1 binary oil/no-oil. Auxiliary image-level signal only."""

    def __init__(
        self,
        root: Path | str = "data/raw/oilspill/binary/data",
        augment: bool = True,
        limit: int | None = None,
        size: int = 256,
        seed: int = 0,
        split: str = "train",
        val_frac: float = 0.2,
    ):
        self.root = Path(root)
        self.augment = augment
        self.size = size
        self.rng = random.Random(seed)

        items: list[tuple[Path, int]] = []
        for cls, label in (("Class_0", 0), ("Class_1", 1)):
            d = self.root / cls
            if not d.is_dir():
                raise FileNotFoundError(f"{d} not found — run the inventory step")
            items += [(f, label) for f in sorted(d.glob("*.jpg"))]

        # This set ships no split of its own, so make a deterministic one.
        r = random.Random(seed)
        r.shuffle(items)
        cut = int(len(items) * (1 - val_frac))
        self.items = items[:cut] if split == "train" else items[cut:]

        if limit is not None and limit < len(self.items):
            self.items = self.items[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, label = self.items[i]
        img = Image.open(p).convert("L")
        if img.size != (self.size, self.size):
            # 400x400 down to the segmentation input size so one backbone serves
            # both heads.
            img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if self.augment:
            arr, _ = _augment(arr, None, self.rng)
        return {"image": _to_tensor(arr), "label": torch.tensor(label).long()}

    @property
    def class_counts(self) -> dict[int, int]:
        out = {0: 0, 1: 0}
        for _, l in self.items:
            out[l] += 1
        return out
