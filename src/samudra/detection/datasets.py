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
                    Class_0 3695, Class_1 1843 on disk.

CLASS SEMANTICS OF THE BINARY SET ARE VERIFIED, NOT INFERRED FROM APPEARANCE.
`data/raw/oilspill/binary/metadata/dataset_metadata.xml` is the publisher's own
Dublin Core record and states it outright:

    "0" for chips not containing any oil features (look-alikes or clean seas)
    "1" for those containing oil features

  Collection : CSIRO Sentinel-1 SAR image dataset of oil- and non-oil features
               for machine learning, Blondeau-Patissier et al., CSIRO, 2022-12-15
  DOI        : 10.25919/4v55-dn16
  Published  : N=5,630 (3,725 class 0 / 1,905 class 1). On disk: 5,538 outside
               the sample/ folder, so this copy is ~92 chips short of the
               published count — reported rather than papered over.
  Coverage   : 101.4E-154.8E, 26.1S-21.1N, 2015-05-01 to 2022-08-31.
               That is Southeast Asia and Australia, NOT the Arabian Sea. Any
               claim that these chips depict Indian waters would be false.

Deciding the labels by eye ("the dark ones must be oil") would have been wrong
in the other direction too: class 0 deliberately contains look-alikes, which are
also dark.

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

import hashlib
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


def _val_bucket(sensor: str, name: str) -> int:
    """Stable 0-999 bucket for a chip, from its identity rather than its order.

    Hashing the path rather than shuffling a list means the train/val partition
    is identical on every machine and every run, and does not move when `limit`
    changes or when files are added. A seeded shuffle gives neither guarantee.
    """
    h = hashlib.sha1(f"{sensor}/{name}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 1000


#: Oil-fraction bands and the share of each training batch they should occupy
#: when small-slick boosting is on. Left column is the band's natural share of
#: the SOS train split, measured over all 5488 chips:
#:
#:     empty  3.8%   tiny 2.3%   small 9.1%   medium 27.1%   large 34.2%   huge 23.6%
#:
#: Mean oil fraction is 24.7% — a quarter of every chip. A model trained on that
#: learns "large dark region = oil", which is exactly the wrong prior for the
#: 150-pixel slicks the detection-limit measurement sits at. The targets below
#: are NOT an estimate of how often each size occurs at sea; they deliberately
#: over-represent the hard cases so the network has to learn them.
#:
#: Empty chips are kept at roughly twice their natural share rather than dropped:
#: they are the only negatives in the set, and boosting small positives without
#: them trades recall for precision.
SMALL_SLICK_BANDS: tuple[tuple[float, float, str, float], ...] = (
    (-1.0, 0.0, "empty", 0.08),
    (0.0, 0.01, "tiny", 0.18),
    (0.01, 0.05, "small", 0.24),
    (0.05, 0.15, "medium", 0.24),
    (0.15, 0.35, "large", 0.16),
    (0.35, 1.01, "huge", 0.10),
)


def small_slick_weights(fractions: np.ndarray,
                        bands=SMALL_SLICK_BANDS) -> tuple[np.ndarray, list[dict]]:
    """Per-chip sampler weights that hit the band targets exactly.

    A chip in band b gets weight `target_b / count_b`, so the expected share of
    draws from band b is exactly `target_b` regardless of how many chips it
    holds. Empty bands are skipped and the remaining targets renormalised, which
    keeps the weights valid on a `--train-limit` subset that happens to contain
    no chips of some size.
    """
    fr = np.asarray(fractions, dtype=np.float64)
    w = np.zeros(len(fr), dtype=np.float64)
    report: list[dict] = []

    present = []
    for lo, hi, name, target in bands:
        m = (fr == 0.0) if hi == 0.0 else ((fr > lo) & (fr <= hi))
        if m.any():
            present.append((m, name, target))
    total_target = sum(t for _, _, t in present) or 1.0

    for m, name, target in present:
        share = target / total_target
        w[m] = share / int(m.sum())
        report.append({
            "band": name,
            "chips": int(m.sum()),
            "natural_share": round(float(m.mean()), 4),
            "sampled_share": round(float(share), 4),
            "boost": round(float(share / max(m.mean(), 1e-9)), 2),
        })

    if w.sum() <= 0:  # nothing matched; fall back to uniform rather than zeros
        return np.ones(len(fr)) / max(len(fr), 1), report
    return w / w.sum(), report


class SOSSegmentation(Dataset):
    """Deep-SAR SOS image/label pairs. The primary segmentation trainer.

    Splits
    ------
    `test`          the dataset's own published test directory. Held out
                    completely: touched once, at the end, by the final
                    evaluation. Never used for checkpoint selection.
    `train`/`val`   a deterministic partition of the published train
                    directory (see `_val_bucket`). Validation drives early
                    stopping and checkpoint selection so that the test split
                    stays genuinely unseen.

    KNOWN LIMITATION, stated rather than hidden: SOS chips are named by bare
    index with no parent-scene identifier, so chips cut from the same original
    SAR scene can land on both sides of the train/val partition. Validation IoU
    is therefore optimistic relative to a truly unseen scene. The published test
    split is the number to quote; val exists to choose a checkpoint without
    touching it.
    """

    SENSORS = ("palsar", "sentinel")

    def __init__(
        self,
        root: Path | str = "data/raw/oilspill/sos",
        split: str = "train",
        sensors: tuple[str, ...] = SENSORS,
        augment: bool = True,
        limit: int | None = None,
        seed: int = 0,
        val_frac: float = 0.15,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train, val or test, not {split!r}")
        self.root = Path(root)
        self.augment = augment
        self.split = split
        self.val_frac = val_frac
        self.rng = random.Random(seed)

        src_dir = "test" if split == "test" else "train"
        cut = int(round(val_frac * 1000))

        self.items: list[tuple[Path, Path, str]] = []
        for sensor in sensors:
            imgd = self.root / src_dir / sensor / "image"
            labd = self.root / src_dir / sensor / "label"
            if not imgd.is_dir():
                raise FileNotFoundError(f"{imgd} not found — run the inventory step")
            for f in sorted(imgd.glob("*.png")):
                lab = labd / f.name
                if not lab.exists():  # never guess a missing label
                    continue
                if split != "test":
                    in_val = _val_bucket(sensor, f.name) < cut
                    if (split == "val") != in_val:
                        continue
                self.items.append((f, lab, sensor))

        if not self.items:
            raise RuntimeError(f"no image/label pairs for split {split!r} under {self.root}")

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

    def oil_pixel_fraction(self, sample: int = 400,
                           weights: np.ndarray | None = None) -> float:
        """Measured oil fraction, used to weight the segmentation loss.

        `weights` is the sampler's per-chip weight. Pass it whenever the loader
        does not sample uniformly: boosting small slicks lowers the oil fraction
        the network actually sees, and a class weight computed on the uniform
        distribution would then be wrong in the direction that hurts most.
        """
        if weights is not None:
            fr = self.oil_fractions()
            w = np.asarray(weights, dtype=np.float64)
            return float((fr * w).sum() / max(w.sum(), 1e-12))
        idx = np.linspace(0, len(self.items) - 1, min(sample, len(self.items))).astype(int)
        tot = oil = 0
        for i in idx:
            m = np.asarray(Image.open(self.items[i][1]).convert("L"))
            oil += int((m >= MASK_THRESHOLD).sum())
            tot += m.size
        return oil / max(tot, 1)

    def oil_fractions(self, cache: bool = True) -> np.ndarray:
        """Oil fraction of every chip, in `self.items` order.

        Reading 5488 labels takes a while, and the answer only changes when the
        files do, so it is cached beside the dataset and keyed on the identity
        of the item list — not on the split name, which would collide when
        `limit` is in use.
        """
        key = hashlib.sha1(
            "|".join(f"{p.name}:{s}" for p, _, s in self.items).encode()
        ).hexdigest()[:16]
        cache_path = self.root / f".oilfrac_{self.split}_{key}.npy"
        if cache and cache_path.exists():
            try:
                arr = np.load(cache_path)
                if len(arr) == len(self.items):
                    return arr
            except Exception:  # noqa: BLE001 - corrupt cache, just recompute
                pass

        out = np.empty(len(self.items), dtype=np.float32)
        for i, (_, lab_p, _) in enumerate(self.items):
            m = np.asarray(Image.open(lab_p).convert("L"))
            out[i] = (m >= MASK_THRESHOLD).mean()
        if cache:
            try:
                np.save(cache_path, out)
            except OSError:
                pass
        return out


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
