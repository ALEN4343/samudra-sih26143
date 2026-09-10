"""DeepLabv3+ training and evaluation — CLAUDE.md 5.2.

torchvision ships DeepLab**v3**, not v3+. The difference is not cosmetic: v3
upsamples the ASPP output straight to full resolution, which blurs boundaries,
while v3+ adds a decoder that fuses high-level ASPP features with low-level
layer1 features before upsampling. Slick boundary geometry is exactly what the
attribution layer consumes — `major_axis_deg`, `eccentricity` and
`shape_complexity` are all boundary-derived — so the decoder is built here
rather than silently substituting v3 and calling it v3+.

An auxiliary image-level classification head shares the backbone, carrying the
binary dataset's per-image labels. See datasets.py for why that set is not mixed
into the segmentation supervision.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from urllib.error import URLError

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP


class DeepLabV3Plus(nn.Module):
    """ResNet-50 + ASPP + v3+ decoder, with an auxiliary classification head."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, aux_classes: int = 2):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        # output_stride 16: dilate the last block instead of downsampling again,
        # keeping 1/16 resolution for the ASPP.
        try:
            backbone = resnet50(
                weights=weights, replace_stride_with_dilation=[False, False, True]
            )
        except URLError as exc:
            raise RuntimeError(
                "\n".join(
                    [
                        "Could not download the ImageNet weights for the ResNet-50 backbone.",
                        f"  {exc}",
                        "This machine sits behind TLS interception, so the download fails",
                        "certificate verification. Options, in order of preference:",
                        "  1. Train on Colab (notebooks/train_colab.ipynb) - clean network,",
                        "     and you need the GPU for a real run anyway.",
                        "  2. Pre-place the file in the torch hub cache",
                        "     (~/.cache/torch/hub/checkpoints/).",
                        "  3. Pass --no-pretrained to train from random initialisation.",
                        "     Fine for a smoke test; materially worse for a real run.",
                        "Certificate verification is NOT disabled automatically: torch.load",
                        "executes pickled data, so fetching weights over an unverified",
                        "channel is a supply-chain risk and has to be a deliberate choice.",
                    ]
                )
            ) from exc

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 256 ch, stride 4  -> low-level features
        self.layer2 = backbone.layer2  # 512 ch, stride 8
        self.layer3 = backbone.layer3  # 1024 ch, stride 16
        self.layer4 = backbone.layer4  # 2048 ch, stride 16 (dilated)

        self.aspp = ASPP(2048, [6, 12, 18], out_channels=256)

        self.low_level = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

        # Auxiliary image-level head for the binary dataset.
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(2048, aux_classes),
        )

    def forward(self, x: torch.Tensor, aux_only: bool = False):
        size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        deep = self.layer4(x)

        if aux_only:
            return self.aux_head(deep)

        y = self.aspp(deep)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.decoder(torch.cat([y, self.low_level(low)], dim=1))
        y = self.classifier(y)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def confusion(pred: np.ndarray, target: np.ndarray, k: int = 2) -> np.ndarray:
    m = (target >= 0) & (target < k)
    return np.bincount(k * target[m].astype(int) + pred[m].astype(int),
                       minlength=k * k).reshape(k, k)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Per-class IoU, precision, recall from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
    names = ["background", "oil"]
    return {
        "confusion_matrix": cm.tolist(),
        "per_class": {
            names[i]: {
                "iou": float(np.nan_to_num(iou[i])),
                "precision": float(np.nan_to_num(prec[i])),
                "recall": float(np.nan_to_num(rec[i])),
                "support_px": int(cm[i].sum()),
            }
            for i in range(len(names))
        },
        "mean_iou": float(np.nanmean(iou)),
        "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
    }


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------


def pick_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        print(f"device            : cuda ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("=" * 78)
    print("!! NO CUDA GPU FOUND — TRAINING ON CPU")
    print("!! Any metrics produced here come from a reduced subset and a reduced")
    print("!! epoch count. They exist to prove the pipeline runs end to end.")
    print("!! THEY ARE NOT REPRESENTATIVE OF MODEL PERFORMANCE. Run the full")
    print("!! training on a GPU (notebooks/train_colab.ipynb) before quoting any")
    print("!! number from this run.")
    print("=" * 78)
    return torch.device("cpu")


def train_one_epoch(
    model, seg_loader, opt, device, class_weights, aux_loader=None,
    aux_weight: float = 0.4, log_every: int = 20,
) -> dict:
    model.train()
    seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    aux_loss_fn = nn.CrossEntropyLoss()
    aux_iter = iter(aux_loader) if aux_loader is not None else None

    tot_seg = tot_aux = 0.0
    n = 0
    t0 = time.time()
    for i, batch in enumerate(seg_loader):
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        opt.zero_grad(set_to_none=True)
        out = model(img)
        loss_seg = seg_loss_fn(out, mask)
        loss = loss_seg
        loss_aux_v = 0.0

        if aux_iter is not None:
            try:
                ab = next(aux_iter)
            except StopIteration:
                aux_iter = iter(aux_loader)
                ab = next(aux_iter)
            logits = model(ab["image"].to(device), aux_only=True)
            loss_aux = aux_loss_fn(logits, ab["label"].to(device))
            loss = loss + aux_weight * loss_aux
            loss_aux_v = float(loss_aux.item())

        loss.backward()
        opt.step()

        tot_seg += float(loss_seg.item())
        tot_aux += loss_aux_v
        n += 1
        if log_every and i % log_every == 0:
            print(f"    batch {i:>4}/{len(seg_loader)}  seg {loss_seg.item():.4f}"
                  + (f"  aux {loss_aux_v:.4f}" if aux_iter else ""), flush=True)

    return {
        "seg_loss": tot_seg / max(n, 1),
        "aux_loss": tot_aux / max(n, 1),
        "seconds": time.time() - t0,
    }


@torch.no_grad()
def evaluate(model, loader, device, per_sensor: bool = True) -> dict:
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    by_sensor: dict[str, np.ndarray] = {}

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].numpy()
        pred = model(img).argmax(1).cpu().numpy()
        cm += confusion(pred.ravel(), mask.ravel())

        if per_sensor:
            for k, s in enumerate(batch["sensor"]):
                by_sensor.setdefault(s, np.zeros((2, 2), dtype=np.int64))
                by_sensor[s] += confusion(pred[k].ravel(), mask[k].ravel())

    out = metrics_from_confusion(cm)
    if per_sensor:
        # Reported separately because PALSAR is L-band and Sentinel-1 is C-band:
        # a model that scores well on one and badly on the other has learned a
        # sensor, not oil.
        out["per_sensor"] = {s: metrics_from_confusion(c) for s, c in by_sensor.items()}
    return out


@torch.no_grad()
def evaluate_aux(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    for batch in loader:
        logits = model(batch["image"].to(device), aux_only=True)
        pred = logits.argmax(1).cpu().numpy()
        cm += confusion(pred, batch["label"].numpy())
    m = metrics_from_confusion(cm)
    m["per_class"] = {
        "no_oil": m["per_class"]["background"],
        "oil": m["per_class"]["oil"],
    }
    return m


def plot_metrics(metrics: dict, out_png: Path) -> Path:
    """Confusion matrix and per-class IoU as a single figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seg = metrics["segmentation"]
    cm = np.array(seg["confusion_matrix"], dtype=float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)

    ax = axes[0]
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    names = ["background", "oil"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Segmentation confusion (row-normalised)", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i, j]:.3f}\n{int(cm[i, j]):,}",
                    ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#111418", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    labels, vals = [], []
    for n in names:
        labels.append(n)
        vals.append(seg["per_class"][n]["iou"])
    for s, m in sorted(seg.get("per_sensor", {}).items()):
        labels.append(f"oil ({s})")
        vals.append(m["per_class"]["oil"]["iou"])
    bars = ax.bar(labels, vals, color=["#8b949e", "#b3261e"] + ["#1f4e8c"] * (len(vals) - 2))
    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU")
    ax.set_title(f"Per-class IoU   (mean {seg['mean_iou']:.3f})", fontsize=10)
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    if not metrics.get("representative", True):
        fig.suptitle("NOT REPRESENTATIVE — CPU smoke test on a reduced subset",
                     color="#b3261e", fontsize=11, y=1.02)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def save_metrics(metrics: dict, out_json: Path, out_png: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    plot_metrics(metrics, out_png)
