"""Generate notebooks/demo.ipynb.

The notebook is a thin tour of the package API rather than a copy of the pipeline, so it
is generated from this script to keep it in sync and free of stale execution output.
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (
        MD,
        """# Emotion recognition — API tour

This notebook demonstrates the `fer` package: loading the dataset, inspecting it,
scoring a trained checkpoint and running a prediction on a single image.

It deliberately does **not** reimplement the pipeline. Training is a CLI command
(`fer train`), and the live webcam demo is `fer webcam` — `cv2.imshow` needs a real
window loop and hangs under Jupyter, so it does not belong in a notebook cell.

Prerequisites: `uv sync`, then `fer data` to build the dataset.""",
    ),
    (
        MD,
        """## 1. Environment

Confirm the GPU is visible and check which backbone the default config selects.""",
    ),
    (
        CODE,
        """import cv2
import onnxruntime
import timm
import torch

from fer.config import Config

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
print(f"timm {timm.__version__}  opencv {cv2.__version__}  onnxruntime {onnxruntime.__version__}")

cfg = Config.from_yaml("../configs/balanced.yaml")
print(f"\\nbackbone   {cfg.model.backbone}")
print(f"input      {cfg.data.image_size}px, label_mode={cfg.data.label_mode}")""",
    ),
    (
        MD,
        """## 2. The dataset

`fer data` merges Microsoft's FER+ crowd votes with the FER2013 pixels and caches the
result as a single `.npz`. Each image carries ten votes rather than one label, which is
what the `soft` label mode trains against.""",
    ),
    (
        CODE,
        """import numpy as np

from fer.data.ferplus import load_ferplus
from fer.labels import CLASS_NAMES

images, votes = load_ferplus("../data", split="train")
print(f"{len(images):,} training images of shape {images.shape[1:]}")

counts = np.bincount(votes.argmax(axis=1), minlength=len(CLASS_NAMES))
for name, count in sorted(zip(CLASS_NAMES, counts, strict=True), key=lambda p: -p[1]):
    bar = "#" * int(60 * count / counts.max())
    print(f"{name:>10}  {count:6,}  {count / counts.sum():5.1%}  {bar}")""",
    ),
    (
        MD,
        """That imbalance is the whole reason the pipeline reports **macro-F1** rather than
accuracy, and why the training loader uses an inverse-frequency sampler: `contempt` is
a rounding error in the dataset, and a model that never predicts it would still score
well on plain accuracy.""",
    ),
    (
        MD,
        """## 3. What the crowd votes look like

A single hard label throws away real information — plenty of faces are genuinely
ambiguous, and the vote spread captures that.""",
    ),
    (
        CODE,
        """import matplotlib.pyplot as plt

shares = votes / votes.sum(axis=1, keepdims=True)
confidence = shares.max(axis=1)

# Pick a few images the annotators disagreed about.
ambiguous = np.argsort(confidence)[:6]

fig, axes = plt.subplots(2, 6, figsize=(15, 5.5), gridspec_kw={"height_ratios": [2, 1]})
for col, idx in enumerate(ambiguous):
    axes[0, col].imshow(images[idx], cmap="gray")
    axes[0, col].axis("off")
    axes[1, col].bar(range(len(CLASS_NAMES)), shares[idx], color="#4c78a8")
    axes[1, col].set_xticks(range(len(CLASS_NAMES)))
    axes[1, col].set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    axes[1, col].set_ylim(0, 1)
    axes[1, col].tick_params(labelsize=7)
fig.suptitle("Images with the least annotator agreement, and their vote distributions")
fig.tight_layout()""",
    ),
    (
        MD,
        """## 4. Augmentation

Geometry and intensity only — the source images are grayscale, so hue and saturation
augmentation would be meaningless. Random erasing simulates occlusion.""",
    ),
    (
        CODE,
        """import torch

from fer.data.datamodule import build_train_transform

# ImageNet statistics stand in here; training reads the real values off the backbone.
transform = build_train_transform(112, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
source = torch.from_numpy(images[7]).unsqueeze(0)

fig, axes = plt.subplots(1, 8, figsize=(15, 2.2))
axes[0].imshow(images[7], cmap="gray")
axes[0].set_title("original", fontsize=9)
axes[0].axis("off")
for ax in axes[1:]:
    out = transform(source)
    ax.imshow(out.permute(1, 2, 0).clamp(0, 1).numpy()[..., 0], cmap="gray")
    ax.axis("off")
fig.suptitle("Training augmentations")
fig.tight_layout()""",
    ),
    (
        MD,
        """## 5. Score a trained checkpoint

Run `fer train -c configs/balanced.yaml` first. Evaluation uses the `test` split
(FER2013's PrivateTest), which model selection never touches.""",
    ),
    (
        CODE,
        """from pathlib import Path

from fer.evaluate import evaluate

checkpoint = Path("../runs/balanced/best.pt")
if checkpoint.exists():
    result = evaluate(checkpoint, split="test", data_root="../data")
else:
    print(f"No checkpoint at {checkpoint}. Run:  fer train -c configs/balanced.yaml")""",
    ),
    (
        MD,
        """## 6. Predict on one image

`EmotionClassifier.load` accepts either a `.pt` checkpoint or an exported `.onnx`, and
reproduces the training preprocessing in both cases.""",
    ),
    (
        CODE,
        """from fer.align import crop_face
from fer.detect import FaceDetector
from fer.infer import EmotionClassifier

image_path = Path("../test.jpg")  # point this at any photo with a face

if checkpoint.exists() and image_path.exists():
    frame = cv2.imread(str(image_path))
    faces = FaceDetector().detect(frame)
    classifier = EmotionClassifier.load(checkpoint, device="cpu")

    crops = [crop_face(frame, f.landmarks, f.box, classifier.spec.image_size) for f in faces]
    probs = classifier.predict_batch(crops)

    fig, axes = plt.subplots(1, max(2, len(faces)), figsize=(4 * max(2, len(faces)), 4))
    for ax, crop, prob in zip(np.atleast_1d(axes), crops, probs, strict=False):
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title("  ".join(f"{n} {p:.0%}" for n, p in classifier.top_k(prob, 2)), fontsize=9)
        ax.axis("off")
else:
    print("Needs both a trained checkpoint and an image at ../test.jpg")""",
    ),
    (
        MD,
        """## 7. Live webcam

Run this from a terminal, not from here:

```bash
fer export runs/balanced/best.pt          # ONNX is noticeably faster on CPU
fer webcam runs/balanced/best.onnx
```

Keys: `q` quit, `b` toggle probability bars, `m` toggle mirroring.""",
    ),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": payload}
        if kind == CODE:
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")
