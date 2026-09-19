"""Evaluate a trained checkpoint on a held-out split.

FER+ has two test sets. ``val`` (FER2013's PublicTest) is what training selects on, so
the honest headline number comes from ``test`` (PrivateTest), which nothing has ever
been tuned against.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from fer.config import Config, DataConfig
from fer.data import build_dataloaders
from fer.engine import predict, resolve_amp_dtype
from fer.metrics import EvalResult, compute_metrics, plot_confusion
from fer.models import load_checkpoint
from fer.train import pick_device

logger = logging.getLogger(__name__)


def evaluate(
    checkpoint: Path | str,
    split: str = "test",
    data_root: Path | str | None = None,
    batch_size: int | None = None,
    device_str: str = "auto",
    out_dir: Path | str | None = None,
    tta: bool = False,
    use_calibration: bool = True,
) -> EvalResult:
    """Score a checkpoint and, when ``out_dir`` is given, write the report and figure."""
    device = pick_device(device_str)
    model, spec, class_names, payload = load_checkpoint(checkpoint, device=device)
    model = model.to(device, memory_format=torch.channels_last)

    # Reuse the data settings the model was trained with, overriding only what was asked.
    data_cfg: DataConfig = Config.from_dict(payload["config"]).data
    if data_root is not None:
        data_cfg.root = Path(data_root)
    if batch_size is not None:
        data_cfg.batch_size = batch_size
    data_cfg.balanced_sampler = False
    data_cfg.image_size = spec.image_size
    data_cfg.grayscale = spec.grayscale

    loader = build_dataloaders(data_cfg, spec.mean, spec.std, splits=(split,))[split]
    amp = resolve_amp_dtype("bf16", device)

    bias = payload.get("logit_bias") if use_calibration else None
    if bias is not None:
        logger.info("Applying fitted logit bias from %s", Path(checkpoint).name)

    if tta or bias is not None:
        # Both paths need raw logits rather than probabilities.
        from fer.calibrate import collect_logits

        logits, labels = collect_logits(model, loader, device, amp, tta=tta)
        if bias is not None:
            logits = logits + np.asarray(bias, dtype=np.float32)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        targets = labels
        loss = float("nan")
    else:
        probs, targets, loss = predict(model, loader, device, amp)
    result = compute_metrics(probs, targets, class_names)

    logger.info("Split %r loss %.4f", split, loss)
    print(result.format_table())

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion(result.confusion, class_names, out_dir / f"confusion_{split}.png")
        (out_dir / f"metrics_{split}.json").write_text(
            json.dumps(
                {"loss": loss, **result.summary(), "per_class": result.per_class}, indent=2
            ),
            encoding="utf-8",
        )
        logger.info("Report written to %s", out_dir)
    return result
