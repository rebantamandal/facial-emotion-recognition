"""Fit a macro-F1-optimal decision rule on the validation split.

``argmax`` is the right decision rule for accuracy, but not for macro-F1. Macro-F1 weights
every class equally, so on a set where *contempt* is 0.6% of the data it is often worth
predicting a rare class on weaker evidence than argmax requires — the recall gained on a
tiny class moves the macro average far more than the precision lost on a large one.

This fits a per-class additive logit bias by coordinate ascent directly on macro-F1. It
is fitted on ``val`` and stored in the checkpoint, so ``test`` stays untouched by it.
The bias costs nothing at inference: it is a single vector addition before the softmax.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from fer.config import Config
from fer.data import build_dataloaders
from fer.engine import resolve_amp_dtype
from fer.models import load_checkpoint
from fer.train import pick_device

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a split, returning ``(logits, integer_labels)``.

    With ``tta`` the horizontally mirrored view is averaged in. Facial expressions are
    close to left-right symmetric, so a mirror is a legitimate second opinion rather than
    a different image.
    """
    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(images).float()
            if tta:
                mirrored = model(torch.flip(images, dims=[3])).float()
                # Average in probability space, then return to log space so the caller can
                # keep treating these as logits.
                averaged = (out.softmax(-1) + mirrored.softmax(-1)) / 2
                out = torch.log(averaged.clamp_min(1e-9))
        logits_out.append(out.cpu().numpy())
        labels_out.append(targets.numpy())
    return np.concatenate(logits_out), np.concatenate(labels_out).argmax(1)


def macro_f1(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray | None = None) -> float:
    """Macro-F1 of ``argmax(logits + bias)``."""
    scores = logits if bias is None else logits + bias
    return float(f1_score(labels, scores.argmax(1), average="macro", zero_division=0))


def fit_logit_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    rounds: int = 8,
    span: float = 3.0,
    step: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Coordinate ascent on a per-class additive bias, maximising macro-F1.

    Macro-F1 is piecewise constant in the bias, so there is no gradient to follow; a
    coordinate sweep over a bounded grid is both simple and adequate for eight classes.
    Returns ``(bias, fitted_macro_f1)``.
    """
    n_classes = logits.shape[1]
    bias = np.zeros(n_classes, dtype=np.float64)
    best = macro_f1(logits, labels, bias)
    grid = np.arange(-span, span + step / 2, step)

    for _ in range(rounds):
        improved = False
        for klass in range(n_classes):
            keep = bias[klass]
            for value in grid:
                bias[klass] = value
                score = macro_f1(logits, labels, bias)
                if score > best + 1e-9:
                    best, keep, improved = score, float(value), True
            bias[klass] = keep
        if not improved:
            break
    return bias, best


def cross_validated_gain(
    logits: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0
) -> tuple[float, list[float]]:
    """Estimate the out-of-sample macro-F1 gain from fitting a bias.

    A bias fitted on a split will always look good *on that split*. What matters is
    whether it still helps on data it did not see. Each fold fits on the rest and scores
    on itself, so the mean is an honest estimate of what the bias buys on new data.

    This matters a lot here: FER+ validation holds only a couple of dozen contempt and
    disgust images, few enough that a per-class bias can easily fit their noise.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    chunks = np.array_split(order, folds)

    gains: list[float] = []
    for i in range(folds):
        held = chunks[i]
        rest = np.concatenate([chunks[j] for j in range(folds) if j != i])
        bias, _ = fit_logit_bias(logits[rest], labels[rest])
        baseline = macro_f1(logits[held], labels[held])
        gains.append(macro_f1(logits[held], labels[held], bias) - baseline)
    return float(np.mean(gains)), gains


def calibrate(
    checkpoint: Path | str,
    split: str = "val",
    tta: bool = False,
    device_str: str = "auto",
    save: bool = True,
    folds: int = 5,
    force: bool = False,
) -> dict[str, object]:
    """Fit the decision rule on ``split`` and write it back into the checkpoint.

    Fitting on ``val`` and reporting on ``test`` is what keeps the headline number
    honest, so this deliberately refuses to fit on the test split.
    """
    if split == "test":
        raise ValueError(
            "Refusing to calibrate on the test split: fitting and reporting on the same "
            "data would inflate the result. Calibrate on 'val'."
        )

    checkpoint = Path(checkpoint)
    device = pick_device(device_str)
    model, spec, class_names, payload = load_checkpoint(checkpoint, device=device)
    model = model.to(device, memory_format=torch.channels_last)

    data_cfg = Config.from_dict(payload["config"]).data
    data_cfg.balanced_sampler = False
    loader = build_dataloaders(data_cfg, spec.mean, spec.std, splits=(split,))[split]

    logits, labels = collect_logits(
        model, loader, device, resolve_amp_dtype("bf16", device), tta=tta
    )
    before = macro_f1(logits, labels)
    bias, after = fit_logit_bias(logits, labels)

    logger.info("In-sample on %r: macro-F1 %.4f -> %.4f", split, before, after)
    for name, value in zip(class_names, bias, strict=True):
        if abs(value) > 1e-6:
            logger.info("  %-11s bias %+.2f", name, value)

    mean_gain, fold_gains = cross_validated_gain(logits, labels, folds=folds)
    logger.info(
        "Cross-validated gain: %+.4f macro-F1 (folds %s)",
        mean_gain,
        ", ".join(f"{g:+.3f}" for g in fold_gains),
    )

    generalises = mean_gain > 0
    if not generalises:
        logger.warning(
            "The fitted bias does not survive cross-validation: it gains %.4f in-sample "
            "but %+.4f held out. That is fitting noise, not a real prior correction — "
            "expect it to hurt on the test split. Not saving.",
            after - before,
            mean_gain,
        )
        if force:
            logger.warning("Saving anyway because force=True was passed.")

    if save and (generalises or force):
        payload["logit_bias"] = bias.tolist()
        payload["calibration"] = {
            "split": split,
            "tta": tta,
            "macro_f1_before": before,
            "macro_f1_after": after,
            "cv_mean_gain": mean_gain,
            "cv_fold_gains": fold_gains,
        }
        torch.save(payload, checkpoint)
        logger.info("Wrote calibration into %s", checkpoint)

    elif save:
        # Clear any bias a previous run stored, so a rejected fit is not left applied.
        if payload.pop("logit_bias", None) is not None:
            payload.pop("calibration", None)
            torch.save(payload, checkpoint)
            logger.info("Removed the previously stored bias from %s", checkpoint)

    return {
        "bias": bias.tolist(),
        "macro_f1_before": before,
        "macro_f1_after": after,
        "cv_mean_gain": mean_gain,
        "generalises": generalises,
        "class_names": class_names,
    }
