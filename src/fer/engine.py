"""Training and evaluation loops.

Everything operates on soft targets, which keeps one loss function covering FER+ vote
distributions, one-hot labels, label smoothing and mixup/cutmix without special cases.
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and prefer deterministic cuDNN kernel choices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Not full determinism: benchmark=True lets cuDNN autotune, which is a large speed win
    # and only affects kernel selection, not correctness.
    torch.backends.cudnn.benchmark = True


def resolve_amp_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    """Pick the best available autocast dtype, degrading rather than failing.

    bf16 needs Ampere or newer. On older cards we fall back to fp16 (which then needs a
    GradScaler), and on CPU we disable autocast entirely.
    """
    if device.type != "cuda":
        return None
    if requested == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        logger.warning("bf16 unsupported on this GPU; falling back to fp16.")
        return torch.float16
    if requested == "fp16":
        return torch.float16
    return None


def _autocast(device: torch.device, dtype: torch.dtype | None):
    """Autocast context that is a genuine no-op when no AMP dtype was resolved."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def soft_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy against a probability vector, with optional smoothing.

    ``F.cross_entropy`` accepts soft targets but applies ``label_smoothing`` only to the
    hard-label path, so smoothing is folded into the target here instead.
    """
    if label_smoothing > 0.0:
        n_classes = targets.size(-1)
        targets = targets * (1.0 - label_smoothing) + label_smoothing / n_classes
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    """Random box covering ``1 - lam`` of the image, for CutMix."""
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = random.randrange(height), random.randrange(width)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    return y1, y2, x1, x2


def mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
    prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup or cutmix to a batch, mixing the soft targets by the same factor.

    Returns the batch unchanged with probability ``1 - prob``; otherwise picks one of the
    two schemes at random. Both are strong regularisers on a dataset this small and noisy.
    """
    if prob <= 0.0 or random.random() > prob:
        return images, targets
    use_cutmix = cutmix_alpha > 0.0 and (mixup_alpha <= 0.0 or random.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha <= 0.0:
        return images, targets

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)

    if use_cutmix:
        y1, y2, x1, x2 = _rand_bbox(images.size(-2), images.size(-1), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        # Recompute lam from the box actually used, which rounding may have shrunk.
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(-2) * images.size(-1)))
    else:
        images = images.mul(lam).add_(images[perm].mul(1.0 - lam))

    targets = targets.mul(lam).add_(targets[perm].mul(1.0 - lam))
    return images, targets


class ModelEma:
    """Exponential moving average of the weights.

    The averaged weights are consistently a little better than the live ones and are what
    gets exported, so the EMA copy is what the evaluation loop scores.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.module = self._clone(model)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _clone(model: nn.Module) -> nn.Module:
        import copy

        clone = copy.deepcopy(model).eval()
        return clone

    @staticmethod
    def _strip(name: str) -> str:
        """Drop the prefix ``torch.compile`` adds, so compiled models still match."""
        return name.removeprefix("_orig_mod.")

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # Warm up the decay so early epochs are not dragged toward the random init.
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        ema_params = dict(self.module.named_parameters())
        for name, param in model.named_parameters():
            ema_params[self._strip(name)].lerp_(param.detach().float(), 1.0 - decay)
        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[self._strip(name)].copy_(buffer)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    min_lr: float,
    base_lrs: list[float],
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup into cosine decay, stepped per optimiser step."""
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        # Floor each group at min_lr relative to its own base LR.
        floor = min_lr / max(base_lrs)
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler | None,
    ema: ModelEma | None,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    clip_grad: float,
    global_step: int,
    progress: Iterable | None = None,
) -> tuple[float, int]:
    """Run one training epoch. Returns ``(mean_loss, new_global_step)``."""
    model.train()
    total_loss, n_batches = 0.0, 0

    iterator = progress if progress is not None else loader
    for images, targets in iterator:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        images, targets = mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha, mixup_prob)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_dtype):
            loss = soft_cross_entropy(model(images), targets, label_smoothing)

        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        scheduler.step()
        global_step += 1
        if ema is not None:
            ema.update(model, global_step)

        total_loss += loss.item()
        n_batches += 1
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{total_loss / n_batches:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    return total_loss / max(1, n_batches), global_step


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    label_smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the model over a loader. Returns ``(probs, targets, mean_loss)``."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss, n_batches = 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets_dev = targets.to(device, non_blocking=True)
        with _autocast(device, amp_dtype):
            logits = model(images)
        total_loss += soft_cross_entropy(logits.float(), targets_dev, label_smoothing).item()
        n_batches += 1
        all_probs.append(logits.float().softmax(dim=-1).cpu().numpy())
        all_targets.append(targets.numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_targets),
        total_loss / max(1, n_batches),
    )
