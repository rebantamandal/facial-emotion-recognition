"""Training entry point.

Model selection is on validation macro-F1 rather than accuracy, because FER+ is skewed
enough that accuracy rewards a model for ignoring the rare emotions entirely. The EMA
weights are what gets scored and saved.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

from fer.config import Config
from fer.data import build_dataloaders
from fer.data.ferplus import build_ferplus
from fer.engine import (
    ModelEma,
    build_scheduler,
    predict,
    resolve_amp_dtype,
    seed_everything,
    train_one_epoch,
)
from fer.labels import CLASS_NAMES
from fer.metrics import compute_metrics, plot_confusion, plot_history
from fer.models import create_model, param_groups, preprocess_spec, save_checkpoint

logger = logging.getLogger(__name__)


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, then Apple MPS, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _try_compile(model: torch.nn.Module, image_size: int, device: torch.device):
    """Compile the model, falling back to eager if the backend cannot actually run.

    ``torch.compile`` is lazy: it returns a wrapper immediately and only compiles on the
    first forward pass, so a missing backend (Triton is commonly unavailable on Windows)
    surfaces mid-training rather than here. A throwaway forward pass forces that work to
    happen now, where it can be caught and downgraded to eager.
    """
    try:
        compiled = torch.compile(model)
        sample = torch.zeros(2, 3, image_size, image_size, device=device).to(
            memory_format=torch.channels_last
        )
        was_training = compiled.training
        compiled.eval()
        with torch.no_grad():
            compiled(sample)
        compiled.train(was_training)
    except Exception as exc:  # noqa: BLE001 - compilation is an optimisation, not a requirement
        logger.warning(
            "torch.compile unavailable (%s: %s); continuing in eager mode.",
            type(exc).__name__,
            str(exc).splitlines()[0][:160],
        )
        torch._dynamo.reset()
        return model
    logger.info("torch.compile enabled")
    return compiled


def train(cfg: Config, device_str: str = "auto") -> dict[str, float]:
    """Run the full training loop and return the best validation metrics."""
    seed_everything(cfg.train.seed)
    device = pick_device(device_str)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.yaml")
    logger.info("Device: %s | output: %s", device, out_dir)

    build_ferplus(cfg.data.root)

    model = create_model(cfg.model)
    spec = preprocess_spec(model, cfg.data.image_size, cfg.data.grayscale)
    logger.info("Input %dpx, normalised with mean=%s std=%s", spec.image_size, spec.mean, spec.std)
    model = model.to(device, memory_format=torch.channels_last)

    loaders = build_dataloaders(cfg.data, spec.mean, spec.std, splits=("train", "val"))
    train_loader, val_loader = loaders["train"], loaders["val"]

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.train.weight_decay, cfg.train.backbone_lr_mult, cfg.train.lr),
        lr=cfg.train.lr,
        betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=cfg.train.epochs,
        steps_per_epoch=max(1, len(train_loader)),
        warmup_epochs=cfg.train.warmup_epochs,
        min_lr=cfg.train.min_lr,
        base_lrs=[g["lr"] for g in optimizer.param_groups],
    )

    amp_dtype = resolve_amp_dtype(cfg.train.amp_dtype, device)
    # Only fp16 needs loss scaling; bf16 has the dynamic range to go without.
    scaler = torch.amp.GradScaler(device.type) if amp_dtype is torch.float16 else None
    ema = ModelEma(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None

    if cfg.train.compile:
        model = _try_compile(model, cfg.data.image_size, device)

    history: list[dict[str, float]] = []
    best_f1, best_epoch, global_step = -1.0, -1, 0
    started = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        bar = tqdm(train_loader, desc=f"epoch {epoch:3d}/{cfg.train.epochs}", leave=False)
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            ema=ema,
            label_smoothing=cfg.train.label_smoothing,
            mixup_alpha=cfg.train.mixup_alpha,
            cutmix_alpha=cfg.train.cutmix_alpha,
            mixup_prob=cfg.train.mixup_prob,
            clip_grad=cfg.train.clip_grad,
            global_step=global_step,
            progress=bar,
        )

        # Score the averaged weights: they are what will be exported.
        eval_model = ema.module if ema is not None else model
        probs, targets, val_loss = predict(eval_model, val_loader, device, amp_dtype)
        result = compute_metrics(probs, targets)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": result.accuracy,
            "val_macro_f1": result.macro_f1,
            "val_balanced_accuracy": result.balanced_accuracy,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        logger.info(
            "epoch %3d  train %.4f  val %.4f  acc %.4f  macro-F1 %.4f  bal-acc %.4f",
            epoch, train_loss, val_loss, result.accuracy, result.macro_f1,
            result.balanced_accuracy,
        )

        if result.macro_f1 > best_f1:
            best_f1, best_epoch = result.macro_f1, epoch
            save_checkpoint(
                out_dir / "best.pt", eval_model, cfg, spec, result.summary(), epoch
            )
            plot_confusion(result.confusion, list(CLASS_NAMES), out_dir / "confusion_val.png")
            logger.info("  new best macro-F1 %.4f -> %s", best_f1, out_dir / "best.pt")
        elif cfg.train.early_stop_patience and epoch - best_epoch >= cfg.train.early_stop_patience:
            logger.info(
                "No macro-F1 improvement for %d epochs; stopping at epoch %d.",
                cfg.train.early_stop_patience, epoch,
            )
            break

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    save_checkpoint(out_dir / "last.pt", ema.module if ema else model, cfg, spec, {}, len(history))
    plot_history(history, out_dir / "history.png")
    elapsed = time.time() - started
    logger.info(
        "Finished in %.1f min. Best macro-F1 %.4f at epoch %d -> %s",
        elapsed / 60, best_f1, best_epoch, out_dir / "best.pt",
    )
    return {"best_macro_f1": best_f1, "best_epoch": float(best_epoch), "minutes": elapsed / 60}
