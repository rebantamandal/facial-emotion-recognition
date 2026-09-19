"""Model construction, parameter grouping and checkpoint I/O.

The backbone is whatever timm name the config asks for, so swapping architectures is a
one-line config change. Two details are worth calling out:

* The preprocessing statistics are *read back off the checkpoint* via timm's data config
  rather than assumed, so the input distribution always matches what the backbone was
  pretrained on.
* Checkpoints carry their own config and class names. Inference never has to reconstruct
  a dataloader to find out what the output indices mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

from fer.config import Config, ModelConfig
from fer.labels import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessSpec:
    """Everything inference needs to turn a face crop into a model input."""

    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    grayscale: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "grayscale": self.grayscale,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PreprocessSpec:
        return cls(
            image_size=int(raw["image_size"]),
            mean=tuple(float(v) for v in raw["mean"]),
            std=tuple(float(v) for v in raw["std"]),
            grayscale=bool(raw.get("grayscale", True)),
        )


def create_model(cfg: ModelConfig, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Instantiate a timm backbone with a fresh ``num_classes`` head."""
    model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )
    scale_head_init(model, cfg.head_init_scale)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Built %s: %.1fM parameters", cfg.backbone, n_params / 1e6)
    return model


@torch.no_grad()
def scale_head_init(model: nn.Module, scale: float) -> None:
    """Shrink the classifier's initial weights and zero its bias.

    A randomly initialised head on top of a pretrained trunk can emit logits with a
    standard deviation of several units, which starts training at a far higher loss than
    the ln(num_classes) a uniform prediction would give and sends correspondingly large
    gradients into the backbone. Scaling the head down costs nothing and starts the run
    from an honest uniform prior.
    """
    if scale is None or scale >= 1.0:
        return
    classifier = model.get_classifier()
    if classifier is None:
        return
    for module in ([classifier] if hasattr(classifier, "weight") else classifier.modules()):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.mul_(scale)
        if getattr(module, "bias", None) is not None:
            module.bias.zero_()


def preprocess_spec(model: nn.Module, image_size: int, grayscale: bool) -> PreprocessSpec:
    """Read the backbone's own normalisation statistics.

    This is the fix for the classic transfer-learning bug of feeding ``[0, 1]`` pixels to
    a network pretrained on ``[-1, 1]`` or on ImageNet statistics: we never guess.
    """
    data_cfg = timm.data.resolve_model_data_config(model)
    return PreprocessSpec(
        image_size=image_size,
        mean=tuple(float(v) for v in data_cfg["mean"]),
        std=tuple(float(v) for v in data_cfg["std"]),
        grayscale=grayscale,
    )


def param_groups(
    model: nn.Module, weight_decay: float, backbone_lr_mult: float, base_lr: float
) -> list[dict[str, Any]]:
    """Split parameters into four groups: {backbone, head} x {decay, no-decay}.

    The freshly initialised head needs a much larger step than the pretrained trunk, and
    norm/bias parameters should never be weight-decayed — decaying them fights the
    normalisation statistics and measurably hurts.
    """
    classifier = model.get_classifier()
    head_ids = {id(p) for p in classifier.parameters()} if classifier is not None else set()

    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("head", True): [],
        ("head", False): [],
        ("backbone", True): [],
        ("backbone", False): [],
    }
    for param in model.parameters():
        if not param.requires_grad:
            continue
        where = "head" if id(param) in head_ids else "backbone"
        # 1-D tensors are biases and norm weights.
        decays = param.ndim > 1
        groups[(where, decays)].append(param)

    out: list[dict[str, Any]] = []
    for (where, decays), params in groups.items():
        if not params:
            continue
        out.append(
            {
                "params": params,
                "lr": base_lr * (backbone_lr_mult if where == "backbone" else 1.0),
                "weight_decay": weight_decay if decays else 0.0,
                "name": f"{where}_{'decay' if decays else 'nodecay'}",
            }
        )
    logger.info(
        "Parameter groups: %s",
        ", ".join(f"{g['name']}={sum(p.numel() for p in g['params']) / 1e6:.2f}M" for g in out),
    )
    return out


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    cfg: Config,
    spec: PreprocessSpec,
    metrics: dict[str, float] | None = None,
    epoch: int | None = None,
) -> Path:
    """Write a self-describing checkpoint.

    ``torch.save`` of a plain dict of tensors keeps the file loadable with
    ``weights_only=True``, so loading someone else's checkpoint cannot execute code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "config": cfg.to_dict(),
            "preprocess": spec.to_dict(),
            "class_names": list(CLASS_NAMES),
            "metrics": metrics or {},
            "epoch": epoch,
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path | str, device: torch.device | str = "cpu", strict: bool = True
) -> tuple[nn.Module, PreprocessSpec, list[str], dict[str, Any]]:
    """Rebuild a model straight from a checkpoint, no config file required.

    Returns ``(model, preprocess_spec, class_names, payload)``. The model is in eval mode
    on ``device``; the backbone is built with ``pretrained=False`` because the checkpoint
    supplies every weight.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    class_names = list(payload.get("class_names", CLASS_NAMES))

    model_cfg = ModelConfig(**payload["config"]["model"])
    model_cfg.pretrained = False
    model = create_model(model_cfg, num_classes=len(class_names))

    state = payload["state_dict"]
    # torch.compile wraps modules and prefixes every key; strip it so compiled and eager
    # checkpoints stay interchangeable.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        logger.warning("Checkpoint key mismatch: missing=%s unexpected=%s", missing, unexpected)

    spec = PreprocessSpec.from_dict(payload["preprocess"])
    model.eval().to(device)
    logger.info(
        "Loaded %s (epoch %s, metrics %s)", path, payload.get("epoch"), payload.get("metrics")
    )
    return model, spec, class_names, payload
