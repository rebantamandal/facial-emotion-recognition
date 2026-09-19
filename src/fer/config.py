"""Typed configuration objects, loadable from the YAML files in ``configs/``.

Everything the pipeline needs is described here so a run is reproducible from a single
file: the same YAML drives training, evaluation, export and the webcam app.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where the images come from and how they are turned into batches."""

    #: ``ferplus`` downloads and builds the dataset; ``imagefolder`` reads train/val dirs.
    source: str = "ferplus"
    root: Path = Path("data")
    image_size: int = 224
    #: ``soft`` trains on the full FER+ vote distribution, ``majority`` on a single label.
    #: The FER+ paper shows the vote distribution is the stronger target, so it is default.
    label_mode: str = "soft"
    #: Drop images where the winning emotion holds less than this share of the votes.
    #: 0.0 keeps everything; raising it trades dataset size for label confidence.
    min_agreement: float = 0.0
    batch_size: int = 128
    num_workers: int = 8
    #: Oversample rare classes. FER+ is severely imbalanced (contempt is well under 1%).
    balanced_sampler: bool = True
    #: How hard to rebalance: sampling weight is ``(1 / count) ** sampler_power``.
    #: 1.0 is full inverse frequency, which makes every class equally likely; 0.5 (square
    #: root) is a softer middle ground that shows the rare classes often without repeating
    #: contempt's ~160 images dozens of times an epoch; 0.0 disables rebalancing.
    #: The shipped checkpoint was trained at 1.0, so that stays the default — the reported
    #: numbers would not reproduce otherwise.
    sampler_power: float = 1.0
    #: Convert to single-channel and replicate. FER2013 is grayscale, so colour jitter on
    #: it is meaningless; this keeps the 3-channel backbone input without faking colour.
    grayscale: bool = True


@dataclass
class ModelConfig:
    """Backbone selection and regularisation that lives inside the network."""

    backbone: str = "mobilenetv4_conv_medium.e500_r256_in1k"
    pretrained: bool = True
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1
    #: Shrink the freshly initialised classifier weights by this factor. timm's default
    #: head init gives logits with std ~5 on this backbone, so training starts at a loss
    #: of ~7 instead of ln(8)=2.08 and dumps huge early gradients into the pretrained
    #: trunk. Scaling the head down starts training from a near-uniform prediction.
    head_init_scale: float = 0.05


@dataclass
class TrainConfig:
    """Optimisation schedule and the modern-training bag of tricks."""

    epochs: int = 30
    lr: float = 3e-4
    #: Backbone learns slower than the fresh head; standard discriminative fine-tuning.
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    #: Probability that a given batch gets mixup/cutmix applied at all.
    mixup_prob: float = 0.5
    ema_decay: float = 0.9998
    clip_grad: float = 1.0
    #: ``bf16`` needs Ampere or newer; the trainer falls back to fp16 then fp32 itself.
    amp_dtype: str = "bf16"
    compile: bool = True
    #: Stop when macro-F1 has not improved for this many epochs. 0 disables early stopping.
    early_stop_patience: int = 8
    seed: int = 42


@dataclass
class Config:
    """Top-level config: the three sections above plus run bookkeeping."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: Path = Path("runs/default")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a config file, filling anything it omits from the dataclass defaults."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        sections = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig}
        kwargs: dict[str, Any] = {}
        for name, section_cls in sections.items():
            known = {f.name for f in dataclasses.fields(section_cls)}
            provided = raw.get(name) or {}
            unknown = set(provided) - known
            if unknown:
                raise ValueError(f"Unknown key(s) in '{name}' config section: {sorted(unknown)}")
            kwargs[name] = section_cls(**provided)
        kwargs["data"].root = Path(kwargs["data"].root)
        if "out_dir" in raw:
            kwargs["out_dir"] = Path(raw["out_dir"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view with Paths stringified, for writing back beside a checkpoint."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        return convert(dataclasses.asdict(self))

    def save(self, path: str | Path) -> None:
        """Snapshot the resolved config next to the run's artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
