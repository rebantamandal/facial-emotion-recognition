"""Evaluation metrics.

Plain accuracy is a poor headline number on FER+: predicting only *happiness* and
*neutral* already scores well because those two classes dominate. Macro-F1 and balanced
accuracy weight every emotion equally, so they are what model selection and early
stopping key off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from fer.labels import CLASS_NAMES


@dataclass
class EvalResult:
    """Metrics for one pass over a split, plus the raw confusion matrix."""

    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    mean_confidence: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: np.ndarray | None = None

    def summary(self) -> dict[str, float]:
        """The scalar metrics only, suitable for logging or a checkpoint payload."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "mean_confidence": self.mean_confidence,
        }

    def format_table(self) -> str:
        """Human-readable per-class breakdown."""
        lines = [f"{'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}"]
        lines.append("-" * 45)
        for name, row in self.per_class.items():
            lines.append(
                f"{name:<12}{row['precision']:>8.3f}{row['recall']:>8.3f}"
                f"{row['f1']:>8.3f}{int(row['support']):>9d}"
            )
        lines.append("-" * 45)
        lines.append(
            f"{'accuracy':<12}{self.accuracy:>8.3f}   "
            f"macro-F1 {self.macro_f1:.3f}   balanced-acc {self.balanced_accuracy:.3f}"
        )
        return "\n".join(lines)


def compute_metrics(
    probs: np.ndarray, targets: np.ndarray, class_names: list[str] | None = None
) -> EvalResult:
    """Score predicted probabilities against soft or hard targets.

    ``probs`` is ``(N, C)`` softmax output. ``targets`` may be ``(N, C)`` probability
    vectors, in which case the argmax is taken as ground truth, or ``(N,)`` integer labels.
    """
    names = class_names or list(CLASS_NAMES)
    y_pred = probs.argmax(axis=1)
    y_true = targets.argmax(axis=1) if targets.ndim == 2 else targets.astype(np.int64)

    labels = list(range(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": float(support[i]),
        }
        for i in labels
    }
    return EvalResult(
        accuracy=float((y_pred == y_true).mean()),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        mean_confidence=float(probs.max(axis=1).mean()),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )


def plot_confusion(
    matrix: np.ndarray, class_names: list[str], path: Path | str, normalize: bool = True
) -> Path:
    """Save a confusion-matrix figure, row-normalised by default.

    Row normalisation is what you usually want here: it shows per-class recall even
    though the classes differ in size by two orders of magnitude.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = matrix.astype(np.float64)
    if normalize:
        totals = data.sum(axis=1, keepdims=True)
        data = np.divide(data, totals, out=np.zeros_like(data), where=totals > 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    image = ax.imshow(data, cmap="magma", vmin=0.0, vmax=1.0 if normalize else data.max())
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Row-normalised confusion matrix" if normalize else "Confusion matrix")

    threshold = (data.max() + data.min()) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}",
                ha="center",
                va="center",
                color="white" if data[i, j] < threshold else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_history(history: list[dict[str, float]], path: Path | str) -> Path:
    """Plot loss and the selection metric across epochs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    if any("val_loss" in h for h in history):
        ax_loss.plot(epochs, [h.get("val_loss", np.nan) for h in history], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    for key, label in (("val_accuracy", "accuracy"), ("val_macro_f1", "macro-F1")):
        if any(key in h for h in history):
            ax_metric.plot(epochs, [h.get(key, np.nan) for h in history], label=label)
    ax_metric.set_xlabel("epoch")
    ax_metric.set_ylabel("score")
    ax_metric.legend()
    ax_metric.grid(alpha=0.3)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
