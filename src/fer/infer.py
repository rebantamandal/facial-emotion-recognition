"""Unified inference over either a PyTorch checkpoint or an exported ONNX model.

The same ``EmotionClassifier`` serves the webcam app, the image CLI and the notebook, and
it reproduces the training preprocessing exactly — resize, grayscale, scale to ``[0, 1]``,
then normalise with the backbone's own mean/std. Getting that last step wrong is the most
common way a model that trained fine performs badly in the demo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from fer.labels import CLASS_NAMES
from fer.models import PreprocessSpec

logger = logging.getLogger(__name__)


def preprocess_batch(crops: list[np.ndarray], spec: PreprocessSpec) -> np.ndarray:
    """Turn BGR face crops into a normalised ``(N, 3, H, W)`` float32 batch."""
    size = spec.image_size
    mean = np.asarray(spec.mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(spec.std, dtype=np.float32).reshape(3, 1, 1)

    batch = np.empty((len(crops), 3, size, size), dtype=np.float32)
    for i, crop in enumerate(crops):
        if crop.shape[:2] != (size, size):
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        if spec.grayscale:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            planes = np.repeat(gray[None, ...], 3, axis=0)
        else:
            # Model was trained on colour: undo OpenCV's BGR ordering.
            planes = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        batch[i] = (planes.astype(np.float32) / 255.0 - mean) / std
    return batch


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class EmotionClassifier:
    """Loads a model once and scores batches of face crops.

    Use :meth:`from_checkpoint` for a ``.pt`` produced by training, or :meth:`from_onnx`
    for an exported ``.onnx``. ONNX Runtime is meaningfully faster on CPU and is the
    recommended path for the webcam app; the torch backend is there for GPU and for
    debugging against the training-time graph.
    """

    def __init__(
        self,
        spec: PreprocessSpec,
        class_names: list[str],
        backend: str,
        logit_bias: np.ndarray | None = None,
    ) -> None:
        self.spec = spec
        self.class_names = class_names
        self.backend = backend
        #: Per-class additive bias fitted by `fer calibrate`, applied before the softmax.
        #: Zero when the model was never calibrated, so the code path is always the same.
        self.logit_bias = (
            np.zeros(len(class_names), dtype=np.float32)
            if logit_bias is None
            else np.asarray(logit_bias, dtype=np.float32)
        )

    @classmethod
    def from_checkpoint(
        cls, path: Path | str, device: str = "cpu"
    ) -> EmotionClassifier:
        import torch

        from fer.models import load_checkpoint

        model, spec, class_names, payload = load_checkpoint(path, device=device)
        obj = cls(spec, class_names, backend="torch", logit_bias=payload.get("logit_bias"))
        obj._model = model
        obj._device = torch.device(device)
        obj._torch = torch
        return obj

    @classmethod
    def from_onnx(cls, path: Path | str, providers: list[str] | None = None) -> EmotionClassifier:
        import onnxruntime as ort

        path = Path(path)
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Expected preprocessing metadata at {meta_path}. Re-export with `fer export`."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        spec = PreprocessSpec.from_dict(meta["preprocess"])
        class_names = list(meta.get("class_names", CLASS_NAMES))

        available = ort.get_available_providers()
        chosen = [p for p in (providers or ["CPUExecutionProvider"]) if p in available]
        if not chosen:
            chosen = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), options, providers=chosen)

        obj = cls(spec, class_names, backend="onnx", logit_bias=meta.get("logit_bias"))
        obj._session = session
        obj._input_name = session.get_inputs()[0].name
        logger.info("ONNX session on %s", session.get_providers())
        return obj

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> EmotionClassifier:
        """Dispatch on file extension so callers do not have to care."""
        path = Path(path)
        if path.suffix == ".onnx":
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device.startswith("cuda")
                else ["CPUExecutionProvider"]
            )
            return cls.from_onnx(path, providers)
        return cls.from_checkpoint(path, device)

    def predict_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Score BGR crops, returning ``(N, C)`` probabilities."""
        if not crops:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)
        batch = preprocess_batch(crops, self.spec)

        if self.backend == "onnx":
            logits = np.asarray(self._session.run(None, {self._input_name: batch})[0], np.float32)
        else:
            torch = self._torch
            with torch.inference_mode():
                tensor = torch.from_numpy(batch).to(self._device)
                # __call__ rather than .predict(): no retracing, no per-call overhead.
                logits = self._model(tensor).float().cpu().numpy()
        return softmax(logits + self.logit_bias)

    def predict(self, crop: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Score a single crop, returning ``(label, confidence, probabilities)``."""
        probs = self.predict_batch([crop])[0]
        index = int(probs.argmax())
        return self.class_names[index], float(probs[index]), probs

    def top_k(self, probs: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Most likely classes for one probability vector, highest first."""
        order = np.argsort(probs)[::-1][:k]
        return [(self.class_names[i], float(probs[i])) for i in order]


class ProbabilitySmoother:
    """Exponential moving average of per-face probabilities across frames.

    Frame-by-frame predictions flicker badly at expression boundaries. Smoothing over a
    short history makes the label stable enough to read without adding noticeable lag,
    and it is far cheaper than running a temporal model.
    """

    def __init__(self, alpha: float = 0.35, max_faces: int = 16) -> None:
        self.alpha = alpha
        self.max_faces = max_faces
        self._state: dict[int, np.ndarray] = {}

    def update(self, track_id: int, probs: np.ndarray) -> np.ndarray:
        previous = self._state.get(track_id)
        blended = probs if previous is None else self.alpha * probs + (1 - self.alpha) * previous
        self._state[track_id] = blended
        if len(self._state) > self.max_faces:
            # Drop the oldest insertions; dicts preserve insertion order.
            for key in list(self._state)[: len(self._state) - self.max_faces]:
                del self._state[key]
        return blended

    def forget(self, keep: set[int]) -> None:
        """Drop state for tracks that are no longer visible."""
        for key in set(self._state) - keep:
            del self._state[key]


class CentroidTracker:
    """Minimal nearest-centroid tracker, just enough to keep smoothing stable.

    Full multi-object tracking is overkill for a webcam demo with a handful of faces; all
    this needs to do is keep the same identity attached to the same face between frames.
    """

    def __init__(self, max_distance: float = 120.0, max_missing: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self._next_id = 0
        self._tracks: dict[int, np.ndarray] = {}
        self._missing: dict[int, int] = {}

    def update(self, boxes: list[tuple[int, int, int, int]]) -> list[int]:
        centroids = [np.array([x + w / 2.0, y + h / 2.0]) for x, y, w, h in boxes]
        assigned: list[int] = []
        unmatched = set(self._tracks)

        for centroid in centroids:
            best_id, best_dist = None, self.max_distance
            for track_id in unmatched:
                distance = float(np.linalg.norm(self._tracks[track_id] - centroid))
                if distance < best_dist:
                    best_id, best_dist = track_id, distance
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            else:
                unmatched.discard(best_id)
            self._tracks[best_id] = centroid
            self._missing[best_id] = 0
            assigned.append(best_id)

        for track_id in unmatched:
            self._missing[track_id] = self._missing.get(track_id, 0) + 1
            if self._missing[track_id] > self.max_missing:
                del self._tracks[track_id]
                del self._missing[track_id]
        return assigned
