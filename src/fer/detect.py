"""Face detection with YuNet.

YuNet replaces the Haar cascade this project started with. It is a ~230 KB CNN that ships
in the OpenCV model zoo, runs comfortably faster than real time on CPU, and — critically
for emotion recognition — returns five facial landmarks alongside each box. Those
landmarks let us align the crop before classification, which removes in-plane head
rotation as a nuisance variable that the classifier would otherwise have to learn around.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ZOO_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet"

#: Newest first. If a download fails we fall back to the previous release rather than
#: leaving the user with no detector at all.
YUNET_MODELS: tuple[str, ...] = (
    "face_detection_yunet_2026may.onnx",
    "face_detection_yunet_2023mar.onnx",
)


def cache_dir() -> Path:
    """Directory for downloaded auxiliary models, overridable via ``FER_CACHE_DIR``."""
    env = os.environ.get("FER_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "fer"


def download_yunet(dest_dir: Path | None = None) -> Path:
    """Fetch the YuNet weights, returning a local path. Cached after the first call."""
    dest_dir = dest_dir or cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for name in YUNET_MODELS:
        dest = dest_dir / name
        # The zoo stores these in Git LFS; a pointer file is ~130 bytes, a real model
        # is ~230 KB. Size is a cheap way to reject a half-finished or pointer download.
        if dest.exists() and dest.stat().st_size > 100_000:
            return dest
        try:
            logger.info("Downloading face detector %s", name)
            with urllib.request.urlopen(f"{ZOO_URL}/{name}", timeout=120) as response:  # noqa: S310
                payload = response.read()
            if len(payload) < 100_000:
                raise OSError(f"got {len(payload)} bytes, expected a ~230 KB model")
            dest.write_bytes(payload)
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next model, report at the end
            errors.append(f"{name}: {exc}")
            logger.warning("Could not fetch %s (%s)", name, exc)

    raise RuntimeError(
        "Failed to download a YuNet face detector. Tried:\n  " + "\n  ".join(errors)
    )


@dataclass(frozen=True)
class Face:
    """One detected face: pixel box, five landmarks, and detector confidence."""

    x: int
    y: int
    w: int
    h: int
    score: float
    #: ``(5, 2)`` float array: right eye, left eye, nose tip, right mouth, left mouth.
    landmarks: np.ndarray

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def padded_crop(self, frame: np.ndarray, margin: float = 0.15) -> np.ndarray:
        """Crop with a margin, clamped to the frame.

        A tight detector box clips the brow and jaw, which carry a lot of the expression
        signal, so a little padding measurably helps when landmarks are unavailable.
        """
        pad_x, pad_y = int(self.w * margin), int(self.h * margin)
        x1 = max(0, self.x - pad_x)
        y1 = max(0, self.y - pad_y)
        x2 = min(frame.shape[1], self.x + self.w + pad_x)
        y2 = min(frame.shape[0], self.y + self.h + pad_y)
        return frame[y1:y2, x1:x2]


class FaceDetector:
    """Thin wrapper over ``cv2.FaceDetectorYN`` that hides the input-size bookkeeping.

    YuNet must be told the frame size up front and re-told whenever it changes, which is
    easy to get wrong when a webcam negotiates a different resolution mid-stream.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 50,
    ) -> None:
        path = Path(model_path) if model_path else download_yunet()
        # Positional arguments: the keyword names differ between OpenCV bindings.
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._input_size: tuple[int, int] | None = None
        self.model_path = path
        logger.info("Face detector ready (%s)", path.name)

    def detect(self, frame: np.ndarray) -> list[Face]:
        """Detect faces in a BGR frame, largest first."""
        height, width = frame.shape[:2]
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, raw = self._detector.detect(frame)
        if raw is None:
            return []

        faces = [
            Face(
                x=int(row[0]),
                y=int(row[1]),
                w=int(row[2]),
                h=int(row[3]),
                score=float(row[14]),
                landmarks=row[4:14].reshape(5, 2).astype(np.float32),
            )
            for row in raw
            # Degenerate boxes occasionally survive NMS at frame edges.
            if row[2] > 1 and row[3] > 1
        ]
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
        return faces
