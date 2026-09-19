"""Real-time webcam emotion recognition.

Three things keep this smooth where a naive loop stutters: every face in a frame is
scored in a single batched forward pass, predictions are smoothed per tracked face
instead of recomputed from scratch, and detection can run on a stride so the detector
does not have to fire on every single frame.

Run it as a script (``fer webcam``), not from a notebook cell — ``cv2.imshow`` needs a
real window loop and tends to hang under Jupyter.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from fer.align import crop_face
from fer.detect import FaceDetector
from fer.infer import CentroidTracker, EmotionClassifier, ProbabilitySmoother
from fer.labels import CLASS_COLORS_BGR

logger = logging.getLogger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fourcc(code: str) -> int:
    """Resolve a FourCC code across OpenCV versions.

    OpenCV 5 moved the helper onto ``VideoWriter``; 4.x had it at module level.
    """
    factory = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return int(factory(*code))


def _draw_face(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    ranked: list[tuple[str, float]],
    show_bars: bool,
) -> None:
    """Draw the box, the winning label, and optionally a small probability bar chart."""
    x, y, w, h = box
    label, confidence = ranked[0]
    color = CLASS_COLORS_BGR.get(label, (0, 255, 0))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text = f"{label} {confidence * 100:.0f}%"
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.6, 2)
    # Keep the caption inside the frame when the face is near the top edge.
    top = y - th - baseline - 4
    if top < 0:
        top = y + h + 4
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 4), color, -1)
    cv2.putText(frame, text, (x + 4, top + th + 2), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if not show_bars:
        return
    # Every class gets a bar, so a low-confidence reading is legible as one rather
    # than looking like a confident wrong answer.
    bar_x, bar_y = x + w + 8, y
    for name, prob in ranked:
        bar_color = CLASS_COLORS_BGR.get(name, (200, 200, 200))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(90 * prob), bar_y + 12), bar_color, -1)
        cv2.putText(
            frame, f"{name[:9]} {prob * 100:.0f}", (bar_x + 94, bar_y + 11),
            FONT, 0.36, (240, 240, 240), 1, cv2.LINE_AA,
        )
        bar_y += 16


def run_webcam(
    model_path: Path | str,
    camera: int = 0,
    device: str = "cpu",
    detect_every: int = 1,
    smoothing: float = 0.35,
    min_score: float = 0.7,
    top_k: int = 8,
    show_bars: bool = True,
    mirror: bool = True,
    width: int = 1280,
    height: int = 720,
    record: Path | str | None = None,
    max_frames: int | None = None,
    headless: bool = False,
    source: Path | str | None = None,
) -> dict[str, float]:
    """Run detection plus classification over a camera or video, and draw the result.

    Keys: ``q`` quits, ``b`` toggles the probability bars, ``m`` toggles mirroring.

    ``max_frames`` stops after a fixed number of frames and ``headless`` skips the
    preview window entirely. Together they make the loop runnable without a display or
    a keypress, which is what lets it be exercised in a test or over SSH; ``source``
    points at a video file instead of a camera for the same reason.

    Returns a summary of the run: frames processed, mean FPS, and how many faces were
    seen, so a caller can assert on it rather than watch a window.
    """
    classifier = EmotionClassifier.load(model_path, device=device)
    detector = FaceDetector(score_threshold=min_score)
    tracker = CentroidTracker()
    smoother = ProbabilitySmoother(alpha=smoothing)

    if source is not None:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file {source}.")
    else:
        # DirectShow negotiates resolution far more reliably than the default Windows
        # backend, which often silently ignores the requested size.
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        capture = cv2.VideoCapture(camera, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera}. Check that it is connected, not already "
                "in use by another app, and that this terminal has camera permission."
            )

    actual = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    logger.info(
        "%s at %dx%d | %s backend | model %s",
        f"Video {source}" if source else f"Camera {camera}",
        actual[0], actual[1], classifier.backend, Path(model_path).name,
    )

    writer = None
    if record is not None:
        writer = cv2.VideoWriter(str(record), _fourcc("mp4v"), 20.0, actual)

    frame_times: deque[float] = deque(maxlen=30)
    faces: list = []
    frame_index = 0
    total_faces = 0
    total_seconds = 0.0

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                # End of file for a video source; a dropped frame for a camera.
                logger.info("No more frames from the source; stopping.")
                break
            started = time.perf_counter()
            if mirror:
                frame = cv2.flip(frame, 1)

            # Detection is the expensive half on CPU; a stride > 1 reuses the last boxes.
            if frame_index % detect_every == 0 or not faces:
                faces = detector.detect(frame)
            frame_index += 1

            if faces:
                size = classifier.spec.image_size
                crops = [crop_face(frame, f.landmarks, f.box, size) for f in faces]
                probs = classifier.predict_batch(crops)
                track_ids = tracker.update([f.box for f in faces])
                smoother.forget(set(track_ids))
                for face, prob, track_id in zip(faces, probs, track_ids, strict=True):
                    ranked = classifier.top_k(smoother.update(track_id, prob), k=top_k)
                    _draw_face(frame, face.box, ranked, show_bars)

            total_faces += len(faces)
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            frame_times.append(elapsed)
            fps = len(frame_times) / max(sum(frame_times), 1e-6)
            cv2.putText(
                frame, f"{fps:5.1f} FPS | {len(faces)} face(s) | {classifier.backend}",
                (12, 28), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, "q quit   b bars   m mirror",
                (12, actual[1] - 14), FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            if headless:
                continue
            cv2.imshow("Emotion Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                show_bars = not show_bars
            if key == ord("m"):
                mirror = not mirror
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        logger.info("Source released after %d frames.", frame_index)

    summary = {
        "frames": float(frame_index),
        "faces_seen": float(total_faces),
        "mean_fps": frame_index / total_seconds if total_seconds > 0 else 0.0,
        "mean_ms_per_frame": total_seconds / frame_index * 1000 if frame_index else 0.0,
    }
    logger.info(
        "Processed %d frames, %d face detections, %.1f FPS (%.1f ms/frame)",
        frame_index, total_faces, summary["mean_fps"], summary["mean_ms_per_frame"],
    )
    return summary


def annotate_image(
    model_path: Path | str,
    image_path: Path | str,
    out_path: Path | str | None = None,
    device: str = "cpu",
    min_score: float = 0.7,
    top_k: int = 8,
) -> list[tuple[tuple[int, int, int, int], list[tuple[str, float]]]]:
    """Run detection and classification over a still image.

    Returns one ``(box, ranked_predictions)`` pair per face, and writes an annotated copy
    when ``out_path`` is given.
    """
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read an image from {image_path}")

    classifier = EmotionClassifier.load(model_path, device=device)
    faces = FaceDetector(score_threshold=min_score).detect(frame)
    if not faces:
        logger.warning("No faces detected in %s", image_path)
        return []

    size = classifier.spec.image_size
    crops = [crop_face(frame, f.landmarks, f.box, size) for f in faces]
    probs = classifier.predict_batch(crops)

    results = []
    for face, prob in zip(faces, probs, strict=True):
        ranked = classifier.top_k(prob, k=top_k)
        results.append((face.box, ranked))
        _draw_face(frame, face.box, ranked, show_bars=True)

    if out_path is not None:
        cv2.imwrite(str(out_path), frame)
        logger.info("Annotated image written to %s", out_path)
    return results
