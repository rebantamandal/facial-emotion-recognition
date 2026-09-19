"""Canonical FER+ label definitions.

FER+ (Barsoum et al., 2016) re-annotates every FER2013 image with ten crowd votes.
Eight of those columns are emotions we train on; the remaining two mark images the
annotators could not label ("unknown") or that are not faces at all ("NF"), and they
are filtered out during dataset construction rather than becoming classes.
"""

from __future__ import annotations

# Order matches the column order of Microsoft's fer2013new.csv, so a raw vote row can be
# sliced directly into a label vector without any reordering.
CLASS_NAMES: tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

# Vote columns that disqualify an image instead of labelling it.
REJECT_COLUMNS: tuple[str, ...] = ("unknown", "NF")

VOTE_COLUMNS: tuple[str, ...] = CLASS_NAMES + REJECT_COLUMNS

NUM_CLASSES: int = len(CLASS_NAMES)

#: Colours used to draw each class in the webcam overlay, in BGR for OpenCV.
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "neutral": (180, 180, 180),
    "happiness": (80, 200, 120),
    "surprise": (240, 200, 60),
    "sadness": (200, 130, 70),
    "anger": (70, 70, 230),
    "disgust": (120, 160, 90),
    "fear": (190, 120, 200),
    "contempt": (90, 140, 210),
}


def class_index(name: str) -> int:
    """Return the training index of a class name, raising a helpful error if unknown."""
    try:
        return CLASS_NAMES.index(name)
    except ValueError as exc:  # pragma: no cover - defensive
        raise KeyError(f"{name!r} is not a FER+ class; expected one of {CLASS_NAMES}") from exc
