"""Real-time facial emotion recognition built on FER+, timm backbones and ONNX Runtime."""

from fer.labels import CLASS_NAMES, NUM_CLASSES

__version__ = "0.2.0"
__all__ = ["CLASS_NAMES", "NUM_CLASSES", "__version__"]
