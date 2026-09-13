"""GigaAM v3 (e2e-RNNT) Russian speech recognition over ONNX Runtime."""

__version__ = "0.1.0"

from .features import compute_features, load_audio, num_frames
from .model import GigaAM
from . import models

__all__ = ["GigaAM", "compute_features", "load_audio", "num_frames", "models", "__version__"]