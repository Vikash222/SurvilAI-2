"""SurvilAI-owned face recognition model components."""

from .network import SurvilFaceNet
from .yolo26_embedding import YOLO26EmbeddingModel

__all__ = ["SurvilFaceNet", "YOLO26EmbeddingModel"]
