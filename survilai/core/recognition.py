from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RecognitionResult:
    identity: str
    score: float


class Recognizer(Protocol):
    def identify(self, face_image) -> RecognitionResult: ...


class RecognitionNotConfigured:
    """Safe Phase 1 placeholder until SurvilAI's own recognition model is ready."""

    def identify(self, face_image) -> RecognitionResult:
        _ = face_image
        return RecognitionResult(identity="unknown", score=0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity without a third-party service."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)
