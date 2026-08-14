from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float


class Detector(Protocol):
    def detect(self, frame) -> list[FaceBox]: ...


class HaarFaceDetector:
    """Local baseline detector shipped with OpenCV.

    This is deliberately an engine adapter. A trained detector can replace it
    later without changing the video pipeline or recognition code.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError("OpenCV face detector could not be loaded")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame) -> list[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(40, 40),
        )
        return [FaceBox(x=x, y=y, width=w, height=h, confidence=1.0) for x, y, w, h in faces]
