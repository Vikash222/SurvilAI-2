from __future__ import annotations

from dataclasses import dataclass

import cv2

from .detection import Detector, FaceBox
from .frame_source import VideoSource
from .recognition import Recognizer, RecognitionResult


@dataclass(frozen=True)
class FaceObservation:
    box: FaceBox
    recognition: RecognitionResult


class LocalPipeline:
    """Connect video, detection and recognition through stable interfaces."""

    def __init__(self, source: VideoSource, detector: Detector, recognizer: Recognizer) -> None:
        self.source = source
        self.detector = detector
        self.recognizer = recognizer

    def run(self, display: bool = True) -> None:
        with self.source:
            for frame in self.source.frames():
                observations: list[FaceObservation] = []
                for box in self.detector.detect(frame.image):
                    crop = frame.image[box.y : box.y + box.height, box.x : box.x + box.width]
                    result = self.recognizer.identify(crop)
                    observations.append(FaceObservation(box=box, recognition=result))
                    if display:
                        cv2.rectangle(
                            frame.image,
                            (box.x, box.y),
                            (box.x + box.width, box.y + box.height),
                            (0, 255, 0),
                            2,
                        )
                        label = f"{result.identity} {result.score:.2f}"
                        cv2.putText(
                            frame.image,
                            label,
                            (box.x, max(20, box.y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                if display:
                    cv2.imshow("SurvilAI - Phase 1", frame.image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if display:
            cv2.destroyAllWindows()
