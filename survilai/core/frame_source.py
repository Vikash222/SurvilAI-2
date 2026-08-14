from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass
class Frame:
    index: int
    image: object


class VideoSource:
    """OpenCV-backed local camera/video source.

    Phase 1 supports webcam indexes and local video files. RTSP is intentionally
    left behind the same interface so it can be added without changing the AI
    pipeline in a later phase.
    """

    def __init__(self, source: str = "0") -> None:
        self.source = source
        self.capture: cv2.VideoCapture | None = None

    def _source_value(self):
        return int(self.source) if self.source.isdigit() else self.source

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self._source_value())
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

    def frames(self) -> Iterator[Frame]:
        if self.capture is None:
            self.open()
        assert self.capture is not None
        index = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            yield Frame(index=index, image=image)
            index += 1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()
