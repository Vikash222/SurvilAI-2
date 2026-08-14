from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source: int | str
    enabled: bool = True


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.config.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera: {self.config.camera_id}")

    def frames(self) -> Iterator[object]:
        if self.capture is None:
            self.open()
        assert self.capture is not None
        while self.config.enabled:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class MultiCameraManager:
    def __init__(self, cameras: list[CameraConfig]) -> None:
        self.cameras = [CameraStream(c) for c in cameras if c.enabled]

    def close(self) -> None:
        for camera in self.cameras:
            camera.close()
