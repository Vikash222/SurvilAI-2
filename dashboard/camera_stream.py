from __future__ import annotations

import time
from pathlib import Path

import cv2

from survildb.database import SurvilDB


class CameraStream:
    """Local MJPEG camera stream with lightweight face detection."""

    def __init__(self, db: SurvilDB, camera_id: int):
        self.db = db
        self.camera_id = camera_id
        self.cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )

    def _source(self):
        with self.db.connect() as conn:
            row = conn.execute("SELECT source, enabled FROM cameras WHERE id=?", (self.camera_id,)).fetchone()
        if not row or not row["enabled"]:
            raise ValueError("camera not found or disabled")
        source = row["source"]
        return int(source) if source.isdigit() else source

    def frames(self):
        cap = cv2.VideoCapture(self._source())
        if not cap.isOpened():
            raise RuntimeError("unable to open camera source")
        try:
            last_event = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                for x, y, w, h in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(frame, "FACE DETECTED", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                if faces.size and time.monotonic() - last_event > 2:
                    self.db.add_event("face_detected", camera_id=self.camera_id, confidence=None, metadata_json=f'{{"count":{len(faces)}}}')
                    last_event = time.monotonic()
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        finally:
            cap.release()
