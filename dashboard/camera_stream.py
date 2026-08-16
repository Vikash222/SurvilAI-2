from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from survildb.database import SurvilDB
from survilai.live_recognition import LiveRecognizer


class CameraStream:
    """Local MJPEG stream with motion-triggered face recognition."""

    def __init__(self, db: SurvilDB, camera_id: int):
        self.db = db
        self.camera_id = camera_id

        self.cascade = cv2.CascadeClassifier(
            str(
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
        )

        checkpoint = os.getenv(
            "SURVILAI_CHECKPOINT",
            "models/survil-face-v1.pt",
        )

        self.recognizer = None

        if Path(checkpoint).exists():
          self.recognizer = LiveRecognizer(
    db,
    checkpoint,
    threshold=float(
        os.getenv("SURVILAI_MATCH_THRESHOLD", "0.72")
    ),
    quality_threshold=float(
        os.getenv("SURVILAI_QUALITY_THRESHOLD", "0.25")
    ),
)
        # ---------------------------------------------------------
        # EVENT COOLDOWN
        # ---------------------------------------------------------

        self.event_cooldown = float(
            os.getenv("SURVILAI_EVENT_COOLDOWN", "10")
        )

        # IMPORTANT:
        # Keep cooldown state on the CameraStream instance.
        # Do NOT create this dictionary inside frames().
        self.last_events: dict[
            tuple[int, int | None, str], float
        ] = {}

        # ---------------------------------------------------------
        # MOTION DETECTION
        # ---------------------------------------------------------

        self.motion_enabled = (
            os.getenv(
                "SURVILAI_MOTION_ENABLED",
                "1",
            )
            == "1"
        )

        # Pixel difference required to consider an area changed.
        self.motion_threshold = float(
            os.getenv(
                "SURVILAI_MOTION_THRESHOLD",
                "25",
            )
        )

        # Minimum contour area considered actual movement.
        self.motion_min_area = int(
            os.getenv(
                "SURVILAI_MOTION_MIN_AREA",
                "1500",
            )
        )

        # Smaller image = faster motion detection.
        self.motion_resize_width = int(
            os.getenv(
                "SURVILAI_MOTION_WIDTH",
                "640",
            )
        )

        # How frequently motion detection runs.
        self.motion_check_interval = float(
            os.getenv(
                "SURVILAI_MOTION_INTERVAL",
                "0.2",
            )
        )

        # ---------------------------------------------------------
        # FACE DETECTION / RECOGNITION SAFETY
        # ---------------------------------------------------------
        # Haar can produce false positives on doors, walls and
        # high-contrast patterns. Keep weak detections away from
        # the recognition model.
        self.face_min_size = int(
            os.getenv("SURVILAI_FACE_MIN_SIZE", "80")
        )
        self.face_min_neighbors = int(
            os.getenv("SURVILAI_FACE_MIN_NEIGHBORS", "8")
        )
        self.face_min_brightness = float(
            os.getenv("SURVILAI_FACE_MIN_BRIGHTNESS", "25")
        )
        self.face_max_brightness = float(
            os.getenv("SURVILAI_FACE_MAX_BRIGHTNESS", "235")
        )
        self.face_min_variance = float(
            os.getenv("SURVILAI_FACE_MIN_VARIANCE", "80")
        )

        # Require the same recognized person across multiple
        # detections before creating a known_person event.
        self.confirm_frames = int(
            os.getenv("SURVILAI_CONFIRM_FRAMES", "3")
        )
        self.confirm_window = float(
            os.getenv("SURVILAI_CONFIRM_WINDOW", "3")
        )
        self.pending_matches: dict[int, dict[str, object]] = {}

        # ---------------------------------------------------------
        # SNAPSHOT
        # ---------------------------------------------------------
        self.snapshot_enabled = (
            os.getenv("SURVILAI_SNAPSHOT_ENABLED", "0") == "1"
        )
        self.snapshot_dir = Path(
            os.getenv("SURVILAI_SNAPSHOT_DIR", "snapshots")
        )
        self.snapshot_quality = int(
            os.getenv("SURVILAI_JPEG_QUALITY", "90")
        )

    def _save_snapshot(
        self,
        frame,
        event_type: str,
        person_id: int | None,
    ) -> str | None:
        """Save an event frame and return its dashboard-relative path."""
        if not self.snapshot_enabled:
            return None

        try:
            self.snapshot_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )
            milliseconds = int(
                (time.time() % 1) * 1000
            )

            safe_event = "".join(
                ch if ch.isalnum() or ch in "_-"
                else "_"
                for ch in str(event_type)
            )

            person_part = (
                str(person_id)
                if person_id is not None
                else "unknown"
            )

            filename = (
                f"camera_{self.camera_id}_"
                f"{timestamp}_{milliseconds:03d}_"
                f"{safe_event}_{person_part}.jpg"
            )

            path = self.snapshot_dir / filename

            ok = cv2.imwrite(
                str(path),
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    max(1, min(100, self.snapshot_quality)),
                ],
            )

            if not ok:
                return None

            return filename

        except Exception:
            # Snapshot failure must never stop the live camera stream.
            return None

    def _source(self):
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT source, enabled FROM cameras WHERE id=?",
                (self.camera_id,),
            ).fetchone()

        if not row or not row["enabled"]:
            raise ValueError("camera not found or disabled")

        source = row["source"]

        return int(source) if source.isdigit() else source

    def frames(self):
        cap = cv2.VideoCapture(self._source())

        if not cap.isOpened():
            raise RuntimeError(
                "unable to open camera source"
            )

        # ---------------------------------------------------------
        # MOTION STATE
        # ---------------------------------------------------------

        previous_gray = None
        last_motion_check = 0.0
        motion_detected = False

        try:
            while True:

                # -------------------------------------------------
                # READ CAMERA FRAME
                # -------------------------------------------------

                ok, frame = cap.read()

                if not ok:
                    break

                current_time = time.monotonic()

                # -------------------------------------------------
                # MOTION DETECTION
                # -------------------------------------------------

                if (
                    not self.motion_enabled
                    or (
                        current_time - last_motion_check
                        >= self.motion_check_interval
                    )
                ):

                    # Resize frame for faster motion detection.
                    new_height = int(
                        frame.shape[0]
                        * self.motion_resize_width
                        / frame.shape[1]
                    )

                    small = cv2.resize(
                        frame,
                        (
                            self.motion_resize_width,
                            new_height,
                        ),
                    )

                    gray_motion = cv2.cvtColor(
                        small,
                        cv2.COLOR_BGR2GRAY,
                    )

                    gray_motion = cv2.GaussianBlur(
                        gray_motion,
                        (21, 21),
                        0,
                    )

                    # First frame establishes the baseline.
                    if previous_gray is None:

                        motion_detected = False

                    else:

                        diff = cv2.absdiff(
                            previous_gray,
                            gray_motion,
                        )

                        _, threshold_frame = cv2.threshold(
                            diff,
                            self.motion_threshold,
                            255,
                            cv2.THRESH_BINARY,
                        )

                        threshold_frame = cv2.dilate(
                            threshold_frame,
                            None,
                            iterations=2,
                        )

                        contours, _ = cv2.findContours(
                            threshold_frame,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        motion_detected = any(
                            cv2.contourArea(contour)
                            >= self.motion_min_area
                            for contour in contours
                        )

                    previous_gray = gray_motion
                    last_motion_check = current_time

                # -------------------------------------------------
                # NO MOTION
                # -------------------------------------------------
                #
                # Camera stream continues, but expensive
                # face detection + recognition is skipped.
                #

                if not motion_detected:

                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [
                            int(
                                cv2.IMWRITE_JPEG_QUALITY
                            ),
                            80,
                        ],
                    )

                    if ok:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + encoded.tobytes()
                            + b"\r\n"
                        )

                    continue

                # -------------------------------------------------
                # MOTION DETECTED
                # -------------------------------------------------

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

                faces = self.cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=self.face_min_neighbors,
                    minSize=(
                        self.face_min_size,
                        self.face_min_size,
                    ),
                )

                # -------------------------------------------------
                # FACE PROCESSING
                # -------------------------------------------------

                for x, y, w, h in faces:

                    # ---------------------------------------------
                    # FACE QUALITY / FALSE-POSITIVE FILTER
                    # ---------------------------------------------
                    # Do not send tiny or obviously invalid Haar
                    # detections to the recognition model.
                    if (
                        w < self.face_min_size
                        or h < self.face_min_size
                    ):
                        continue

                    aspect_ratio = w / float(h)
                    if aspect_ratio < 0.55 or aspect_ratio > 1.80:
                        continue

                    face_crop = frame[
                        y : y + h,
                        x : x + w,
                    ]

                    if face_crop.size == 0:
                        continue

                    face_gray = cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2GRAY,
                    )

                    brightness = float(face_gray.mean())
                    variance = float(face_gray.var())

                    if (
                        brightness < self.face_min_brightness
                        or brightness > self.face_max_brightness
                        or variance < self.face_min_variance
                    ):
                        continue

                    label = "FACE DETECTED"
                    person_id = None
                    score = None

                    # ---------------------------------------------
                    # LOCAL FACE RECOGNITION
                    # ---------------------------------------------
                    if self.recognizer is not None:
                        match = self.recognizer.match(face_crop)

                        label = (
                            f"{match.name} "
                            f"{match.score:.2f}"
                        )

                        person_id = match.person_id
                        score = match.score

                    # ---------------------------------------------
                    # DRAW VALID FACE BOX
                    # ---------------------------------------------
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            x,
                            max(20, y - 8),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )

                    # ---------------------------------------------
                    # UNKNOWN / UNRECOGNIZED FACE
                    # ---------------------------------------------
                    # An unknown detection is NOT stored as an event
                    # immediately. This prevents random Haar boxes
                    # from filling Recent Events.
                    if person_id is None:
                        continue

                    now = time.monotonic()

                    # ---------------------------------------------
                    # MULTI-FRAME CONFIRMATION
                    # ---------------------------------------------
                    # One accidental recognition is not enough.
                    pending = self.pending_matches.get(person_id)

                    if (
                        pending is None
                        or now - float(pending["last_seen"])
                        > self.confirm_window
                    ):
                        pending = {
                            "count": 0,
                            "last_seen": now,
                            "score": 0.0,
                            "bbox": [x, y, w, h],
                        }

                    pending["count"] = int(pending["count"]) + 1
                    pending["last_seen"] = now
                    pending["score"] = max(
                        float(pending["score"]),
                        float(score or 0.0),
                    )
                    pending["bbox"] = [x, y, w, h]
                    self.pending_matches[person_id] = pending

                    # Not confirmed yet.
                    if int(pending["count"]) < self.confirm_frames:
                        continue

                    # ---------------------------------------------
                    # EVENT TYPE
                    # ---------------------------------------------
                    event_type = "known_person"

                    # ---------------------------------------------
                    # PER CAMERA + PERSON COOLDOWN
                    # ---------------------------------------------
                    cooldown_key = (
                        self.camera_id,
                        person_id,
                        event_type,
                    )

                    last_time = self.last_events.get(
                        cooldown_key,
                        0.0,
                    )

                    # ---------------------------------------------
                    # SAVE CONFIRMED EVENT
                    # ---------------------------------------------
                    if now - last_time >= self.event_cooldown:

                        snapshot_path = self._save_snapshot(
                            frame,
                            event_type,
                            person_id,
                        )

                        self.db.add_event(
                            event_type,
                            camera_id=self.camera_id,
                            person_id=person_id,
                            confidence=float(
                                pending["score"]
                            ),
                            snapshot_path=snapshot_path,
                            metadata_json=(
                                f'{{"bbox":'
                                f'{pending["bbox"]}}}'
                                f', "confirmed_frames":'
                                f'{int(pending["count"])}'
                            ),
                        )

                        self.last_events[
                            cooldown_key
                        ] = now

                    # Reset confirmation after an event so the next
                    # event must be independently confirmed again.
                    self.pending_matches.pop(person_id, None)

                # -------------------------------------------------
                # ENCODE FRAME
                # -------------------------------------------------

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),
                        80,
                    ],
                )

                if ok:

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + encoded.tobytes()
                        + b"\r\n"
                    )

        finally:
            cap.release()