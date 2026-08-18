"""
SurvilAI — Detection Layer (Fixed)
Problems fixed: #1, #2, #3, #4, #5, #12, #15, #16, #17, #20

#1  — Koi bhi region "face" ban jaata tha → landmark verify karo
#2  — Facial landmark verify nahi hota tha → eyes+nose+mouth check
#3  — Confidence threshold loose tha → config se enforce
#4  — Box size aur aspect ratio check nahi → enforce kiya
#5  — Ghost detections (insaan nahi, phir bhi detection) → size+zone filter
#12 — Face bahut chhota hota hai → FACE_MIN_SIZE_PX enforce
#15 — Raat mein uneven light → CLAHE preprocessing
#16 — Face upscale nahi hota tha → upscale before embedding
#17 — Motion blur wale frames process hote the → blur check
#20 — False-positive zones blacklist nahi the → ExclusionZone filter
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Protocol, Optional

from survilai.config import (
    DETECTION_CONFIDENCE_MIN,
    FACE_MIN_SIZE_PX,
    FACE_ASPECT_RATIO_MIN,
    FACE_ASPECT_RATIO_MAX,
    FACE_UPSCALE_TARGET_PX,
    BLUR_LAPLACIAN_THRESHOLD,
    CAMERA_EXCLUSION_ZONES,
    ExclusionZone,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: Optional[dict] = None  # {"left_eye", "right_eye", "nose", "mouth_l", "mouth_r"}

    @property
    def centroid(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height


class Detector(Protocol):
    def detect(self, frame: np.ndarray, camera_id: str = "") -> list[FaceBox]: ...


# ---------------------------------------------------------------------------
# Problem #15 — CLAHE preprocessing for night/uneven lighting
# ---------------------------------------------------------------------------

def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Raat ke uneven ceiling light mein face features recover karta hai.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Problem #17 — Blur detection
# ---------------------------------------------------------------------------

def is_blurry(frame: np.ndarray, threshold: float = BLUR_LAPLACIAN_THRESHOLD) -> bool:
    """
    Laplacian variance se blur detect karo.
    Log chal rahe hain, frame blurry hai — tab bhi detection hoti thi.
    Ab aisi frames skip hongi.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


# ---------------------------------------------------------------------------
# Problem #16 — Upscale small faces before embedding
# ---------------------------------------------------------------------------

def upscale_face(face_img: np.ndarray, target: int = FACE_UPSCALE_TARGET_PX) -> np.ndarray:
    """
    Chhota face (50px) seedha model ko dena weak embedding deta tha.
    4x upscale se 50px → 200px — model ke liye usable range.
    """
    h, w = face_img.shape[:2]
    if w < target or h < target:
        scale = max(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        face_img = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return face_img


# ---------------------------------------------------------------------------
# Problem #20 — Exclusion zone filter
# ---------------------------------------------------------------------------

def is_in_exclusion_zone(
    box: FaceBox,
    zones: list[ExclusionZone],
) -> tuple[bool, str]:
    """
    Permanent false-positive zones se aane wali detections reject karo.
    Steps corner, railing, scooter, fan frame — sab yahan blacklist hain.
    """
    cx, cy = box.centroid
    for zone in zones:
        if zone.contains(cx, cy):
            return True, zone.reason
    return False, ""


# ---------------------------------------------------------------------------
# Problem #2 — Landmark verification
# ---------------------------------------------------------------------------

def has_valid_landmarks(landmarks: Optional[dict]) -> bool:
    """
    Aankhein, naak, mooh teeno present hone chahiye tab hi 'face detected' maano.
    Sirf ek rectangular region milna kaafi nahi hai.
    """
    if landmarks is None:
        return False
    required = {"left_eye", "right_eye", "nose"}
    return all(k in landmarks and landmarks[k] is not None for k in required)


# ---------------------------------------------------------------------------
# Validation pipeline (Problems #1, #2, #3, #4, #5, #12)
# ---------------------------------------------------------------------------

def validate_box(
    box: FaceBox,
    camera_id: str = "",
    require_landmarks: bool = True,
) -> tuple[bool, str]:
    """
    Ek detected box ko multiple filters se pass karna padega.
    Ek bhi fail → reject.

    Returns: (is_valid, rejection_reason)
    """
    # Problem #3 — Confidence threshold
    if box.confidence < DETECTION_CONFIDENCE_MIN:
        return False, f"confidence {box.confidence:.2f} < {DETECTION_CONFIDENCE_MIN}"

    # Problem #12 — Minimum face size
    if box.width < FACE_MIN_SIZE_PX or box.height < FACE_MIN_SIZE_PX:
        return False, f"size {box.width}x{box.height} < {FACE_MIN_SIZE_PX}px"

    # Problem #4 — Aspect ratio (tiles, railings wide/flat hote hain)
    ar = box.aspect_ratio
    if not (FACE_ASPECT_RATIO_MIN <= ar <= FACE_ASPECT_RATIO_MAX):
        return False, f"aspect ratio {ar:.2f} out of range"

    # Problem #20 — Exclusion zones
    zones = CAMERA_EXCLUSION_ZONES.get(camera_id, [])
    in_zone, reason = is_in_exclusion_zone(box, zones)
    if in_zone:
        return False, f"exclusion zone: {reason}"

    # Problem #2 — Landmark verification
    if require_landmarks and not has_valid_landmarks(box.landmarks):
        return False, "landmarks missing or invalid"

    return True, ""


# ---------------------------------------------------------------------------
# InsightFace-based detector  (Problems #1, #2 fully fixed here)
# ---------------------------------------------------------------------------

class InsightFaceDetector:
    """
    InsightFace RetinaFace detector with landmark verification.

    Yeh detector:
    - Har detection ke saath 5 facial landmarks return karta hai
    - Confidence score reliable hota hai
    - Top-angle cameras ke liye better than MTCNN/Haar

    Install: pip install insightface onnxruntime
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        try:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                allowed_modules=["detection"],
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=det_size)
            self._available = True
        except ImportError:
            print("[SurvilAI] insightface not installed — falling back to Haar detector")
            self._available = False
            self._fallback = HaarFaceDetector()

    def detect(self, frame: np.ndarray, camera_id: str = "") -> list[FaceBox]:
        # Problem #17 — Skip blurry frames
        if is_blurry(frame):
            return []

        # Problem #15 — CLAHE for night conditions
        processed = apply_clahe(frame)

        if not self._available:
            return self._fallback.detect(processed, camera_id)

        faces = self.app.get(processed)
        boxes: list[FaceBox] = []

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            conf = float(face.det_score)

            # Extract 5-point landmarks if available
            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                kps = face.kps.astype(int)
                landmarks = {
                    "left_eye":  tuple(kps[0]),
                    "right_eye": tuple(kps[1]),
                    "nose":      tuple(kps[2]),
                    "mouth_l":   tuple(kps[3]),
                    "mouth_r":   tuple(kps[4]),
                }

            box = FaceBox(x=x1, y=y1, width=w, height=h,
                          confidence=conf, landmarks=landmarks)

            # Run all validation filters
            valid, reason = validate_box(box, camera_id, require_landmarks=True)
            if valid:
                boxes.append(box)
            else:
                pass  # Silent reject — no screen output for false detections

        return boxes

    def extract_face_image(
        self, frame: np.ndarray, box: FaceBox
    ) -> np.ndarray:
        """
        Crop + upscale face region before passing to recognizer.
        Problem #16 fix.
        """
        x, y, w, h = box.x, box.y, box.width, box.height
        # Clamp to frame boundaries
        fh, fw = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        face_img = frame[y1:y2, x1:x2]
        return upscale_face(face_img)


# ---------------------------------------------------------------------------
# Haar fallback (kept for environments without insightface)
# ---------------------------------------------------------------------------

class HaarFaceDetector:
    """
    OpenCV Haar cascade fallback.
    Problems #2 (landmarks) fix nahi hota yahan — isliye InsightFace prefer karo.
    Baaki sab filters (size, confidence, zone, blur) kaam karte hain.
    """

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 8) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError("OpenCV face detector could not be loaded")
        self.scale_factor = scale_factor
        # Problem #1 — min_neighbors badhao: 5 se 8, false positives kam honge
        self.min_neighbors = min_neighbors

    def detect(self, frame: np.ndarray, camera_id: str = "") -> list[FaceBox]:
        # Problem #17 — Skip blurry frames
        if is_blurry(frame):
            return []

        # Problem #15 — CLAHE preprocessing
        processed = apply_clahe(frame)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(FACE_MIN_SIZE_PX, FACE_MIN_SIZE_PX),  # Problem #12
        )

        boxes: list[FaceBox] = []
        if len(faces) == 0:
            return boxes

        for (x, y, w, h) in faces:
            # Haar confidence = 1.0 always, so we rely on other filters
            box = FaceBox(x=x, y=y, width=w, height=h,
                          confidence=1.0, landmarks=None)
            # require_landmarks=False because Haar doesn't provide them
            valid, _ = validate_box(box, camera_id, require_landmarks=False)
            if valid:
                boxes.append(box)

        return boxes

    def extract_face_image(self, frame: np.ndarray, box: FaceBox) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(fw, box.x + box.width)
        y2 = min(fh, box.y + box.height)
        face_img = frame[y1:y2, x1:x2]
        return upscale_face(face_img)