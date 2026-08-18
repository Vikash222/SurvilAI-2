"""
SurvilAI — Central Configuration
Sabhi thresholds, limits, aur camera settings yahan hain.
Problems fixed: #3, #10, #11, #12, #20
"""
from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Problem #3 — Confidence threshold bahut loose tha (0.59–0.70 pe detections)
# Ab 0.82 se neeche kuch bhi accept nahi hoga
DETECTION_CONFIDENCE_MIN: float = 0.82

# Problem #12 — Face bahut chhota hota hai (40–80px), model ko 112x112 chahiye
# Chhote se chhota acceptable face size
FACE_MIN_SIZE_PX: int = 60          # width aur height dono

# Problem #4 — Aspect ratio check nahi tha (flat wide boxes, tiles, railing)
FACE_ASPECT_RATIO_MIN: float = 0.5  # width/height
FACE_ASPECT_RATIO_MAX: float = 2.0  # width/height

# Problem #16 — Upscale nahi hota tha; ab 4x upscale hoga chhote faces ke liye
FACE_UPSCALE_TARGET_PX: int = 200   # target resolution before embedding

# Problem #17 — Blurry frames skip nahi hote the
BLUR_LAPLACIAN_THRESHOLD: float = 60.0   # variance; is se neeche = blurry, skip


# ---------------------------------------------------------------------------
# Recognition / matching thresholds
# ---------------------------------------------------------------------------

# Problem #10 — Cosine distance threshold normal cameras ke liye set tha
# Top-angle cameras ke liye thoda relaxed, lekin false-positive safe
RECOGNITION_COSINE_THRESHOLD: float = 0.42   # distance (lower = stricter)

# Problem #8 — Single frame se decision
# Voting window: itne frames mein majority naam decide karega
VOTING_WINDOW_FRAMES: int = 10
VOTING_MIN_VOTES: int = 6           # 10 mein se 6 agar same naam -> confirm

# Problem #21 — Screen pe naam tab aaye jab confident ho
DISPLAY_MIN_CONFIDENCE: float = 0.82


# ---------------------------------------------------------------------------
# Tracker settings
# ---------------------------------------------------------------------------

# Problem #18 — Multi-person tracking
TRACKER_MAX_DISTANCE: float = 80.0  # centroid distance pixels
TRACKER_MAX_MISSED: int = 12        # frames before track drop


# ---------------------------------------------------------------------------
# Camera-wise exclusion zones  (Problem #20)
# Format: camera_id -> list of (x1, y1, x2, y2) pixel rectangles
# Yeh zones permanently false positives dete hain — railing, steps, fan frame
# ---------------------------------------------------------------------------

@dataclass
class ExclusionZone:
    x1: int
    y1: int
    x2: int
    y2: int
    reason: str = ""

    def contains(self, cx: int, cy: int) -> bool:
        """Check if centroid (cx, cy) falls inside this zone."""
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


# Camera 9 — GF-C2-Parking-BH1 (1456x816 resolution observed)
# Steps corner bottom-left, railing area bottom-center
CAMERA_EXCLUSION_ZONES: dict[str, list[ExclusionZone]] = {
    "camera_9": [
        ExclusionZone(0,   650, 350, 816, "steps/tile corner bottom-left"),
        ExclusionZone(350, 720, 750, 816, "railing base area"),
        ExclusionZone(0,   0,   200, 200, "tree shadow top-left"),
    ],
    "camera_11": [
        ExclusionZone(100, 400, 450, 620, "staircase ramp — repeated ghost"),
        ExclusionZone(800, 280, 1100, 500, "fan/grille near desk"),
        ExclusionZone(0,   550, 300, 720, "plant pot + ground"),
    ],
}


# ---------------------------------------------------------------------------
# Registration / enrollment settings  (Problem #7, #22)
# ---------------------------------------------------------------------------

# Problem #22 — DB audit: minimum diversity required during registration
ENROLL_MIN_EMBEDDINGS: int = 8      # kam se kam 8 angles se register karo
ENROLL_MAX_COSINE_SIMILARITY: float = 0.98  # agar do embeddings itni similar
                                             # hain toh duplicate hai, skip karo

# Problem #7 — Registration angle mismatch
# Enrollment ke time yeh angles cover karne chahiye
ENROLL_REQUIRED_ANGLES: list[str] = [
    "front",
    "slight_left",
    "slight_right",
    "top_angle",    # camera ke angle se match karne ke liye zaroori
    "low_light",    # raat ke conditions ke liye
]