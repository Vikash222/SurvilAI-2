"""
SurvilAI — Tracking Layer (Fixed)
Problems fixed: #9, #13, #18, #19

#9  — Do logon ko ek "face" samajh liya → individual track per person
#13 — Multiple cameras pe ek hi person track nahi hoti → cross-camera ID
#18 — Multi-person tracking nahi tha → proper ID assignment
#19 — Voting system per-track → VotingBuffer ek ek track ke saath
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Iterable

from survilai.core.recognition import VotingBuffer
from survilai.config import TRACKER_MAX_DISTANCE, TRACKER_MAX_MISSED, VOTING_WINDOW_FRAMES, VOTING_MIN_VOTES


# ---------------------------------------------------------------------------
# Track dataclass — Problem #18, #19
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    camera_id: str
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    label: str = "unknown"
    score: float = 0.0
    confirmed: bool = False            # Problem #19 — voting confirmed?
    missed: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)
    voting_buffer: VotingBuffer = field(default_factory=VotingBuffer)  # Problem #19

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(
        self,
        bbox: tuple[int, int, int, int],
        label: str,
        score: float,
        confirmed: bool,
    ) -> None:
        self.bbox = bbox
        self.label = label
        self.score = score
        self.confirmed = confirmed
        self.missed = 0
        self.history.append(self.centroid)
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def display_label(self) -> str:
        """
        Problem #21 — Screen pe naam tabhi aana chahiye jab confirmed ho.
        """
        if self.confirmed and self.label != "unknown":
            return f"{self.label} {self.score:.2f}"
        return ""   # Unconfirmed track screen pe nahi dikhta


# ---------------------------------------------------------------------------
# CentroidTracker — Problem #18 (multi-person), #9 (individual IDs)
# ---------------------------------------------------------------------------

class CentroidTracker:
    """
    Problem #18 fix:
    Jab 2–4 log ek saath hain, system individual IDs assign karta hai.
    Har frame mein naya detection nahi hota — continuity maintain hoti hai.

    Problem #9 fix:
    Do logon ke beech ka "blob" ab alag-alag face detections ke baad aata hai,
    aur har detection apna alag track_id paata hai.

    Problem #19 fix:
    Har track ka apna VotingBuffer hai — ek track ka vote doosre pe nahi jaata.
    """

    def __init__(
        self,
        camera_id: str = "",
        max_distance: float = TRACKER_MAX_DISTANCE,
        max_missed: int = TRACKER_MAX_MISSED,
    ) -> None:
        self.camera_id = camera_id
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(
        self,
        detections: Iterable[tuple[tuple[int, int, int, int], str, float, bool]],
        # Each detection: (bbox, label, score, confirmed)
    ) -> list[Track]:
        """
        Match existing tracks to new detections via centroid distance.
        Unmatched detections → new tracks.
        Unmatched tracks → missed count++, drop if exceeded.
        """
        detections = list(detections)
        unmatched_det = set(range(len(detections)))

        # Match existing tracks to detections
        for track in self.tracks.values():
            best_idx = None
            best_dist = self.max_distance

            for i in unmatched_det:
                bbox, _, _, _ = detections[i]
                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                dist = hypot(
                    center[0] - track.centroid[0],
                    center[1] - track.centroid[1],
                )
                if dist < best_dist:
                    best_dist, best_idx = dist, i

            if best_idx is not None:
                bbox, label, score, confirmed = detections[best_idx]
                track.update(bbox, label, score, confirmed)
                unmatched_det.remove(best_idx)
            else:
                track.missed += 1

        # New tracks for unmatched detections
        for i in unmatched_det:
            bbox, label, score, confirmed = detections[i]
            track = Track(
                track_id=self._next_id,
                camera_id=self.camera_id,
                bbox=bbox,
                label=label,
                score=score,
                confirmed=confirmed,
                voting_buffer=VotingBuffer(
                    window=VOTING_WINDOW_FRAMES,
                    min_votes=VOTING_MIN_VOTES,
                ),
            )
            track.history.append(track.centroid)
            self.tracks[self._next_id] = track
            self._next_id += 1

        # Drop tracks that have been missing too long
        self.tracks = {
            k: v for k, v in self.tracks.items()
            if v.missed <= self.max_missed
        }

        return list(self.tracks.values())

    def active_confirmed_tracks(self) -> list[Track]:
        """
        Sirf woh tracks return karo jo voting se confirm ho chuke hain.
        Problem #21 fix — screen pe sirf confirmed tracks dikhao.
        """
        return [t for t in self.tracks.values() if t.confirmed and t.label != "unknown"]


# ---------------------------------------------------------------------------
# Problem #13 — Cross-camera person tracking
# ---------------------------------------------------------------------------

class CrossCameraTracker:
    """
    Camera 9 aur Camera 11 dono pe alag-alag detections hain.
    Same person alag-alag treat hoti thi.

    Yeh class multiple CentroidTrackers ko coordinate karta hai
    aur embedding similarity se same person ko cross-camera identify karta hai.

    Usage:
        tracker = CrossCameraTracker(["camera_9", "camera_11"])
        tracker.update("camera_9", detections_cam9)
        tracker.update("camera_11", detections_cam11)
        global_id = tracker.get_global_id("camera_9", local_track_id)
    """

    def __init__(self, camera_ids: list[str]) -> None:
        self.trackers: dict[str, CentroidTracker] = {
            cam_id: CentroidTracker(camera_id=cam_id)
            for cam_id in camera_ids
        }
        # Maps (camera_id, local_track_id) → global_person_id
        self._global_map: dict[tuple[str, int], int] = {}
        self._next_global_id = 1

        # Stores last known embedding per global ID for cross-camera match
        self._global_embeddings: dict[int, list] = {}

    def update(
        self,
        camera_id: str,
        detections: list[tuple[tuple[int, int, int, int], str, float, bool]],
    ) -> list[Track]:
        if camera_id not in self.trackers:
            self.trackers[camera_id] = CentroidTracker(camera_id=camera_id)
        return self.trackers[camera_id].update(detections)

    def assign_global_id(
        self,
        camera_id: str,
        local_track_id: int,
        embedding=None,
        similarity_fn=None,
        threshold: float = 0.55,
    ) -> int:
        """
        Local track ID ko global person ID se map karo.
        Agar embedding match hoti hai doosre camera ke track se → same global ID.
        """
        key = (camera_id, local_track_id)
        if key in self._global_map:
            return self._global_map[key]

        # Try to match with existing global embeddings
        if embedding is not None and similarity_fn is not None:
            for gid, stored_embs in self._global_embeddings.items():
                for stored in stored_embs:
                    sim = similarity_fn(embedding, stored)
                    if sim >= threshold:
                        self._global_map[key] = gid
                        self._global_embeddings[gid].append(embedding)
                        return gid

        # New global person
        gid = self._next_global_id
        self._next_global_id += 1
        self._global_map[key] = gid
        if embedding is not None:
            self._global_embeddings[gid] = [embedding]
        return gid

    def get_global_id(self, camera_id: str, local_track_id: int) -> int | None:
        return self._global_map.get((camera_id, local_track_id))

    def all_confirmed_tracks(self) -> dict[str, list[Track]]:
        """Sabhi cameras ke confirmed tracks."""
        return {
            cam_id: tracker.active_confirmed_tracks()
            for cam_id, tracker in self.trackers.items()
        }