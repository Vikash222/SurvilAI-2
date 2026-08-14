from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Iterable


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str = "Unknown"
    score: float = 0.0
    missed: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class CentroidTracker:
    """Small dependency-free tracker for the Phase 5 baseline."""

    def __init__(self, max_distance: float = 80.0, max_missed: int = 12) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: Iterable[tuple[tuple[int, int, int, int], str, float]]) -> list[Track]:
        detections = list(detections)
        unmatched = set(range(len(detections)))
        for track in self.tracks.values():
            best = None
            best_distance = self.max_distance
            for i in unmatched:
                bbox, _, _ = detections[i]
                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance = hypot(center[0] - track.centroid[0], center[1] - track.centroid[1])
                if distance < best_distance:
                    best_distance, best = distance, i
            if best is not None:
                bbox, label, score = detections[best]
                track.bbox, track.label, track.score, track.missed = bbox, label, score, 0
                track.history.append(track.centroid)
                unmatched.remove(best)
            else:
                track.missed += 1

        for i in unmatched:
            bbox, label, score = detections[i]
            track = Track(self._next_id, bbox, label, score)
            track.history.append(track.centroid)
            self.tracks[self._next_id] = track
            self._next_id += 1

        self.tracks = {k: v for k, v in self.tracks.items() if v.missed <= self.max_missed}
        return list(self.tracks.values())
