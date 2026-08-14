from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


@dataclass(frozen=True)
class Event:
    event_type: str
    camera_id: str
    track_id: int
    label: str
    score: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EventEngine:
    """Rule-light event dispatcher; persistence/alerts can be added later."""

    def __init__(self) -> None:
        self.handlers: list[Callable[[Event], None]] = []

    def on_event(self, handler: Callable[[Event], None]) -> None:
        self.handlers.append(handler)

    def emit(self, event: Event) -> None:
        for handler in tuple(self.handlers):
            handler(event)

    def identity_event(self, camera_id: str, track_id: int, label: str, score: float) -> Event:
        event_type = "known_person" if label != "Unknown" else "unknown_person"
        return Event(event_type, camera_id, track_id, label, score)
