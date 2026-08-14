from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """Runtime configuration for the local Phase 1 engine."""

    source: str = "0"
    display: bool = True
    max_width: int = 1280
    max_height: int = 720
    process_every_n_frames: int = 1
