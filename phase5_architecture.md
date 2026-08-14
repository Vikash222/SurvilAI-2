# Phase 5 — Multi-Camera, Tracking & Events

Phase 5 adds the commercial CCTV runtime primitives without introducing any hosted AI service.

## Pipeline

```text
Camera 1 ─┐
Camera 2 ─┼→ Camera Manager → Detection → Recognition → Tracking → Event Engine
Camera N ─┘
```

## Components

- `survilai.cameras`: local webcam/video/RTSP stream abstraction and multi-camera manager.
- `survilai.tracking`: dependency-light centroid tracker with stable track IDs and missed-frame expiry.
- `survilai.events`: local event objects and an in-process event dispatcher.

## Event types

The baseline emits `known_person` and `unknown_person`. Later phases can add configurable rules such as zone intrusion, loitering, line crossing, schedules, and alert escalation.

## Design rule

Each camera keeps its own track namespace at the application layer. A future global identity layer can correlate identities across cameras only when explicitly enabled.

This phase does not claim production-grade tracking accuracy. Benchmarking, persistence, asynchronous workers, health monitoring, backpressure, and alert delivery belong to subsequent hardening phases.
