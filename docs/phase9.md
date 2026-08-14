# Phase 9 — Live Local CCTV

Phase 9 connects registered camera sources to the local dashboard through an MJPEG stream and performs lightweight local face detection on each frame.

## Run

```bash
pip install -r requirements-phase9.txt
python run_dashboard.py
```

Open `http://127.0.0.1:5000`, add a camera with source `0` for the default webcam or an RTSP URL for a compatible camera.

## Pipeline

```text
Camera / RTSP
    ↓
OpenCV capture
    ↓
Local face detection
    ↓
Bounding boxes
    ↓
MJPEG stream
    ↓
Dashboard
    ↓
SQLite face_detected event
```

## Scope

This phase is a live-video and detection integration baseline. It does **not** claim identity recognition in the dashboard yet. The trained SurvilFaceNet recognition path must be integrated and benchmarked separately before treating labels as reliable biometric decisions.

## Production requirements

Before exposing camera streams outside localhost, add authentication/RBAC, TLS, secure camera credential handling, stream access controls, rate limits, camera health state, worker isolation, resource limits, and privacy/retention controls.
