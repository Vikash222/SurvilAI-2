# Phase 10 — Live Face Recognition

Phase 10 connects the local SurvilFaceNet embedding model to the live CCTV stream.

## Pipeline

Camera -> local face detection -> 112x112 preprocessing -> SurvilFaceNet -> 128-D normalized embedding -> SQLite gallery -> cosine similarity -> Known/Unknown -> event.

## Enrollment

Use authorized images only. Each image must contain exactly one detectable face.

```bash
python enroll_survil_person.py "Person Name" data/person/01.jpg data/person/02.jpg --checkpoint models/survil-face-v1.pt
```

## Live dashboard

```bash
export SURVILAI_CHECKPOINT=models/survil-face-v1.pt
export SURVILAI_MATCH_THRESHOLD=0.72
python run_dashboard.py
```

Add a camera with source `0` or an RTSP URL, then open the dashboard locally.

## Important

The threshold is an operational setting, not a proof of identity. The model must be benchmarked on representative, identity-disjoint validation data before any security-critical deployment. False accepts/rejects, lighting, pose, camera quality, demographic performance, spoofing and privacy requirements need dedicated validation.
