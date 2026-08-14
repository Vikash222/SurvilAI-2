# SurvilAI

Privacy-first, offline-capable AI CCTV security platform.

## Phase 1 — Local Vision Engine

Phase 1 establishes the local computer-vision foundation for SurvilAI. The core runtime does not call OpenAI, Gemini, cloud face-recognition APIs, Firebase, Supabase, or other hosted AI services.

### Current Phase 1 flow

```text
Webcam / Local Video
        ↓
   VideoSource
        ↓
   Face Detector
        ↓
 Recognition Adapter
        ↓
 Local Observation
```

### Included

- OpenCV video source abstraction
- Local OpenCV face detection baseline
- Pluggable recognition interface
- Local cosine-similarity utility
- Processing pipeline with bounding boxes and labels
- CLI runner
- Initial unit tests

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run_phase1.py
```

Press **q** to stop the displayed camera window.

Use a local video file:

```bash
python run_phase1.py --source ./data/test.mp4
```

### Architecture rule

The recognition implementation is intentionally an adapter. Phase 1 does **not** hide a third-party cloud API behind the interface. The next milestone is to implement SurvilAI's own local recognition model/pipeline and local gallery.

### Roadmap

- [x] Phase 1 architecture
- [x] Local video source
- [x] Local face detection baseline
- [x] Recognition interface
- [x] Local similarity utility
- [ ] SurvilAI-owned recognition model
- [ ] Local face enrollment/gallery
- [ ] RTSP/ONVIF camera support
- [ ] Multi-camera engine
- [ ] Local event database
- [ ] Dashboard
- [ ] Alerts and rules
- [ ] Enterprise edge architecture

## Commercialization note

Before any commercial release, all model weights, datasets, dependencies, and licenses must be reviewed for commercial use and redistribution. Privacy, retention, access control, and applicable data-protection requirements will be designed into later phases.
