# YOLO26 Embedding Model Integration

## Overview

यह document YOLO26 (YOLOv8) के internal features को face identity embeddings के लिए use करने की implementation को explain करता है।

### Architecture

```
Live Camera Frame
    ↓
YOLO26 Person Detection (yolov8m model)
    ↓
Face Extraction & Upscaling (detector.extract_face_image)
    ↓
YOLO26 Backbone Feature Extraction
    ↓
Projection Layer (1024 → 512 dims)
    ↓
L2 Normalization
    ↓
Cosine Similarity Matching
```

## Files Created/Modified

### New Files

1. **survilai/model/yolo26_embedding.py**
   - `YOLO26EmbeddingModel` class
   - Extracts features from YOLO26 backbone
   - Projects to 512-dimensional embedding space
   - L2 normalization for cosine similarity

2. **enroll_survil_person_yolo26.py**
   - YOLO26 embeddings के साथ face enrollment script
   - Camera से real-time capture करता है
   - Database में YOLO26 embeddings store करता है

### Modified Files

1. **survilai/model/__init__.py**
   - YOLO26EmbeddingModel को export किया

2. **survilai/live_recognition.py**
   - `model_type` parameter add किया
   - "survilface" (default) और "yolo26" दोनों को support करता है
   - Graceful fallback: YOLO26 fail हो तो SurvilFaceNet use करो

3. **survilai/config.py**
   - EMBEDDING_MODEL configuration add किया
   - Environment variable: `SURVILAI_EMBEDDING_MODEL`

4. **dashboard/app.py**
   - LiveRecognizer initialization में `model_type` add किया
   - Environment-based model selection

5. **dashboard/camera_stream.py**
   - LiveRecognizer initialization में `model_type` add किया
   - `SURVILAI_EMBEDDING_MODEL` env var support

## Usage

### 1. Enable YOLO26 Embeddings

Environment variable set करो:

```bash
export SURVILAI_EMBEDDING_MODEL=yolo26
export SURVILAI_DETECTOR=yolo26
```

### 2. Dashboard चलाओ (YOLO26 embeddings के साथ)

```bash
export SURVILAI_EMBEDDING_MODEL=yolo26
python run_dashboard.py
```

**Output:**
```
[LiveRecognizer] Using YOLO26 embedding model on cpu
[YOLO26Detector] Loaded yolov8m
```

### 3. Face Enroll करो (YOLO26 embeddings)

```bash
python enroll_survil_person_yolo26.py --name "Vikash" --camera-id 0 --captures 8
```

**Workflow:**
- Camera से frame दिखेगा
- SPACE key दबाओ → face capture होगा
- 8 captures लो
- Database में YOLO26 embeddings store होंगे

### 4. Live Recognition चलाओ

Dashboard में live camera stream देखो:
- YOLO26 से person detect होगा
- YOLO26 embeddings से identity match होगी
- Matching accuracy SurvilFaceNet से कम हो सकती है (experimental)

## Technical Details

### YOLO26EmbeddingModel

```python
from survilai.model import YOLO26EmbeddingModel

# Initialize
model = YOLO26EmbeddingModel(
    model_name="yolov8m",      # yolov8n/s/m/l/x
    embedding_dim=512,         # output dimension
    device="cpu"               # or "cuda"
)

# Extract embedding from face image
embedding = model.extract_embedding(face_image)  # (512,)
```

**Features:**
- YOLO26 backbone से features extract करता है
- Projection layer: 1024 → 512 dims
- L2 normalization: ||embedding|| = 1
- Cosine similarity matching compatible

### LiveRecognizer Configuration

```python
from survilai.live_recognition import LiveRecognizer

# SurvilFaceNet (default)
recognizer = LiveRecognizer(
    db,
    checkpoint="models/survil-face-v1.pt",
    model_type="survilface"  # या "yolo26"
)

# YOLO26
recognizer = LiveRecognizer(
    db,
    checkpoint="models/survil-face-v1.pt",  # checkpoint path (कोई भी)
    model_type="yolo26"
)
```

### Environment Variables

| Variable | Values | Default |
|----------|--------|---------|
| `SURVILAI_EMBEDDING_MODEL` | `survilface`, `yolo26` | `survilface` |
| `SURVILAI_DETECTOR` | `yolo26`, `insightface`, `haar` | `yolo26` |
| `SURVILAI_CHECKPOINT` | Path to checkpoint | `models/survil-face-v1.pt` |

## Performance Notes

### YOLO26 vs SurvilFaceNet

| Aspect | SurvilFaceNet | YOLO26 |
|--------|----------------|---------|
| Accuracy | ✓ Optimized for faces | ⚠ Optimized for detection |
| Speed | Good | Fast (YOLOv8) |
| Robustness | High | Medium (experimental) |
| Threshold | 0.72 (cosine distance) | May need tuning |

### Recommendations

1. **SurvilFaceNet** (production) — Better accuracy
2. **YOLO26** (experimental) — Faster, unified detection+embedding

### If accuracy drops with YOLO26:

1. **Adjust thresholds:**
   ```bash
   export SURVILAI_MATCH_THRESHOLD=0.60  # relax threshold
   ```

2. **Increase enrollment samples:**
   ```bash
   python enroll_survil_person_yolo26.py --name "Person" --captures 16
   ```

3. **Use larger YOLO model:**
   - Change `model_name="yolov8l"` in YOLO26EmbeddingModel
   - Trade-off: slower but potentially better features

## Database Structure

YOLO26 embeddings database में same format में store होते हैं:

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL,
    embedding TEXT NOT NULL,  -- JSON: [f1, f2, ..., f512]
    angle TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);
```

## Migration from SurvilFaceNet to YOLO26

यदि पहले SurvilFaceNet embeddings use कर रहे हो:

### Option 1: Fresh Enrollment
```bash
# सभी persons को re-enroll करो
python enroll_survil_person_yolo26.py --name "Person1" --captures 8
python enroll_survil_person_yolo26.py --name "Person2" --captures 8
```

### Option 2: Feature Space Adaptation
```python
# YOLO26 से gallery regenerate करो
python -c "
from survilai.live_recognition import LiveRecognizer
from survildb.database import SurvilDB

db = SurvilDB('data/survilai.db')
recognizer = LiveRecognizer(db, 'models/survil-face-v1.pt', model_type='yolo26')
recognizer.reload_gallery()
print(f'Gallery reloaded with YOLO26 embeddings')
"
```

## Troubleshooting

### "YOLO26 model not available"
```bash
pip install ultralytics>=8.0
```

### "Embedding dimension mismatch"
- Check YOLO26EmbeddingModel embedding_dim setting
- Default: 512 (same as SurvilFaceNet)

### "Low recognition accuracy with YOLO26"
1. Ensure good enrollment samples (8+ captures)
2. Check lighting conditions during capture and recognition
3. Adjust thresholds in config.py
4. Try larger YOLO model (yolov8l)

### Fallback to SurvilFaceNet
यदि YOLO26 initialization fail हो:
```python
# LiveRecognizer automatically fallback करता है
recognizer = LiveRecognizer(
    db,
    checkpoint,
    model_type="yolo26"  # fail हो तो SurvilFaceNet use होगा
)
```

## Future Improvements

1. **YOLO26-Face**: Specialized face detection model (currently using generic yolov8m)
2. **Feature Fine-tuning**: YOLO26 backbone को face embeddings के लिए fine-tune करना
3. **Dimension optimization**: 512-dim से कम dimensions try करना (speed/accuracy trade-off)
4. **Metric learning**: YOLO26 embeddings को contrastive loss से optimize करना

## References

- YOLO26 (YOLOv8): https://docs.ultralytics.com
- SurvilFaceNet: survilai/model/network.py
- LiveRecognizer: survilai/live_recognition.py
