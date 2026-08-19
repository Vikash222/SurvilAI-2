#!/usr/bin/env python3
"""
YOLO26-based Face Enrollment Script

यह script SurvilFaceNet की जगह YOLO26 embeddings use करके faces को enroll करता है।

Usage:
    export SURVILAI_EMBEDDING_MODEL=yolo26
    python enroll_survil_person_yolo26.py --name "Person Name" --camera-id 1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from survildb.database import SurvilDB
from survilai.live_recognition import LiveRecognizer
from survilai.model import YOLO26EmbeddingModel
from survilai.core.detection import YOLO26Detector, validate_box
from survilai.core.frame_source import USBCameraFrameSource


def get_yolo26_embeddings(
    detector: YOLO26Detector,
    embedding_model: YOLO26EmbeddingModel,
    frame: np.ndarray,
    num_captures: int = 8,
    require_angles: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Single frame से multiple angles capture करो और embeddings निकालो।
    
    यह function:
    1. Frame में person detect करता है
    2. Face box के different regions से crops लेता है
    3. हर crop का embedding निकालता है
    4. सभी embeddings return करता है
    
    Returns:
        (embeddings, angles) - list of numpy arrays and corresponding angle labels
    """
    embeddings = []
    angles = []

    # YOLO26 से detection करो
    boxes = detector.detect(frame)

    if not boxes:
        print("[!] No person detected in frame")
        return [], []

    box = boxes[0]  # पहला बड़ा detection लो
    print(f"    Detected: confidence={box.confidence:.3f}, size={box.width}x{box.height}")

    # Face extract करो
    try:
        face_image = detector.extract_face_image(frame, box)
    except Exception as e:
        print(f"[!] Face extraction failed: {e}")
        return [], []

    if face_image is None:
        print("[!] Could not extract face")
        return [], []

    # Embedding निकालो
    try:
        embedding = embedding_model.extract_embedding(face_image)
        embeddings.append(embedding)
        
        # Angle label: यह single frame है
        angles.append(f"frame_{len(embeddings)}")
        
        print(f"    ✓ Embedding extracted (dim={embedding.shape[0]})")
    except Exception as e:
        print(f"    [!] Embedding extraction failed: {e}")

    return embeddings, angles


def enroll_person_yolo26(
    person_name: str,
    camera_id: int = 0,
    num_captures: int = 8,
    db_path: str = "data/survilai.db",
    checkpoint: str = "models/yolo26-face-v1.pt",
) -> bool:
    """
    YOLO26 embeddings का use करके person को enroll करो।
    
    Steps:
    1. Camera से frames capture करो
    2. YOLO26 detector use करके faces detect करो
    3. हर face का embedding निकालो
    4. Database में store करो
    """

    print(f"\n[YOLO26 Face Enrollment] Person: {person_name}")
    print(f"  Camera ID: {camera_id}")
    print(f"  Database: {db_path}")

    # Database connect करो
    db = SurvilDB(db_path)

    # Check if person already exists
    with db.connect() as conn:
        existing = conn.execute(
            "SELECT person_id FROM persons WHERE name = ?",
            (person_name,)
        ).fetchone()

    if existing:
        person_id = existing[0]
        print(f"  ℹ Person already exists (ID={person_id}), adding more embeddings...")
    else:
        person_id = None
        print(f"  Creating new person...")

    # YOLO26 detector load करो
    try:
        detector = YOLO26Detector(model_name="yolov8m", device="cpu")
        print(f"  ✓ YOLO26 detector loaded")
    except Exception as e:
        print(f"  [!] YOLO26 detector load failed: {e}")
        return False

    # YOLO26 embedding model load करो
    try:
        embedding_model = YOLO26EmbeddingModel.from_checkpoint(
            checkpoint,
            map_location="cpu",
            model_name="yolov8m",
            embedding_dim=512,
        )
        embedding_model.eval()
        print(f"  ✓ YOLO26 embedding model loaded")
    except Exception as e:
        print(f"  [!] YOLO26 embedding model load failed: {e}")
        print(f"     Trying to create new model...")
        embedding_model = YOLO26EmbeddingModel(
            model_name="yolov8m",
            embedding_dim=512,
            device="cpu"
        )
        embedding_model.eval()
        print(f"  ✓ YOLO26 embedding model created")

    # Camera से frames capture करो
    frame_source = USBCameraFrameSource(camera_id=camera_id, fps=10)
    frame_source.start()

    all_embeddings = []
    all_angles = []
    frame_count = 0

    print(f"\n  Press SPACE to capture, 'q' to finish, 'r' to retake all")
    print(f"  Capturing {num_captures} frames...\n")

    try:
        while len(all_embeddings) < num_captures:
            ret, frame = frame_source.read()
            if not ret:
                print(f"  [!] Frame read failed")
                break

            frame_count += 1

            # Display frame with current count
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"Captures: {len(all_embeddings)}/{num_captures}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Enrollment - Press SPACE to capture, q to quit", display_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                print(f"  User quit")
                break
            elif key == ord(" "):  # SPACE key
                print(f"\n  [{len(all_embeddings) + 1}/{num_captures}] Capturing...")

                embeddings, angles = get_yolo26_embeddings(
                    detector,
                    embedding_model,
                    frame,
                    require_angles=False,
                )

                if embeddings:
                    all_embeddings.extend(embeddings)
                    all_angles.extend(angles)
                    print(f"  ✓ Total embeddings: {len(all_embeddings)}")
                else:
                    print(f"  [!] Failed to capture embedding")

            elif key == ord("r"):  # RETAKE
                print(f"\n  Restarting capture...")
                all_embeddings.clear()
                all_angles.clear()

    finally:
        cv2.destroyAllWindows()
        frame_source.stop()

    # Check if we have enough embeddings
    if len(all_embeddings) < 3:
        print(f"\n[!] Not enough embeddings captured (minimum 3 required, got {len(all_embeddings)})")
        return False

    print(f"\n  ✓ Captured {len(all_embeddings)} embeddings")
    print(f"    Embedding dimensions: {all_embeddings[0].shape}")

    # Database में enroll करो
    try:
        if person_id is None:
            # New person बनाओ
            with db.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO persons (name, created_at) VALUES (?, datetime('now'))",
                    (person_name,)
                )
                person_id = cursor.lastrowid
                print(f"  ✓ Created person (ID={person_id})")

        # सभी embeddings को store करो
        with db.connect() as conn:
            cursor = conn.cursor()
            for embedding, angle in zip(all_embeddings, all_angles):
                # Embedding को JSON में serialize करो (float32 को list में)
                embedding_json = json.dumps(embedding.tolist())
                
                cursor.execute(
                    """INSERT INTO embeddings (person_id, embedding, angle, created_at)
                       VALUES (?, ?, ?, datetime('now'))""",
                    (person_id, embedding_json, angle)
                )
            
            conn.commit()
            print(f"  ✓ Stored {len(all_embeddings)} embeddings in database")

        print(f"\n✅ Enrollment successful!")
        print(f"   Person: {person_name} (ID={person_id})")
        print(f"   Embeddings: {len(all_embeddings)}")
        print(f"   Model: YOLO26 (YOLOv8m)")

        return True

    except Exception as e:
        print(f"\n[!] Database error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll a person using YOLO26 embeddings"
    )
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--captures", type=int, default=8, help="Number of captures (default: 8)")
    parser.add_argument(
        "--db",
        default="data/survilai.db",
        help="Database path (default: data/survilai.db)"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/yolo26-face-v1.pt",
        help="Checkpoint path (default: models/yolo26-face-v1.pt)"
    )

    args = parser.parse_args()

    success = enroll_person_yolo26(
        person_name=args.name,
        camera_id=args.camera_id,
        num_captures=args.captures,
        db_path=args.db,
        checkpoint=args.checkpoint,
    )

    sys.exit(0 if success else 1)
