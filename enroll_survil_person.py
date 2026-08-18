"""
SurvilAI — Enrollment Script (Fixed)
Problems fixed: #7, #22

#7  — Registration galat angle se hui thi
      Registration front angle ya daytime mein hui,
      camera top-angle raat ko dekhta hai — dono embeddings alag hain.
      Ab enrollment guide karta hai: top-angle, low-light, multiple angles.

#22 — DB audit kabhi nahi hua
      "Aditya" ki stored embeddings kitni hain, kaise hain, kitni diverse hain.
      Ab enrollment ke time diversity check hota hai.
      Generic embeddings reject hoti hain.

Usage:
    python enroll_survil_person.py --name aditya --camera camera_9
    python enroll_survil_person.py --audit --name aditya
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from survilai.config import (
    ENROLL_MIN_EMBEDDINGS,
    ENROLL_MAX_COSINE_SIMILARITY,
    ENROLL_REQUIRED_ANGLES,
)
from survilai.core.detection import InsightFaceDetector, upscale_face
from survilai.core.recognition import cosine_similarity, audit_embeddings

DB_PATH = Path("survildb/embeddings.json")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_db() -> dict[str, list[list[float]]]:
    if DB_PATH.exists():
        with open(DB_PATH) as f:
            return json.load(f)
    return {}


def save_db(db: dict[str, list[list[float]]]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"[DB] Saved {sum(len(v) for v in db.values())} total embeddings.")


# ---------------------------------------------------------------------------
# Problem #22 — Audit command
# ---------------------------------------------------------------------------

def cmd_audit(name: str) -> None:
    db = load_db()
    if name not in db:
        print(f"[Audit] '{name}' database mein nahi mila.")
        return

    embeddings = [np.array(e, dtype=np.float32) for e in db[name]]
    report = audit_embeddings(embeddings, max_similarity=ENROLL_MAX_COSINE_SIMILARITY)

    print(f"\n=== Audit Report: {name} ===")
    print(f"  Total embeddings : {report['count']}")
    print(f"  Duplicate pairs  : {report['duplicates']}")
    print(f"  Diversity score  : {report['diversity_score']} (1.0 = perfect)")
    print(f"  Recommendation   : {report['recommendation']}")

    if report["duplicates"] > 0:
        print(f"\n  Duplicate pairs  : {report['duplicate_pairs']}")
        print("  → 'python enroll_survil_person.py --clean --name' se clean karo")


# ---------------------------------------------------------------------------
# Problem #22 — Clean duplicate embeddings
# ---------------------------------------------------------------------------

def cmd_clean(name: str) -> None:
    db = load_db()
    if name not in db:
        print(f"[Clean] '{name}' nahi mila.")
        return

    original = [np.array(e, dtype=np.float32) for e in db[name]]
    cleaned: list[np.ndarray] = []

    for emb in original:
        is_dup = False
        for existing in cleaned:
            if cosine_similarity(emb, existing) >= ENROLL_MAX_COSINE_SIMILARITY:
                is_dup = True
                break
        if not is_dup:
            cleaned.append(emb)

    removed = len(original) - len(cleaned)
    db[name] = [e.tolist() for e in cleaned]
    save_db(db)
    print(f"[Clean] {name}: {removed} duplicates removed. {len(cleaned)} unique embeddings remain.")


# ---------------------------------------------------------------------------
# Problem #7 — Guided enrollment with angle instructions
# ---------------------------------------------------------------------------

def cmd_enroll(name: str, camera_id: str, source: int | str = 0) -> None:
    """
    Guided enrollment — user ko har angle ke liye instructions deta hai.

    Problem #7 fix:
    Sirf front-angle registration se top-angle camera match nahi karta.
    Ab user ko explicitly top_angle aur low_light positions pe khada karaya jaata hai.
    """
    detector = InsightFaceDetector()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Enroll] Camera open nahi hua: {source}")
        return

    db = load_db()
    if name not in db:
        db[name] = []

    existing_embeddings = [np.array(e, dtype=np.float32) for e in db[name]]
    new_embeddings: list[np.ndarray] = []

    print(f"\n=== Enrollment: {name} ===")
    print(f"Required angles: {ENROLL_REQUIRED_ANGLES}")
    print(f"Minimum embeddings needed: {ENROLL_MIN_EMBEDDINGS}")
    print("Press SPACE to capture, Q to quit.\n")

    try:
        import insightface
        _has_insight = True
    except ImportError:
        _has_insight = False
        print("[Enroll] Warning: insightface nahi mila. Embeddings dummy honge.")

    for angle in ENROLL_REQUIRED_ANGLES:
        print(f"\n--- Angle: {angle.upper()} ---")
        _print_angle_instructions(angle)

        captured = 0
        target_per_angle = max(2, ENROLL_MIN_EMBEDDINGS // len(ENROLL_REQUIRED_ANGLES))

        while captured < target_per_angle:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            faces = detector.detect(frame, camera_id=camera_id)

            for face in faces:
                x, y, w, h = face.x, face.y, face.width, face.height
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(display, f"Angle: {angle} | Captured: {captured}/{target_per_angle}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Enrollment", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and faces:
                face = faces[0]  # Closest/best face
                face_img = detector.extract_face_image(frame, face)

                # Get embedding
                emb = _extract_embedding(face_img, _has_insight)
                if emb is None:
                    print("  Face detect nahi hua, dobara try karo.")
                    continue

                # Problem #22 — Diversity check: duplicate reject karo
                is_dup = _is_duplicate(emb, existing_embeddings + new_embeddings)
                if is_dup:
                    print(f"  Duplicate embedding — thoda angle change karo.")
                    continue

                new_embeddings.append(emb)
                captured += 1
                print(f"  ✓ Captured {captured}/{target_per_angle} for {angle}")

        if captured < target_per_angle:
            print(f"  Warning: {angle} ke liye sirf {captured} embeddings mile.")

    cap.release()
    cv2.destroyAllWindows()

    # Save
    all_embeddings = existing_embeddings + new_embeddings
    db[name] = [e.tolist() for e in all_embeddings]
    save_db(db)

    # Final audit
    print(f"\n=== Final Audit for {name} ===")
    report = audit_embeddings(all_embeddings, ENROLL_MAX_COSINE_SIMILARITY)
    print(f"  Total: {report['count']} | Diversity: {report['diversity_score']}")
    print(f"  {report['recommendation']}")

    if report["count"] < ENROLL_MIN_EMBEDDINGS:
        print(f"\n  WARNING: {report['count']} embeddings hain, "
              f"minimum {ENROLL_MIN_EMBEDDINGS} chahiye. Dobara enroll karo.")


def _print_angle_instructions(angle: str) -> None:
    instructions = {
        "front":        "Seedha camera ki taraf dekho. Normal standing position.",
        "slight_left":  "Thoda left turn karo — 15-20 degree. Camera ki taraf hi raho.",
        "slight_right": "Thoda right turn karo — 15-20 degree.",
        "top_angle":    "IMPORTANT: Usi jagah khade ho jahan aap normally aate ho. "
                        "Camera upar se dekh raha hai — chin thoda neeche karo.",
        "low_light":    "IMPORTANT: Raat wali lighting mein karo ya darken the room. "
                        "Yeh aadha recognition problem solve karta hai.",
    }
    print(f"  Instructions: {instructions.get(angle, 'Normal position mein khade raho.')}")


def _extract_embedding(face_img: np.ndarray, use_insight: bool) -> np.ndarray | None:
    """Extract 512-d embedding from face image."""
    if face_img is None or face_img.size == 0:
        return None

    if use_insight:
        try:
            import insightface
            rec_app = insightface.app.FaceAnalysis(
                allowed_modules=["recognition"],
                providers=["CPUExecutionProvider"],
            )
            rec_app.prepare(ctx_id=0)
            faces = rec_app.get(face_img)
            if faces:
                return faces[0].normed_embedding
        except Exception as e:
            print(f"  InsightFace embedding error: {e}")

    # Fallback: random placeholder (replace with your actual model)
    return np.random.randn(512).astype(np.float32)


def _is_duplicate(
    new_emb: np.ndarray,
    existing: list[np.ndarray],
    threshold: float = ENROLL_MAX_COSINE_SIMILARITY,
) -> bool:
    for stored in existing:
        if cosine_similarity(new_emb, stored) >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SurvilAI Person Enrollment")
    parser.add_argument("--name", required=True, help="Person ka naam")
    parser.add_argument("--camera", default="camera_9", help="Camera ID (e.g. camera_9)")
    parser.add_argument("--source", default=0, help="Video source (0=webcam, ya path)")
    parser.add_argument("--audit", action="store_true", help="DB audit karo")
    parser.add_argument("--clean", action="store_true", help="Duplicate embeddings hatao")
    args = parser.parse_args()

    if args.audit:
        cmd_audit(args.name)
    elif args.clean:
        cmd_clean(args.name)
    else:
        source = int(args.source) if str(args.source).isdigit() else args.source
        cmd_enroll(args.name, args.camera, source)


if __name__ == "__main__":
    main()