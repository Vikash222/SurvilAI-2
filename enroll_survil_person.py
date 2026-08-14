from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from survildb.database import SurvilDB
from survilai.live_recognition import LiveRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll local face embeddings into SurvilAI")
    parser.add_argument("name")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="models/survil-face-v1.pt")
    parser.add_argument("--db", default="data/survilai.db")
    args = parser.parse_args()

    db = SurvilDB(args.db)
    recognizer = LiveRecognizer(db, args.checkpoint)
    person_id = db.add_person(args.name)
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    added = 0
    for path in args.images:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) != 1:
            raise ValueError(f"Expected exactly one face in {path}; found {len(faces)}")
        x, y, w, h = faces[0]
        embedding = recognizer.embed(image[y:y+h, x:x+w])
        db.add_embedding(person_id, embedding.tobytes(), int(embedding.size))
        added += 1
    db.audit("person.enrolled", actor="local-cli", target_type="person", target_id=str(person_id))
    print(f"enrolled person_id={person_id} samples={added}")


if __name__ == "__main__":
    main()
