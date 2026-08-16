from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from survildb.database import SurvilDB
from survilai.live_recognition import LiveRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll local face embeddings into SurvilAI"
    )
    parser.add_argument("name")
    parser.add_argument("images", nargs="+")
    parser.add_argument(
        "--checkpoint",
        default="models/survil-face-v1.pt",
    )
    parser.add_argument(
        "--db",
        default="data/survilai.db",
    )

    args = parser.parse_args()

    db = SurvilDB(args.db)

    recognizer = LiveRecognizer(
        db,
        args.checkpoint,
    )

    person_id = 1

    detector = cv2.CascadeClassifier(
        str(
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )
    )

    added = 0

    for path in args.images:

        image = cv2.imread(path)

        if image is None:
            raise ValueError(
                f"Unable to read image: {path}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            print(
                f"WARNING: No face detected in {path}. Skipping."
            )
            continue

        # Largest detected face = primary face
        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3],
        )

        print(
            f"{path}: "
            f"detected {len(faces)} face(s), "
            f"using largest face {w}x{h}"
        )

        face = image[
            y : y + h,
            x : x + w,
        ]

        embedding, quality = recognizer.embed(
            face
        )

        print(
            f"  quality={quality:.3f}"
        )

        if quality < recognizer.quality_threshold:
            print(
                "  WARNING: Low quality, skipping."
            )
            continue

        db.add_embedding(
            person_id,
            embedding.tobytes(),
            int(embedding.size),
        )

        added += 1

    if added == 0:
        raise RuntimeError(
            "No valid face embeddings were added."
        )

    db.audit(
        "person.enrolled",
        actor="local-cli",
        target_type="person",
        target_id=str(person_id),
    )

    print(
        f"enrolled person_id={person_id} "
        f"samples={added}"
    )


if __name__ == "__main__":
    main()
