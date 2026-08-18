from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

load_dotenv()

from survildb.database import SurvilDB
from survilai.live_recognition import LiveRecognizer
from .camera_stream import CameraStream


def create_app(db_path: str | Path = "data/survilai.db") -> Flask:

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Maximum upload size: 20 MB per request
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    db = SurvilDB(db_path)

    # =========================================================
    # FACE ENROLLMENT / RECOGNITION
    # =========================================================

    checkpoint = os.getenv(
        "SURVILAI_CHECKPOINT",
        "models/survil-face-v1.pt",
    )

    recognizer = LiveRecognizer(
        db,
        checkpoint,
        threshold=float(
            os.getenv(
                "SURVILAI_MATCH_THRESHOLD",
                "0.72",
            )
        ),
        quality_threshold=float(
            os.getenv(
                "SURVILAI_QUALITY_THRESHOLD",
                "0.25",
            )
        ),
    )

    face_detector = cv2.CascadeClassifier(
        str(
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )
    )

    enrollment_dir = Path(
        os.getenv(
            "SURVILAI_ENROLLMENT_DIR",
            "my_photos",
        )
    )

    enrollment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # DASHBOARD
    # =========================================================

    @app.get("/")
    def index():
        return render_template("index.html")

    # =========================================================
    # HEALTH
    # =========================================================

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "database": "local",
                "video": "mjpeg",
            }
        )

    # =========================================================
    # EVENTS
    # =========================================================

    @app.get("/api/events")
    def events():
        return jsonify(
            [
                dict(row)
                for row in db.recent_events(100)
            ]
        )

    # =========================================================
    # SNAPSHOTS
    # =========================================================

    @app.get("/snapshots/<path:filename>")
    def snapshot(filename: str):

        snapshot_dir = Path(
            "snapshots"
        ).resolve()

        requested = (
            snapshot_dir / filename
        ).resolve()

        # Security: only allow files inside snapshots/
        if snapshot_dir not in requested.parents:
            return jsonify(
                {
                    "error": "invalid snapshot path"
                }
            ), 400

        if not requested.is_file():
            return jsonify(
                {
                    "error": "snapshot not found"
                }
            ), 404

        return send_from_directory(
            snapshot_dir,
            requested.relative_to(
                snapshot_dir
            ),
        )

    # =========================================================
    # CAMERAS
    # =========================================================

    @app.get("/api/cameras")
    def cameras():

        with db.connect() as conn:

            rows = conn.execute(
                """
                SELECT
                    id,
                    name,
                    source,
                    enabled,
                    created_at
                FROM cameras
                ORDER BY id
                """
            ).fetchall()

        return jsonify(
            [
                dict(row)
                for row in rows
            ]
        )

    @app.post("/api/cameras")
    def add_camera():

        payload = (
            request.get_json(
                silent=True
            )
            or {}
        )

        name = str(
            payload.get(
                "name",
                "",
            )
        ).strip()

        source = str(
            payload.get(
                "source",
                "",
            )
        ).strip()

        if not name or not source:

            return jsonify(
                {
                    "error":
                        "name and source are required"
                }
            ), 400

        return jsonify(
            {
                "id": db.add_camera(
                    name,
                    source,
                )
            }
        ), 201

    @app.delete(
        "/api/cameras/<int:camera_id>"
    )
    def delete_camera(camera_id: int):

        with db.connect() as conn:

            cur = conn.execute(
                """
                DELETE FROM cameras
                WHERE id = ?
                """,
                (camera_id,),
            )

        if cur.rowcount == 0:

            return jsonify(
                {
                    "error":
                        "camera not found"
                }
            ), 404

        return jsonify(
            {
                "deleted": camera_id
            }
        )

    # =========================================================
    # PEOPLE
    # =========================================================

    @app.get("/api/people")
    def people():

        rows = db.get_people()

        return jsonify(
            [
                dict(row)
                for row in rows
            ]
        )

    # ---------------------------------------------------------
    # GET SINGLE PERSON
    # ---------------------------------------------------------

    @app.get(
        "/api/people/<int:person_id>"
    )
    def get_person(person_id: int):

        person = db.get_person(
            person_id
        )

        if person is None:

            return jsonify(
                {
                    "error":
                        "person not found"
                }
            ), 404

        return jsonify(
            dict(person)
        )

    # ---------------------------------------------------------
    # ADD PERSON
    # ---------------------------------------------------------

    @app.post("/api/people")
    def add_person():

        payload = (
            request.get_json(
                silent=True
            )
            or {}
        )

        name = str(
            payload.get(
                "name",
                "",
            )
        ).strip()

        roll_number = str(
            payload.get(
                "roll_number",
                "",
            )
        ).strip()

        if not name:

            return jsonify(
                {
                    "error":
                        "name is required"
                }
            ), 400

        try:

            person_id = db.add_person(
                name,
                roll_number or None,
            )

        except Exception as exc:

            return jsonify(
                {
                    "error":
                        "person could not be created",
                    "detail":
                        str(exc),
                }
            ), 409

        return jsonify(
            {
                "id": person_id,
                "name": name,
                "roll_number": roll_number,
            }
        ), 201

    # ---------------------------------------------------------
    # DELETE PERSON
    # ---------------------------------------------------------

    @app.delete(
        "/api/people/<int:person_id>"
    )
    def delete_person(person_id: int):

        deleted = db.delete_person(
            person_id
        )

        if not deleted:

            return jsonify(
                {
                    "error":
                        "person not found"
                }
            ), 404

        # Reload recognition gallery after deletion
        recognizer.reload_gallery()

        return jsonify(
            {
                "deleted": person_id
            }
        )

    # =========================================================
    # PERSON FACE IMAGES
    # =========================================================

    # ---------------------------------------------------------
    # VIEW ENROLLED IMAGES
    # ---------------------------------------------------------

    @app.get(
        "/api/people/<int:person_id>/images"
    )
    def person_images(person_id: int):

        person = db.get_person(
            person_id
        )

        if person is None:

            return jsonify(
                {
                    "error":
                        "person not found"
                }
            ), 404

        images = db.get_face_images(
            person_id
        )

        return jsonify(
            [
                dict(row)
                for row in images
            ]
        )

    # ---------------------------------------------------------
    # SERVE ENROLLED IMAGE
    # ---------------------------------------------------------

    @app.get(
        "/person-images/<path:filename>"
    )
    def person_image(filename: str):

        image_root = (
            enrollment_dir.resolve()
        )

        requested = (
            image_root / filename
        ).resolve()

        # Security: prevent ../ traversal
        if image_root not in requested.parents:

            return jsonify(
                {
                    "error":
                        "invalid image path"
                }
            ), 400

        if not requested.is_file():

            return jsonify(
                {
                    "error":
                        "image not found"
                }
            ), 404

        return send_from_directory(
            image_root,
            requested.relative_to(
                image_root
            ),
        )

    # ---------------------------------------------------------
    # ADD FACE IMAGES
    # ---------------------------------------------------------

    @app.post(
        "/api/people/<int:person_id>/images"
    )
    def add_person_images(person_id: int):

        person = db.get_person(
            person_id
        )

        if person is None:

            return jsonify(
                {
                    "error":
                        "person not found"
                }
            ), 404

        files = request.files.getlist(
            "images"
        )

        if not files:

            # Also support a single "image"
            single = request.files.get(
                "image"
            )

            if single is not None:

                files = [single]

        if not files:

            return jsonify(
                {
                    "error":
                        "no images uploaded"
                }
            ), 400

        person_dir = (
            enrollment_dir
            / str(person_id)
        )

        person_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        added = []
        skipped = []

        for uploaded in files:

            if not uploaded or not uploaded.filename:

                continue

            original_name = (
                uploaded.filename
            )

            # -------------------------------------------------
            # Read uploaded image
            # -------------------------------------------------

            try:

                raw = uploaded.read()

                if not raw:

                    skipped.append(
                        {
                            "filename":
                                original_name,
                            "reason":
                                "empty file",
                        }
                    )

                    continue

                image = cv2.imdecode(
                    __import__(
                        "numpy"
                    ).frombuffer(
                        raw,
                        dtype=__import__(
                            "numpy"
                        ).uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )

            except Exception as exc:

                skipped.append(
                    {
                        "filename":
                            original_name,
                        "reason":
                            f"image read failed: {exc}",
                    }
                )

                continue

            if image is None:

                skipped.append(
                    {
                        "filename":
                            original_name,
                        "reason":
                            "invalid image",
                    }
                )

                continue

            # -------------------------------------------------
            # Face detection
            # -------------------------------------------------

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(80, 80),
            )

            if len(faces) == 0:

                skipped.append(
                    {
                        "filename":
                            original_name,
                        "reason":
                            "no face detected",
                    }
                )

                continue

            # Largest face = primary face
            x, y, w, h = max(
                faces,
                key=lambda box:
                    box[2] * box[3],
            )

            face = image[
                y:y + h,
                x:x + w,
            ]

            # -------------------------------------------------
            # Generate embedding
            # -------------------------------------------------

            try:

                embedding, quality = (
                    recognizer.embed(face)
                )

            except Exception as exc:

                skipped.append(
                    {
                        "filename":
                            original_name,
                        "reason":
                            f"embedding failed: {exc}",
                    }
                )

                continue

            if (
                quality
                < recognizer.quality_threshold
            ):

                skipped.append(
                    {
                        "filename":
                            original_name,
                        "reason":
                            (
                                "low image quality "
                                f"({quality:.3f})"
                            ),
                    }
                )

                continue

            # -------------------------------------------------
            # Save original image
            # -------------------------------------------------

            extension = (
                Path(
                    original_name
                ).suffix.lower()
            )

            if extension not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:

                extension = ".jpg"

            filename = (
                f"{uuid.uuid4().hex}"
                f"{extension}"
            )

            image_path = (
                person_dir / filename
            )

            # Save normalized JPEG/PNG data
            save_ok = cv2.imwrite(
                str(image_path),
                image,
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    92,
                ],
            )

            if not save_ok:

                skipped.append(
                    {
                        "filename":
                            original_name,
                        "reason":
                            "failed to save image",
                    }
                )

                continue

            # -------------------------------------------------
            # Save embedding
            # -------------------------------------------------

            try:

                embedding_id = (
                    db.add_embedding(
                        person_id,
                        embedding.tobytes(),
                        int(
                            embedding.size
                        ),
                    )
                )

                # Store path relative to enrollment directory
                relative_image_path = str(
                    image_path.relative_to(
                        enrollment_dir
                    )
                )

                image_id = (
                    db.add_face_image(
                        person_id,
                        embedding_id,
                        relative_image_path,
                    )
                )

            except Exception as exc:

                # Remove image if DB operation failed
                try:
                    image_path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass

                skipped.append(
                    {
                        "filename":
                            original_name,
                        "reason":
                            f"database error: {exc}",
                    }
                )

                continue

            added.append(
                {
                    "id": image_id,
                    "embedding_id":
                        embedding_id,
                    "filename":
                        original_name,
                    "image_path":
                        relative_image_path,
                    "quality":
                        round(
                            float(quality),
                            3,
                        ),
                }
            )

        # -----------------------------------------------------
        # Reload recognition gallery
        # -----------------------------------------------------

        if added:

            recognizer.reload_gallery()

        return jsonify(
            {
                "person_id": person_id,
                "added": added,
                "added_count":
                    len(added),
                "skipped": skipped,
                "skipped_count":
                    len(skipped),
            }
        ), 201 if added else 400

    # ---------------------------------------------------------
    # DELETE ONE ENROLLED IMAGE
    # ---------------------------------------------------------

    @app.delete(
        "/api/people/images/<int:image_id>"
    )
    def delete_person_image(image_id: int):

        image = db.get_face_image(
            image_id
        )

        if image is None:

            return jsonify(
                {
                    "error":
                        "image not found"
                }
            ), 404

        image_path = (
            enrollment_dir
            / image["image_path"]
        ).resolve()

        image_root = (
            enrollment_dir.resolve()
        )

        # Security check
        if image_root not in image_path.parents:

            return jsonify(
                {
                    "error":
                        "invalid image path"
                }
            ), 400

        deleted = db.delete_face_image(
            image_id
        )

        if not deleted:

            return jsonify(
                {
                    "error":
                        "image could not be deleted"
                }
            ), 404

        # Delete physical image
        try:

            image_path.unlink(
                missing_ok=True
            )

        except Exception:

            pass

        # Reload gallery
        recognizer.reload_gallery()

        return jsonify(
            {
                "deleted": image_id
            }
        )

    # =========================================================
    # VIDEO
    # =========================================================

    @app.get(
        "/video_feed/<int:camera_id>"
    )
    def video_feed(camera_id: int):

        return Response(
            CameraStream(
                db,
                camera_id,
            ).frames(),
            mimetype=(
                "multipart/x-mixed-replace; "
                "boundary=frame"
            ),
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
