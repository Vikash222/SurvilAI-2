from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from survildb.database import SurvilDB
from .camera_stream import CameraStream


def create_app(db_path: str | Path = "data/survilai.db") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    db = SurvilDB(db_path)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "database": "local", "video": "mjpeg"})

    @app.get("/api/events")
    def events():
        return jsonify([dict(row) for row in db.recent_events(100)])

    @app.get("/api/cameras")
    def cameras():
        with db.connect() as conn:
            rows = conn.execute("SELECT id,name,source,enabled,created_at FROM cameras ORDER BY id").fetchall()
        return jsonify([dict(row) for row in rows])

    @app.post("/api/cameras")
    def add_camera():
        payload = request.get_json(silent=True) or {}
        name, source = str(payload.get("name", "")).strip(), str(payload.get("source", "")).strip()
        if not name or not source:
            return jsonify({"error": "name and source are required"}), 400
        return jsonify({"id": db.add_camera(name, source)}), 201

    @app.delete("/api/cameras/<int:camera_id>")
    def delete_camera(camera_id: int):
        with db.connect() as conn:
            cur = conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        if cur.rowcount == 0:
            return jsonify({"error": "camera not found"}), 404
        return jsonify({"deleted": camera_id})

    @app.get("/api/people")
    def people():
        with db.connect() as conn:
            rows = conn.execute("SELECT id,name,active,created_at FROM people ORDER BY name").fetchall()
        return jsonify([dict(row) for row in rows])

    @app.post("/api/people")
    def add_person():
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name", "")).strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        try:
            person_id = db.add_person(name)
        except Exception as exc:
            return jsonify({"error": "person could not be created", "detail": str(exc)}), 409
        return jsonify({"id": person_id}), 201

    @app.get("/video_feed/<int:camera_id>")
    def video_feed(camera_id: int):
        return Response(
            CameraStream(db, camera_id).frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
