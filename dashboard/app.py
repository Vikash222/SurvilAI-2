from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template

from survildb.database import SurvilDB


def create_app(db_path: str | Path = "data/survilai.db") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    db = SurvilDB(db_path)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "database": "local"})

    @app.get("/api/events")
    def events():
        rows = db.recent_events(100)
        return jsonify([dict(row) for row in rows])

    @app.get("/api/cameras")
    def cameras():
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT id, name, source, enabled, created_at FROM cameras ORDER BY id"
            ).fetchall()
        return jsonify([dict(row) for row in rows])

    @app.get("/api/people")
    def people():
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT id, name, active, created_at FROM people ORDER BY name"
            ).fetchall()
        return jsonify([dict(row) for row in rows])

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
