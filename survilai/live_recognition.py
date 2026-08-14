from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from survildb.database import SurvilDB
from .model import SurvilFaceNet


@dataclass(frozen=True)
class Match:
    person_id: int | None
    name: str
    score: float


class LiveRecognizer:
    """Local SurvilFaceNet inference against embeddings stored in SQLite."""

    def __init__(self, db: SurvilDB, checkpoint: str | Path, threshold: float = 0.72):
        self.db = db
        self.threshold = float(threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SurvilFaceNet.from_checkpoint(str(checkpoint), map_location=str(self.device)).to(self.device)
        self.model.eval()
        self.gallery: list[tuple[int, str, np.ndarray]] = []
        self.reload_gallery()

    def reload_gallery(self) -> None:
        self.gallery.clear()
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT p.id, p.name, e.embedding, e.dimension
                   FROM people p JOIN face_embeddings e ON e.person_id=p.id
                   WHERE p.active=1"""
            ).fetchall()
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size == int(row["dimension"]):
                self.gallery.append((int(row["id"]), str(row["name"]), vector))

    @staticmethod
    def preprocess(face: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face
        gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
        return tensor

    @torch.inference_mode()
    def embed(self, face: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(face).to(self.device)
        vector = self.model(tensor, classify=False)[0].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)

    def match(self, face: np.ndarray) -> Match:
        if not self.gallery:
            return Match(None, "Unknown", 0.0)
        query = self.embed(face)
        best: tuple[int, str, float] | None = None
        for person_id, name, vector in self.gallery:
            score = float(np.dot(query, vector))
            if best is None or score > best[2]:
                best = (person_id, name, score)
        assert best is not None
        person_id, name, score = best
        if score < self.threshold:
            return Match(None, "Unknown", score)
        return Match(person_id, name, score)
