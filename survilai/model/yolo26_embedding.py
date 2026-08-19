from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLO26EmbeddingModel(nn.Module):
    """YOLOv8m backbone + trainable 512-D face embedding projection."""

    def __init__(
        self,
        model_name: str = "yolov8m",
        embedding_dim: int = 512,
        device: str = "cpu",
        checkpoint: str | Path | None = "models/yolo26-face-v1.pt",
        load_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device

        from ultralytics import YOLO

        yolo_model = YOLO(f"{model_name}.pt", task="detect")
        yolo_model.to(device)
        yolo_model.model.eval()
        yolo_model.model.requires_grad_(False)

        self._backbone = nn.Sequential(*list(yolo_model.model.model[:10]))
        self._backbone.eval()
        self._backbone.requires_grad_(False)

        self.feature_dim = self._backbone[-1].cv2.conv.out_channels

        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        self._available = True
        self.to(device)

        # Production/live path automatically loads the trained projection.
        # Training explicitly passes load_checkpoint=False so it starts fresh.
        if load_checkpoint and checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.exists():
                self._load_checkpoint_file(checkpoint_path, strict=True)

        self.eval()

    def _load_checkpoint_file(self, path: Path, strict: bool = True) -> None:
        checkpoint = torch.load(
            str(path),
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError(f"Invalid YOLO26 embedding checkpoint: {path}")

        architecture = checkpoint.get("architecture")
        if architecture not in {None, "YOLO26Embedding-v1"}:
            raise ValueError(
                f"Unsupported checkpoint architecture {architecture!r}: {path}"
            )

        saved_dim = int(checkpoint.get("embedding_dim", self.embedding_dim))
        if saved_dim != self.embedding_dim:
            raise ValueError(
                f"Checkpoint embedding_dim={saved_dim}, expected {self.embedding_dim}"
            )

        self.load_state_dict(checkpoint["model_state"], strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._available:
            raise RuntimeError("YOLO26 embedding model unavailable")

        # Backbone is intentionally frozen; only projection is trainable.
        with torch.no_grad():
            features = self._backbone(x)

        if features.ndim != 4:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(features.shape)}")

        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.flatten(1)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        if face_image is None or face_image.ndim != 3:
            raise ValueError("Expected BGR face image with shape (H, W, 3)")

        import cv2

        resized = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.inference_mode():
            embedding = self.forward(tensor)

        return embedding.cpu().numpy()[0]

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str = "cpu",
        model_name: str = "yolov8m",
        embedding_dim: int = 512,
        strict: bool = True,
    ) -> "YOLO26EmbeddingModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO26 checkpoint not found: {path}")

        model = cls(
            model_name=model_name,
            embedding_dim=embedding_dim,
            device=map_location,
            checkpoint=None,
            load_checkpoint=False,
        )
        model._load_checkpoint_file(path, strict=strict)
        model.eval()
        return model

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "architecture": "YOLO26Embedding-v1",
                "backbone": self.model_name,
            },
            str(path),
        )
        print(f"[YOLO26Embedding] Checkpoint saved to {path}")
