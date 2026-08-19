
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLO26EmbeddingModel(nn.Module):
    """
    YOLO26 के backbone से face embeddings निकालने के लिए wrapper।
    
    यह model:
    - YOLO26 model load करता है
    - Backbone layers को extract करता है
    - Face crops से features निकालता है
    - L2-normalized embeddings return करता है
    """

    def __init__(
        self,
        model_name: str = "yolov8m",
        embedding_dim: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device

        try:
            from ultralytics import YOLO

            # Load pretrained YOLO weights for inference only.
            yolo_model = YOLO(f"{model_name}.pt", task="detect")
            yolo_model.to(device)
            yolo_model.model.eval()
            yolo_model.model.requires_grad_(False)

            # The first ten YOLOv8 layers are the sequential backbone.
            self._backbone = nn.Sequential(*list(yolo_model.model.model[:10]))
            self._backbone.eval()

            # YOLOv8m emits 576 channels at the end of this backbone.
            self.feature_dim = self._backbone[-1].cv2.conv.out_channels

            self._available = True

        except ImportError:
            print("[YOLO26Embedding] ultralytics not installed")
            self._available = False
            self._backbone = None

        # Projection head maps backbone features into the matching space.
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(768, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) face image tensor
        return: (B, embedding_dim) L2-normalized embeddings
        """
        if not self._available:
            raise RuntimeError("YOLO26 not available")

        # YOLO backbone से features निकालो
        with torch.no_grad():
            features = self._backbone(x)

        # Features को flatten करो
        if len(features.shape) == 4:  # (B, C, H, W)
            features = F.adaptive_avg_pool2d(features, (1, 1))  # (B, C, 1, 1)
            features = features.view(features.size(0), -1)  # (B, C)

        # Projection head के through भेजो
        embeddings = self.projection(features)

        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Single face image (numpy) से embedding निकालो।

        Args:
            face_image: (H, W, 3) BGR image, uint8

        Returns:
            (embedding_dim,) normalized embedding vector
        """
        if not self._available:
            raise RuntimeError("YOLO26 not available")

        # Preprocessing
        if face_image.ndim == 3:
            h, w, c = face_image.shape
        else:
            raise ValueError(f"Expected 3D image, got {face_image.ndim}D")

        # Resize to 224x224 (YOLO expects this)
        import cv2

        resized = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)

        # BGR to RGB, normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = rgb / 255.0  # [0, 1]

        # z-score normalization
        rgb = (rgb - rgb.mean()) / (rgb.std() + 1e-6)

        # To tensor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
        tensor = tensor.to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.forward(tensor)

        return embedding.cpu().numpy()[0]  # (embedding_dim,)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str = "cpu",
        model_name: str = "yolov8m",
        embedding_dim: int = 512,
    ) -> "YOLO26EmbeddingModel":
        """
        Checkpoint से model load करो।
        Note: YOLO26EmbeddingModel checkpoint format में state_dict हो सकता है।
        """
        model = cls(
            model_name=model_name,
            embedding_dim=embedding_dim,
            device=map_location,
        )

        try:
            checkpoint = torch.load(
                str(path),
                map_location=map_location,
                weights_only=False,
            )
            state_dict = (
                checkpoint.get("model_state")
                if isinstance(checkpoint, dict)
                else None
            )
            if state_dict is not None:
                model.load_state_dict(state_dict)
            else:
                print(
                    "[YOLO26Embedding] Ignoring non-YOLO26 checkpoint; "
                    "using pretrained YOLO weights"
                )
        except Exception as exc:
            print(
                "[YOLO26Embedding] Ignoring incompatible checkpoint: "
                f"{exc}"
            )

        model.eval()
        return model

    def save_checkpoint(self, path: str | Path) -> None:
        """Model को checkpoint में save करो।"""
        checkpoint = {
            "model_state": self.state_dict(),
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": str(self.device),
        }
        torch.save(checkpoint, str(path))
        print(f"[YOLO26Embedding] Checkpoint saved to {path}")
