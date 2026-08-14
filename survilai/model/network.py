from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SurvilFaceNet(nn.Module):
    """Small face embedding network owned by SurvilAI.

    The network produces normalized embeddings and, during training, a classifier
    head. It is intentionally trained from SurvilAI's own dataset rather than
    calling a hosted face-recognition API or importing a pretrained identity model.
    """

    def __init__(self, embedding_dim: int = 128, num_classes: int | None = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            ConvBlock(1, 32, 2),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 192, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

    def forward(self, x: torch.Tensor, classify: bool = True):
        x = self.features(x)
        embedding = F.normalize(self.embedding(x), p=2, dim=1)
        if classify and self.classifier is not None:
            return embedding, self.classifier(embedding)
        return embedding

    @classmethod
    def from_checkpoint(cls, path: str, map_location: str = "cpu") -> "SurvilFaceNet":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            embedding_dim=int(checkpoint["embedding_dim"]),
            num_classes=int(checkpoint["num_classes"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
