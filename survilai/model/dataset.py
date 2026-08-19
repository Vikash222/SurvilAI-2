from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceFolderDataset(Dataset):
    """Dataset layout: root/person_name/image.jpg."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, image_size: int = 224) -> None:
        self.root = Path(root)
        self.classes = sorted(
            p.name for p in self.root.iterdir() if p.is_dir()
        )
        self.class_to_idx = {
            name: i for i, name in enumerate(self.classes)
        }

        self.samples = []

        for name in self.classes:
            for path in sorted((self.root / name).rglob("*")):
                if path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append(
                        (path, self.class_to_idx[name])
                    )

        if not self.samples:
            raise ValueError(
                f"No face images found under {self.root}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]

        # Always force 3-channel RGB.
        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long,
        )
