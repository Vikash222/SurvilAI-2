from __future__ import annotations

from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceFolderDataset(Dataset):
    """Face dataset at root/person_name/image.* with consistent face cropping."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, image_size: int = 224, augment: bool = True) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = []

        for name in self.classes:
            for path in sorted((self.root / name).rglob("*")):
                if path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[name]))

        if not self.samples:
            raise ValueError(f"No face images found under {self.root}")

        self.face_detector = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.15,
                    )
                ], p=0.8),
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=8,
                        translate=(0.05, 0.05),
                        scale=(0.92, 1.08),
                    )
                ], p=0.6),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
                ], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_largest_face(self, image: Image.Image) -> Image.Image:
        rgb = cv2.cvtColor(__import__("numpy").array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            return image

        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.30)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(rgb.shape[1], x + w + pad_x)
        y2 = min(rgb.shape[0], y + h + pad_y)
        crop = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = self._crop_largest_face(image)
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
