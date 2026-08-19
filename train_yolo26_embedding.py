from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from survilai.model.yolo26_embedding import YOLO26EmbeddingModel
from survilai.model.dataset import FaceFolderDataset


class YOLO26FaceTrainer(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512):
        super().__init__()

        self.embedding_model = YOLO26EmbeddingModel(
            model_name="yolov8m",
            embedding_dim=embedding_dim,
            device="cpu",
        )

        # During initial training keep the pretrained YOLO backbone frozen.
        self.embedding_model._backbone.requires_grad_(False)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedding = self.embedding_model(x)
        logits = self.classifier(embedding)
        return embedding, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/faces")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out",
        default="models/yolo26-face-v1.pt",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Dataset: {args.data}")

    dataset = FaceFolderDataset(args.data)

    print(f"Images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    if len(dataset.classes) < 2:
        raise ValueError("Need at least 2 identities.")

    if len(dataset) < 6:
        raise ValueError("Need at least 6 images.")

    val_size = max(2, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = YOLO26FaceTrainer(
        num_classes=len(dataset.classes),
        embedding_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")

    Path(args.out).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, args.epochs + 1):

        model.train()

        # Keep YOLO backbone frozen.
        model.embedding_model._backbone.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            _, logits = model(images)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            train_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            train_total += labels.numel()

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                _, logits = model(images)

                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)

                val_correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()

                val_total += labels.numel()

        train_loss /= max(1, len(train_set))
        val_loss /= max(1, len(val_set))

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            checkpoint = {
                "model_state": model.embedding_model.state_dict(),
                "embedding_dim": 512,
                "num_classes": len(dataset.classes),
                "classes": dataset.classes,
                "image_size": 224,
                "architecture": "YOLO26Embedding-v1",
                "backbone": "yolov8m",
            }

            torch.save(
                checkpoint,
                args.out,
            )

            print(f"✓ saved={args.out}")


if __name__ == "__main__":
    main()
