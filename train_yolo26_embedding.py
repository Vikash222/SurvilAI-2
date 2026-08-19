from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from survilai.model.yolo26_embedding import YOLO26EmbeddingModel
from survilai.model.dataset import FaceFolderDataset


class YOLO26FaceTrainer(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512):
        super().__init__()
        self.embedding_model = YOLO26EmbeddingModel(
            model_name="yolov8m",
            embedding_dim=embedding_dim,
            device="cpu",
            checkpoint=None,
            load_checkpoint=False,
        )
        self.embedding_model._backbone.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedding = self.embedding_model(x)
        logits = self.classifier(embedding)
        return embedding, logits


def metric_loss(embeddings: torch.Tensor, labels: torch.Tensor, negative_margin: float = 0.20) -> torch.Tensor:
    """Pull same-identity embeddings together and push different identities apart."""
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.T
    total = torch.tensor(0.0, device=z.device)
    pairs = 0

    for i in range(z.size(0)):
        positive = labels == labels[i]
        positive[i] = False
        negative = labels != labels[i]

        if positive.any():
            total = total + (1.0 - sim[i][positive]).mean()
            pairs += 1
        if negative.any():
            total = total + F.relu(sim[i][negative] - negative_margin).mean()
            pairs += 1

    return total / max(1, pairs)


def stratified_split(dataset: FaceFolderDataset, val_fraction: float = 0.20):
    generator = torch.Generator().manual_seed(42)
    train_indices = []
    val_indices = []

    for class_id in range(len(dataset.classes)):
        indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_id]
        perm = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in perm]
        val_count = max(2, int(round(len(indices) * val_fraction)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    return train_indices, val_indices


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 face embeddings")
    parser.add_argument("--data", default="data/faces")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-images-per-class", type=int, default=10)
    parser.add_argument("--out", default="models/yolo26-face-v1.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Dataset: {args.data}")

    base_dataset = FaceFolderDataset(args.data, image_size=224, augment=False)
    print(f"Images: {len(base_dataset)}")
    print(f"Classes: {base_dataset.classes}")

    if len(base_dataset.classes) < 2:
        raise ValueError("Need at least 2 identities.")

    counts = {name: 0 for name in base_dataset.classes}
    for _, label in base_dataset.samples:
        counts[base_dataset.classes[label]] += 1

    print("Images per identity:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
        if count < args.min_images_per_class:
            raise ValueError(
                f"Identity {name!r} has only {count} images. "
                f"Add at least {args.min_images_per_class} real images per identity "
                "with different angles, lighting and distances before training."
            )

    train_indices, val_indices = stratified_split(base_dataset)

    train_dataset = FaceFolderDataset(args.data, image_size=224, augment=True)
    val_dataset = FaceFolderDataset(args.data, image_size=224, augment=False)
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    print(f"Train images: {len(train_set)}")
    print(f"Validation images: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = YOLO26FaceTrainer(
        num_classes=len(base_dataset.classes),
        embedding_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.embedding_model._backbone.eval()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            embeddings, logits = model(images)
            ce = criterion(logits, labels)
            metric = metric_loss(embeddings, labels)
            loss = ce + 0.5 * metric

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.numel()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                embeddings, logits = model(images)
                loss = criterion(logits, labels) + 0.5 * metric_loss(embeddings, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.numel()

        scheduler.step()

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
                "num_classes": len(base_dataset.classes),
                "classes": base_dataset.classes,
                "image_size": 224,
                "architecture": "YOLO26Embedding-v1",
                "backbone": "yolov8m",
                "training_images": len(base_dataset),
                "min_images_per_class": args.min_images_per_class,
            }
            torch.save(checkpoint, args.out)
            print(f"✓ saved={args.out}")

    print(f"\nBest validation loss: {best_val:.4f}")
    print(f"Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
