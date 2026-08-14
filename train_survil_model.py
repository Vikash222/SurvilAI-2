from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from survilai.model import SurvilFaceNet
from survilai.model.dataset import FaceFolderDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SurvilAI's own face embedding model")
    parser.add_argument("--data", default="data/faces", help="root/person/images layout")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/survil-face-v1.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    dataset = FaceFolderDataset(args.data)
    if len(dataset.classes) < 2:
        raise ValueError("Training requires at least 2 identities")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurvilFaceNet(embedding_dim=128, num_classes=len(dataset.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, logits = model(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.numel()

        train_loss /= len(train_set)
        val_loss /= len(val_set)
        accuracy = correct / max(1, total)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={accuracy:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "embedding_dim": model.embedding_dim,
                "num_classes": len(dataset.classes),
                "classes": dataset.classes,
                "image_size": 112,
                "architecture": "SurvilFaceNet-v1",
            }, output)
            print(f"saved={output}")


if __name__ == "__main__":
    main()
