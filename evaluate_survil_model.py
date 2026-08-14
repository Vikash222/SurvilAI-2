from __future__ import annotations

import argparse

import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

from survilai.model import SurvilFaceNet
from survilai.model.dataset import FaceFolderDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SurvilAI recognition checkpoint")
    parser.add_argument("--data", default="data/faces")
    parser.add_argument("--checkpoint", default="models/survil-face-v1.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurvilFaceNet.from_checkpoint(args.checkpoint, str(device)).to(device)
    dataset = FaceFolderDataset(args.data)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    embeddings, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            embeddings.append(model(images.to(device), classify=False).cpu())
            labels.append(batch_labels)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    # A simple nearest-neighbour self-evaluation is only a sanity check; it is
    # not a substitute for a held-out identity-disjoint benchmark.
    correct = 0
    for i in range(len(embeddings)):
        scores = cosine_similarity(embeddings[i:i+1], embeddings, dim=1)
        scores[i] = -1
        prediction = labels[scores.argmax()]
        correct += int(prediction == labels[i])
    print(f"nearest_neighbor_accuracy={correct / max(1, len(labels)):.3f}")
    print("For release decisions, evaluate on identities/sessions held out from training.")


if __name__ == "__main__":
    main()
