"""Train a dog-vs-human image classifier with transfer learning.

We take ResNet-18 (already trained on 1.2M ImageNet photos, so it knows edges,
fur, faces, shapes...) and retrain only its last layer to answer one question:
dog or human?

Usage:
    python train.py [--epochs 3]
"""
import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "dog_vs_human.pt"

# ImageNet statistics: the pretrained network expects inputs normalised this way.
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Training images get random flips/crops/colour shifts so the model can't just
# memorise them; validation images are left as-is for an honest score.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes=2, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for p in model.parameters():          # freeze everything it already knows...
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # ...and add a fresh head
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel image loaders (use 0 if multiprocessing fails)")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = pick_device()
    print(f"device: {device}")

    # Folder names become labels: data/dog -> 0, data/human -> 1
    full = datasets.ImageFolder(DATA)
    classes = full.classes
    print(f"classes: {classes}, images: {len(full)}")

    # 80% train / 20% validation
    n_val = len(full) // 5
    train_idx, val_idx = random_split(range(len(full)), [len(full) - n_val, n_val])
    train_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA, train_tf), train_idx.indices)
    val_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA, eval_tf), val_idx.indices)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    model = build_model(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running_loss = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)   # how wrong were we?
            loss.backward()               # which way should the weights move?
            optimizer.step()              # nudge them
            running_loss += loss.item() * y.size(0)
        val_acc = evaluate(model, val_dl, device)
        print(f"epoch {epoch}/{args.epochs}  loss {running_loss / len(train_ds):.4f}  "
              f"val accuracy {val_acc:.2%}  ({time.time() - start:.0f}s)")

    torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
