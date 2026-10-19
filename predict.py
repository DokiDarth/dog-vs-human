"""Ask the trained model whether a photo shows a dog or a human.

Usage:
    python predict.py photo1.jpg [photo2.jpg ...]
"""
import sys

import torch
from PIL import Image

from train import MODEL_PATH, build_model, eval_tf, pick_device


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = build_model(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["classes"]


def predict(model, classes, path, device):
    x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    best = probs.argmax().item()
    return classes[best], probs[best].item()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    device = pick_device()
    model, classes = load_model(device)
    for path in sys.argv[1:]:
        label, confidence = predict(model, classes, path, device)
        print(f"{path}: {label} ({confidence:.1%} sure)")
