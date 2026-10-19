# Dog or Human?

An image classifier that tells whether a photo shows a **dog** or a **human**, with a small
local website to try it in your browser.

It uses **transfer learning**: a ResNet-18 network pretrained on ImageNet, with only its final
layer retrained on 2,500 dog photos and 2,500 human photos. It reaches **99.9% accuracy** on
held-out photos after about 5 minutes of training on a laptop CPU.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gh release download v1.0 -p dog_vs_human.pt   # the trained model (43 MB)
python app.py
```

No `gh` CLI? Download `dog_vs_human.pt` from the repo's **Releases** page into this folder instead.

Open http://localhost:8000 and drop in a photo.

Or use the command line:

```bash
python predict.py samples/dog_0.jpg samples/human_0.jpg
```

## Retrain from scratch

1. Download the datasets:
   - Dogs: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), `images.tar.gz` (792 MB)
   - Humans: [Human Action Recognition](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition),
     `data/test-00000-of-00001.parquet` (98 MB)
2. Build the `data/dog` and `data/human` folders:
   ```bash
   python prepare_data.py path/to/images.tar.gz path/to/test-00000-of-00001.parquet
   ```
3. Train (saves `dog_vs_human.pt`):
   ```bash
   python train.py --epochs 3
   ```

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Turns the raw downloads into `data/dog/` and `data/human/` |
| `train.py` | Trains the model with transfer learning |
| `predict.py` | Classifies photos from the command line |
| `app.py` | Local website (Python's built-in server, no extra installs) |
| `dog_vs_human.pt` | The trained model (attached to the v1.0 release, not stored in git) |
| `samples/` | Test photos the model never saw during training |

## Limitations

The model only knows two answers, so it has to pick one. A cat gets labelled "dog" (with lower
confidence). To fix that, add a `data/other/` folder with other images and retrain.
