# DataSci 3000 Project: Sign-Language Digit Classifier

A PyTorch CNN that classifies hand-gesture digits (American Sign Language digits, Kaggle). Hands are cropped, converted to 128×128 grayscale, and classified. Includes training, evaluation, data collection from a camera, and live webcam inference.

## Setup

```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install matplotlib pytest     # used by test.py and the test suite
```

## Data

Data folders are gitignored and must be built locally.

1. Download the Kaggle *American Sign Language Digits Dataset* into `archive/`.
2. Build `data/` with one folder per class, images inside:

```
data/
  0/  img1.jpg  img2.jpg ...
  1/
  ...
  9/
```

Class order comes from the sorted folder names. `Test_Data/` uses the same layout for evaluation.

## Usage

| Task | Command |
|---|---|
| Train (saves a checkpoint) | `python train.py` |
| Evaluate on `Test_Data/` | `python test.py` |
| Live inference, Jetson CSI camera | `python Camera.py` |
| Live inference, regular webcam | `python Camera_windows.py` |
| Collect labelled images (press `0`-`9` to save, `q` to quit) | `python collect.py` / `collect_windows.py` |
| Run unit tests | `pytest tests` |

All configuration lives in `settings.py`: paths, model architecture (`ARCH`, default `ASLNet`), checkpoint path, training hyperparameters, preprocessing, and camera settings. Edit it instead of the scripts. Press `Esc` to quit the camera windows.

## How it works

- **`visionPreprocess.py`**: `crop_hand(img, method)` returns the hand crop as a model-ready tensor plus its bounding box, or `None` if no hand is found. `method="yolo"` uses a YOLOv8 detector (used for training); `method="skin"` uses an HSV skin mask (no extra weights).
- **`dataset.py`**: `PreprocessedImageDataset` loads class folders and applies `crop_hand`.
- **`settings.py`**: central constants (no heavy imports).
- **`models/`**: `ASLNet` (default; BatchNorm and dropout), `SimpleCNN` (smaller), plus `SVM` and `random_forest` experiments.
- **`training.py`**: generic `train` / `evaluate` loops for any `nn.Module`.
- **`trained_model.py`**: `save` / `load` checkpoints with their architecture, class names and preprocessing method, so scripts don't hardcode them. Old bare `state_dict` files still load if you pass `arch` and `num_classes`.

## Notes

- `yolov8n.pt` is downloaded automatically on first use.
- The camera scripts use a Jetson GStreamer pipeline; use the `_windows` variants on other machines.
- `train.py`, `test.py` and the camera scripts all use `settings.MODEL_PATH` (`asl_net_model.pth`), so run `python train.py` first.
