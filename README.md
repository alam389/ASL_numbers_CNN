# DataSci 3000 Project: Sign-Language Hand-Gesture Classifier

A PyTorch CNN that classifies American Sign Language hand gestures: digits 0-9 and letters A-Z (36 classes). Hands are cropped, converted to 128×128 grayscale, and classified. Includes training, evaluation, data collection from a camera, and live webcam inference.

## Setup

```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install matplotlib pytest     # used by test.py and the test suite
```

## Data

Data folders are gitignored and built locally from the **ASL-HG** dataset (CC BY 4.0, photographed on phones with real backgrounds and varied skin tones).

1. Download it from https://data.mendeley.com/datasets/j4y5w2c8w9/1 and unzip until you find `ASL_Raw_Images.zip` (use the raw images, not the processed ones).
2. Import all 36 classes (add `--digits-only` for just 0-9):

```
python prepare_hg.py path/to/ASL_Raw_Images.zip --replace
```

This writes `data/0/` ... `data/Z/` for training and holds out 15% in `Test_Data/` (same layout) for evaluation. Class order comes from the sorted folder names.

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

## Results

ASLNet trained on ASL-HG (36 classes), MediaPipe hand crops, 128×128 grayscale, 20 epochs, batch size 32, Adam (lr 1e-3), Apple GPU (MPS).

| Split | Images | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| Validation (5% of `data/`) | 1,529 | 0.9987 | – | – | – |
| Test (`Test_Data/`, held out) | 5,394 | 0.9998 | 0.9998 | 0.9998 | 0.9998 |

The only test error was one H predicted as I. MediaPipe found no hand in 12 training and 6 test images, which were dropped.

These scores are optimistic. The train/test split is random, so photos of the same volunteers and backgrounds appear on both sides. Expect lower accuracy on a new person or webcam, especially for similar handshapes (M/N/S/T/E, V/2, O/0) and for J and Z, which are motion signs captured here as single frames.

## How it works

- **`visionPreprocess.py`**: `crop_hand(img, method)` returns the hand crop as a model-ready tensor plus its bounding box, or `None` if no hand is found. `method="mediapipe"` (default) crops around MediaPipe hand landmarks (the model file downloads on first use); `"yolo"` and `"skin"` (HSV mask) are alternatives.
- **`dataset.py`**: `PreprocessedImageDataset` loads class folders, applies `crop_hand` once and caches the crops in `cache/`.
- **`augment.py`**: training-time shift/rotation/scale/brightness changes, plus an optional random-background swap for black-background data (`BACKGROUND_SWAP_PROB`).
- **`device.py`**: picks CUDA, then Apple GPU (MPS), then CPU.
- **`settings.py`**: central constants (no heavy imports).
- **`models/`**: `ASLNet` (default; BatchNorm and dropout), `SimpleCNN` (smaller), plus `SVM` and `random_forest` experiments.
- **`training.py`**: generic `train` / `evaluate` loops for any `nn.Module`.
- **`trained_model.py`**: `save` / `load` checkpoints with their architecture, class names and preprocessing method, so scripts don't hardcode them. Old bare `state_dict` files still load if you pass `arch` and `num_classes`.

## Notes

- `hand_landmarker.task` (MediaPipe) is downloaded automatically on first use; `yolov8n.pt` only if you switch to the YOLO method.
- The live preview is mirrored (`FLIP_DISPLAY`); the model input is not.
- The camera scripts use a Jetson GStreamer pipeline; use the `_windows` variants on other machines.
- `train.py`, `test.py` and the camera scripts all use `settings.MODEL_PATH` (`asl_net_model.pth`), so run `python train.py` first.
