# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Sign-language hand-gesture classifier (course project, DataSci 3000). Images are cropped to the hand, converted to 128×128 grayscale, and classified by a small PyTorch CNN. There is no build system, linter, or test suite; everything is run as plain scripts from the repo root.

## Setup and commands

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # test.py also needs matplotlib (not listed)
python train.py                   # train + save weights + print (val loss, accuracy)
python test.py                    # classification report / confusion matrix on Test_Data/
python Camera.py                  # live webcam inference (Jetson CSI camera via GStreamer)
python Camera_windows.py          # same, for a regular webcam on Windows
python collect.py                 # capture preprocessed images from camera (collect_windows.py for Windows)
```

`train.py` and `test.py` are configured by editing constants at the top of the file (`DATA_DIR`, `NUM_CLASSES`, `BATCH_SIZE`, `MODEL_PATH`, `EVAL_DIR`), not via CLI args.

## Data (gitignored, must be built locally)

- `archive/` is the Kaggle American Sign Language Digits dataset. `data/` must be built from it by hand: one subfolder per class (`0/`, `1/`, ...), images inside. Labels come from `sorted(os.listdir(root))` index, so folder names define class order.
- `Test_Data/` (note: `.gitignore` says `Test_data/`) is the same layout, used by `test.py`.
- `yolov8n.pt` is loaded by `dataset.py` at import time (`*.pt` is gitignored; ultralytics downloads it on first use).

## Architecture

- `visionPreprocess.py` is the shared preprocessing layer. Two paths exist:
  - `preprocess_with_yolo(img, yolo)`: crops the highest-confidence YOLO box → grayscale → 128×128 → `cnn_transform` (normalize to [-1, 1]). Used for training/eval via `dataset.py`. Returns `None` if nothing is detected.
  - `preprocess_live(img)`: HSV skin-mask + largest-contour square crop, no YOLO. Used by the camera/collection scripts.
  - Training and live inference therefore use different crops; keep them in mind when debugging train/serve mismatch. `preprocess_live`'s return arity differs between callers (`dataset.py`'s commented code unpacks 3 values, `Camera.py` unpacks 2); check the function before relying on either.
- `dataset.py`: `PreprocessedImageDataset` runs YOLO on every `__getitem__` (slow). Images with no detection fall through to the next index.
- `training.py`: generic `train(model, loader, epochs, lr)` (Adam + CrossEntropy) and `evaluate(model, loader)` → `(loss, accuracy)`. It works with any `nn.Module` that maps `(B,1,128,128)` to logits.
- `models/`: `model_CNN.SimpleCNN` (the active model), `complex_CNN.ASLNet`, `SVM.SVMClassifier` (linear + hinge loss as an `nn.Module`), `random_forest` (sklearn wrapper, not wired into `train.py`). New models go here and get swapped in inside `train.py`.
- `train.py` splits 90/5/5 (train/val/test) randomly, trains `SimpleCNN`, and saves a `.pth` state dict to the repo root. Alternate models are left as commented-out blocks.

## Gotchas

- `NUM_CLASSES` is hardcoded separately in `train.py` (2), `test.py` (2), and `Camera.py` (10). They must match the checkpoint being loaded.
- Checkpoint filenames are hardcoded and differ per script (`train.py` writes `simple_cnn_model_3.pth`; `test.py` loads `simple_cnn_model.pth`; `Camera.py` loads `cnn_model.pth` into `SimpleCNN`).
- `Camera.py`/`collect.py` use a Jetson `nvarguscamerasrc` GStreamer pipeline; use the `_windows` variants elsewhere.
- Committed `.pth` weights, the `.pptx` and the `.docx` are project deliverables, not code.
