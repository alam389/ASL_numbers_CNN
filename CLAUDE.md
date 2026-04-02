# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Sign-language hand-gesture classifier (course project, DataSci 3000). Images are cropped to the hand, converted to 128×128 grayscale, and classified by a PyTorch CNN. There is no build system or linter; everything runs as plain scripts from the repo root.

## Setup and commands

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # test.py also needs matplotlib; tests need pytest
python train.py                   # train + save checkpoint + print (val loss, accuracy)
python test.py                    # classification report / confusion matrix on Test_Data/
python Camera.py                  # live inference, Jetson CSI camera via GStreamer
python Camera_windows.py          # live inference, regular webcam
python collect.py                 # capture labelled images (collect_windows.py for webcam)
pytest tests                      # all tests
pytest tests/test_hand_crop.py::test_name   # a single test
```

All configuration is in `settings.py` (paths, `ARCH`, `MODEL_PATH`, hyperparameters, preprocessing, camera). Scripts have no CLI args.

## Data (gitignored, must be built locally)

- `archive/` is the Kaggle American Sign Language Digits dataset. `data/` is built from it by hand: one subfolder per class (`0/`, `1/`, ...). Labels are the index in the sorted folder list, so folder names define class order (`settings.class_names()`).
- `Test_Data/` uses the same layout (note `.gitignore` says `Test_data/`).
- `yolov8n.pt` is downloaded by ultralytics on first use (`*.pt` is gitignored).

## Architecture

- `settings.py`: single source of config. Keep it free of heavy imports.
- `visionPreprocess.py`: `crop_hand(img, method="yolo"|"skin") -> HandCrop | None`. The YOLO adapter (used for training) takes the highest-confidence box; the skin adapter (HSV mask + CLAHE) needs no weights. Both return a `HandCrop` (tensor, bbox in original coordinates, display image) or `None` when nothing is found. YOLO loads lazily via `get_yolo()`; tests pass a stub through `yolo=`.
- `dataset.py`: `PreprocessedImageDataset` applies `crop_hand` per item (slow) and falls through to the next sample on `None`.
- `trained_model.py`: `save`/`load` checkpoints as a dict with arch, class names and preprocess method; `load` returns a `TrainedModel` with `.predict`. Bare legacy `state_dict` files load only if `arch` and `num_classes` are passed (`settings.LEGACY_*`). New architectures must be added to `ARCHITECTURES`.
- `training.py`: generic `train` / `evaluate` for any `nn.Module` mapping `(B,1,128,128)` to logits.
- `models/`: `ASLNet` (default), `SimpleCNN`, `SVMClassifier` (hinge loss `nn.Module`), `random_forest` (sklearn wrapper, not wired up; its `fit` is buggy).
- `train.py` splits 90/5/5 randomly (the test split is unused), trains `settings.ARCH`, and saves to `settings.MODEL_PATH`.

## Gotchas

- Training uses YOLO crops (`TRAIN_PREPROCESS_METHOD`) while `Camera.py` uses the skin adapter and `Camera_windows.py` uses YOLO. `preprocess_method` is saved in the checkpoint but nothing reads it yet. The stock COCO `yolov8n.pt` has no hand class, so crops may not be hands.
- `train.py` derives `NUM_CLASSES` from the folders in `data/`, falling back to `LEGACY_NUM_CLASSES` if the folder is missing.
- Committed `.pth` files are legacy bare state dicts; the `.pptx` and `.docx` are project deliverables, not code.
