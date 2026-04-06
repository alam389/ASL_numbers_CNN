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

- Dataset is **ASL-HG** (Mendeley, CC BY 4.0): download `ASL_Raw_Images.zip` and run `python prepare_hg.py <zip> --replace`. It writes `data/0..9,A..Z/` (36 classes; `--digits-only` for 0-9) and holds out 15% in `Test_Data/` (same layout). The script skips the "processed" images. Labels are the index in the sorted folder list (`settings.class_names()`).
- `cache/<folder>_<method>.pt` holds cropped tensors, rebuilt automatically when files, crop method or input size change. Delete it if you change crop code.
- `hand_landmarker.task` (MediaPipe) downloads on first use; `*.pt` and `*.task` are gitignored.

## Architecture

- `settings.py`: single source of config. Keep it free of heavy imports.
- `visionPreprocess.py`: `crop_hand(img, method="mediapipe"|"yolo"|"skin") -> HandCrop | None`. MediaPipe (default) squares a padded box around the hand landmarks; YOLO takes the best box of the stock COCO model (no hand class, so it boxes "person"); skin uses an HSV mask. Each returns a `HandCrop` (tensor, bbox in original coordinates, display image) or `None`. Detectors load lazily; tests inject stubs via `yolo=` / `landmarker=`.
- `dataset.py`: `PreprocessedImageDataset` crops every image once (dropping images with no hand) and serves cached tensors.
- `augment.py`: `HandAugment` for training only (affine + brightness; optional background swap, which only makes sense for black-background data, see `settings.BACKGROUND_SWAP_PROB`).
- `device.py`: `pick_device()` honours `settings.DEVICE` (auto: cuda > mps > cpu).
- `trained_model.py`: `save`/`load` checkpoints as a dict with arch, class names and preprocess method; `load` returns a `TrainedModel` (`.predict`, `.logits`). Bare legacy `state_dict` files load only with `legacy_arch` / `legacy_num_classes`. New architectures must be added to `ARCHITECTURES`.
- `training.py`: generic `train` / `evaluate` for any `nn.Module` mapping `(B,1,128,128)` to logits.
- `models/`: `ASLNet` (default), `SimpleCNN`, `SVMClassifier`, `random_forest` (sklearn wrapper, not wired up; its `fit` is buggy).
- `train.py` splits train/val/test 90/5/5 randomly (the test split is unused; `Test_Data/` is the real held-out set), trains `settings.ARCH`, and saves to `settings.MODEL_PATH`.

## Gotchas

- Camera scripts read the crop method from the checkpoint (`preprocess_method`), so retrain after changing `TRAIN_PREPROCESS_METHOD`. `Camera_windows.py` mirrors only the preview (`FLIP_DISPLAY`); the model sees the unflipped frame.
- `NUM_CLASSES` comes from the folders in `data/` (fallback `LEGACY_NUM_CLASSES`). Class "0" in ASL-HG is the two-handed sign.
- Models trained on clean datasets fail on webcam backgrounds; validation accuracy on the train distribution says little about live performance. Evaluate with `test.py` on `Test_Data/` and on your own captures.
- Committed `.pth` files other than `asl_net_model.pth` are legacy bare state dicts; the `.pptx` and `.docx` are project deliverables, not code.
