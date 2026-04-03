import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import visionPreprocess
from visionPreprocess import HandCrop, crop_hand


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(conf, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class StubYolo:
    """Mimics ultralytics: yolo(img_rgb)[0].boxes."""
    def __init__(self, xyxy=(), conf=()):
        self.boxes = _Boxes(xyxy, conf)
        self.calls = 0

    def __call__(self, img, **kwargs):
        self.calls += 1
        return [self]


def _skin_image():
    img = np.full((240, 320, 3), 20, np.uint8)
    img[60:180, 100:220] = (80, 120, 200)  # BGR skin-like tone
    return img


def test_yolo_crop_picks_best_box_and_normalizes():
    img = np.random.RandomState(0).randint(0, 255, (200, 300, 3), dtype=np.uint8)
    stub = StubYolo([[10, 20, 60, 90], [100, 50, 180, 150]], [0.3, 0.9])
    crop = crop_hand(img, "yolo", yolo=stub)
    assert isinstance(crop, HandCrop)
    assert crop.bbox == (100, 50, 80, 100)
    assert crop.tensor.shape == (1, 128, 128) and crop.tensor.dtype == torch.float32
    assert crop.tensor.min() >= -1.0 and crop.tensor.max() <= 1.0
    assert crop.image.shape == (128, 128) and crop.image.dtype == np.uint8


def test_yolo_matches_legacy_numerics():
    img = np.random.RandomState(1).randint(0, 255, (100, 100, 3), dtype=np.uint8)
    crop = crop_hand(img, "yolo", yolo=StubYolo([[5, 5, 80, 90]], [0.8]))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[5:90, 5:80]
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (128, 128))
    expected = visionPreprocess.cnn_transform(torch.tensor(g, dtype=torch.float32).unsqueeze(0) / 255.0)
    assert torch.equal(crop.tensor, expected)


def test_yolo_no_detection_returns_none():
    img = np.zeros((50, 50, 3), np.uint8)
    assert crop_hand(img, "yolo", yolo=StubYolo()) is None


def test_yolo_empty_crop_returns_none():
    img = np.zeros((50, 50, 3), np.uint8)
    assert crop_hand(img, "yolo", yolo=StubYolo([[30, 30, 30, 40]], [0.9])) is None


def test_skin_crop_finds_hand():
    crop = crop_hand(_skin_image(), "skin")
    assert isinstance(crop, HandCrop)
    x, y, w, h = crop.bbox
    assert 90 <= x <= 105 and 50 <= y <= 65 and w >= 110 and h >= 110
    assert crop.tensor.shape == (1, 128, 128) and crop.tensor.dtype == torch.float32
    assert 0.0 <= crop.tensor.min() and crop.tensor.max() <= 1.0 and crop.tensor.max() > 0
    assert crop.image.shape == (128, 128)
    assert crop.debug is not None and crop.debug.ndim == 2


def test_skin_no_skin_returns_none():
    assert crop_hand(np.full((100, 100, 3), 10, np.uint8), "skin") is None


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        crop_hand(np.zeros((10, 10, 3), np.uint8), "nope")


def test_import_does_not_load_yolo_weights():
    assert visionPreprocess._yolo_model is None


class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class StubLandmarker:
    def __init__(self, pts=None):
        self.pts = pts

    def detect(self, image):
        class R:
            pass
        r = R()
        r.hand_landmarks = [self.pts] if self.pts else []
        return r


def test_mediapipe_crop_uses_landmarks():
    pytest.importorskip("mediapipe")
    img = np.full((200, 300, 3), 90, np.uint8)
    pts = [_Pt(0.4, 0.3), _Pt(0.6, 0.7)]
    crop = crop_hand(img, "mediapipe", landmarker=StubLandmarker(pts))
    assert crop.tensor.shape == (1, 128, 128)
    assert -1.0 <= float(crop.tensor.min()) and float(crop.tensor.max()) <= 1.0
    x, y, w, h = crop.bbox
    assert 0 <= x < 300 and 0 <= y < 200 and w > 0 and h > 0
    assert x <= 0.4 * 300 and x + w >= 0.6 * 300  # box covers the landmarks


def test_mediapipe_no_hand_returns_none():
    pytest.importorskip("mediapipe")
    assert crop_hand(np.zeros((50, 50, 3), np.uint8), "mediapipe", landmarker=StubLandmarker()) is None
