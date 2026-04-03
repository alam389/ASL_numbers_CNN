import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataset
import settings


@pytest.fixture
def image_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "imgs"
    for cls in ("0", "1"):
        (root / cls).mkdir(parents=True)
        for i in range(3):
            img = np.full((64, 64, 3), 40, np.uint8)
            cv2.circle(img, (32, 32), 15, (120, 160, 220), -1)  # skin-ish blob
            cv2.imwrite(str(root / cls / f"{i}.png"), img)
    return root


def test_cache_built_then_reused(image_dir, monkeypatch):
    ds = dataset.PreprocessedImageDataset(str(image_dir), method="skin")
    assert len(ds) == 6
    x, y = ds[0]
    assert x.shape == (1, 128, 128) and y in (0, 1)

    def boom(*a, **k):
        raise AssertionError("crop_hand called on a warm cache")
    monkeypatch.setattr(dataset, "crop_hand", boom)
    ds2 = dataset.PreprocessedImageDataset(str(image_dir), method="skin")
    assert len(ds2) == 6


def test_cache_rebuilt_when_files_change(image_dir):
    dataset.PreprocessedImageDataset(str(image_dir), method="skin")
    extra = np.full((64, 64, 3), 40, np.uint8)
    cv2.circle(extra, (32, 32), 15, (120, 160, 220), -1)
    cv2.imwrite(str(image_dir / "0" / "new.png"), extra)
    assert len(dataset.PreprocessedImageDataset(str(image_dir), method="skin")) == 7
