import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from settings import INPUT_SIZE, YOLO_WEIGHTS, SKIN_CROP_PADDING

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),  # now shape is (1, H, W)
    transforms.Normalize(mean=[0.5], std=[0.5])
])


@dataclass
class HandCrop:
    """Result of crop_hand().

    tensor: model-ready float32 torch tensor, shape (1, H, W). Scaling depends on method:
            "yolo" -> normalized to [-1, 1] (cnn_transform); "skin" -> [0, 1].
    bbox:   (x, y, w, h) of the crop in the ORIGINAL image's coordinates.
    image:  uint8 (H, W) view of the model input, for display / saving.
    debug:  optional uint8 grayscale of the crop before resizing (skin path only).
    """
    tensor: torch.Tensor
    bbox: Tuple[int, int, int, int]
    image: np.ndarray
    debug: Optional[np.ndarray] = None


_yolo_model = None


def get_yolo(weights=YOLO_WEIGHTS):
    """Lazily load (and cache) the YOLO detector so importing this module is cheap."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(weights)
    return _yolo_model


def _crop_yolo(img, yolo=None):
    yolo = yolo if yolo is not None else get_yolo()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = yolo(img_rgb, verbose=False)[0]

    if len(results.boxes) == 0:
        return None

    # Highest-confidence box
    boxes = results.boxes
    best_idx = torch.argmax(boxes.conf)
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.resize(crop_gray, INPUT_SIZE)
    crop_tensor = torch.tensor(crop_gray, dtype=torch.float32).unsqueeze(0) / 255.0
    crop_tensor = cnn_transform(crop_tensor)

    image = np.clip((crop_tensor[0].numpy() * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return HandCrop(crop_tensor, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), image)


def _crop_skin(img, size=INPUT_SIZE):
    H, W, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([160, 20, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    skin_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Square crop
    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2

    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(cx + side // 2, W)
    y2 = min(cy + side // 2, H)

    # Padding
    pad = int(SKIN_CROP_PADDING * side)
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, W)
    y2 = min(y2 + pad, H)

    hand_region = img[y1:y2, x1:x2]
    if hand_region.size == 0:
        return None

    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray[skin_mask[y1:y2, x1:x2] == 0] = 0
    digit_u8 = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    digit = digit_u8.astype("float32") / 255.0

    return HandCrop(torch.from_numpy(digit).reshape(1, size[0], size[1]),
                    (x1, y1, x2 - x1, y2 - y1), digit_u8, debug=gray)


def crop_hand(img, method="yolo", yolo=None) -> Optional[HandCrop]:
    """Find the hand in a BGR image and return a model-ready HandCrop, or None if not found.

    method: "yolo" (highest-confidence YOLO box; used for training) or
            "skin" (HSV skin mask + largest contour; no model needed).
    yolo:   optional detector to use instead of the lazily loaded default (yolo only).
    """
    if method == "yolo":
        return _crop_yolo(img, yolo)
    if method == "skin":
        return _crop_skin(img)
    raise ValueError(f"unknown method {method!r}; expected 'yolo' or 'skin'")


# Process folder just used for testing
def process_folder(input_folder, output_folder, method="skin"):
    """Process all images in a folder and save the results. """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            crop = crop_hand(img, method)
            if crop is None:
                print(f"No hand found: {filename}")
                continue

            cv2.imwrite(output_path, crop.image)
            print(f"Processed: {filename}")

"""
Grayscale simplifies the data

Models require fixed input size

Gaussian Blur removes noise and helps the model focus on structure rather than pixel-level randomness

Normalize makes pixel values small which is better for neural networks

Have to convert back to 0-255 to save with OpenCV
"""
