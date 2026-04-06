import os

import cv2
import torch
from torch.utils.data import Dataset

import settings
from visionPreprocess import crop_hand


def _list_samples(root_dir):
    samples = []
    for label, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith(settings.IMAGE_EXTENSIONS):
                samples.append((os.path.join(class_path, fname), label))
    return samples


class PreprocessedImageDataset(Dataset):
    """Hand crops for every image under root_dir (one subfolder per class).

    Crops are computed once and cached in settings.CACHE_DIR, so later runs skip the
    expensive detector. The cache is rebuilt if the file list, crop method or input size
    changes. Images where no hand is found are dropped when the cache is built.
    """

    def __init__(self, root_dir, method=settings.TRAIN_PREPROCESS_METHOD, use_cache=True):
        samples = _list_samples(root_dir)
        signature = {
            "method": method,
            "input_size": tuple(settings.INPUT_SIZE),
            "files": [(p, label, os.path.getsize(p)) for p, label in samples],
        }
        cache_path = os.path.join(
            settings.CACHE_DIR, f"{os.path.basename(os.path.normpath(root_dir))}_{method}.pt")

        cached = self._read_cache(cache_path, signature) if use_cache else None
        if cached is None:
            cached = self._build(samples, method)
            if use_cache:
                os.makedirs(settings.CACHE_DIR, exist_ok=True)
                torch.save({"signature": signature, **cached}, cache_path)
                print(f"Cached {len(cached['labels'])} crops to {cache_path}")
        self.tensors = cached["tensors"]
        self.labels = cached["labels"]

    @staticmethod
    def _read_cache(path, signature):
        if not os.path.isfile(path):
            return None
        data = torch.load(path)
        if data.get("signature") != signature:
            print("Crop cache is stale, rebuilding")
            return None
        print(f"Loaded crop cache {path}")
        return data

    @staticmethod
    def _build(samples, method):
        tensors, labels, dropped = [], [], 0
        for i, (path, label) in enumerate(samples, 1):
            img = cv2.imread(path)
            crop = crop_hand(img, method=method) if img is not None else None
            if crop is None:
                dropped += 1
                continue
            tensors.append(crop.tensor)
            labels.append(label)
            if i % 500 == 0:
                print(f"Preprocessed {i}/{len(samples)}")
        if dropped:
            print(f"Dropped {dropped} images with no hand found")
        return {"tensors": torch.stack(tensors), "labels": torch.tensor(labels)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], int(self.labels[idx])
