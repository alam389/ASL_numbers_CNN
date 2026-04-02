import os
from torch.utils.data import Dataset
import cv2
from visionPreprocess import crop_hand

class PreprocessedImageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)

        crop = crop_hand(img, method="yolo")
        if crop is None:
            return self.__getitem__((idx + 1) % len(self))

        return crop.tensor, label
