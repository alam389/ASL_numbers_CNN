import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from dataset import PreprocessedImageDataset
from training import train, evaluate
import trained_model
import settings
from settings import (ARCH, AUGMENT, DATA_DIR, BATCH_SIZE, MODEL_PATH, EPOCHS, LEARNING_RATE,
                      NUM_WORKERS, TRAIN_SPLIT, VAL_SPLIT)

PREPROCESS_METHOD = settings.TRAIN_PREPROCESS_METHOD   # what dataset.py applies; stored in the checkpoint

CLASS_NAMES = settings.class_names(DATA_DIR)
NUM_CLASSES = len(CLASS_NAMES) or settings.LEGACY_NUM_CLASSES

class Augmented(Dataset):
    """Random shift/rotate/scale plus brightness/contrast on tensors in [-1, 1] (training only)."""

    def __init__(self, subset):
        self.subset = subset
        self.affine = T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), fill=-1.0)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x = self.affine(x)
        x = (x * (0.8 + 0.4 * torch.rand(1)) + (torch.rand(1) - 0.5) * 0.4).clamp(-1, 1)
        return x, y


def main():
    # load full dataset
    dataset = PreprocessedImageDataset(DATA_DIR)

    # compute split sizes
    total = len(dataset)
    print(f"Length of dataset: {total}")

    train_size = int(TRAIN_SPLIT * total)
    val_size = int(VAL_SPLIT * total)
    test_size = total - train_size - val_size

    #Perform Split
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    if AUGMENT:
        train_ds = Augmented(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    #test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = trained_model.ARCHITECTURES[ARCH](num_classes=NUM_CLASSES)
    train(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    trained_model.save(MODEL_PATH, model, ARCH, class_names=CLASS_NAMES or None,
                       num_classes=NUM_CLASSES, preprocess_method=PREPROCESS_METHOD)
    print("model saved")
    print(f"{ARCH} Evaluation")
    print(evaluate(model, val_loader))

if __name__ == "__main__":
    main()
