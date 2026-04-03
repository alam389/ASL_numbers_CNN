import os
import torch
from device import pick_device
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from dataset import PreprocessedImageDataset
import matplotlib.pyplot as plt
import seaborn as sns


import trained_model

from settings import MODEL_PATH, EVAL_DIR, EVAL_BATCH_SIZE, LEGACY_ARCH, LEGACY_NUM_CLASSES

device = pick_device()

def load_model():
    return trained_model.load(MODEL_PATH, device=device, arch=LEGACY_ARCH, num_classes=LEGACY_NUM_CLASSES).model

def get_dataloader():
    dataset = PreprocessedImageDataset(EVAL_DIR)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # Class names come from folder names
    class_names = sorted(os.listdir(EVAL_DIR))

    return loader, class_names

def evaluate(model, loader, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # === TEXT REPORT ===
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # === CONFUSION MATRIX ===
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # === PER-CLASS F1, PRECISION, RECALL BAR CHART ===
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        values = [report[c][metric] for c in class_names]
        plt.plot(class_names, values, marker="o", label=metric)

    plt.ylim(0, 1)
    plt.title("Per-Class Metrics")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === PREDICTION DISTRIBUTION ===
    plt.figure(figsize=(6, 4))
    sns.countplot(x=all_preds, palette="viridis")
    plt.title("Prediction Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.tight_layout()
    plt.show()
def main():
    print("Loading model...")
    model = load_model()

    print("Loading evaluation dataset...")
    loader, class_names = get_dataloader()

    print("Running evaluation...")
    evaluate(model, loader, class_names)

if __name__ == "__main__":
    main()