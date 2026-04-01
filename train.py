from torch.utils.data import DataLoader, random_split
import torch
from dataset import PreprocessedImageDataset
from models.model_CNN import SimpleCNN
from models.complex_CNN import ASLNet
from models.SVM import SVMClassifier
from training import train, evaluate
import os
import trained_model

DATA_DIR = "data"
BATCH_SIZE = 32
MODEL_PATH = "simple_cnn_model_3.pth"
PREPROCESS_METHOD = "yolo"   # what dataset.py applies; stored in the checkpoint

# class order matches dataset.py (sorted folder names)
CLASS_NAMES = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))) if os.path.isdir(DATA_DIR) else []
NUM_CLASSES = len(CLASS_NAMES) or 2

def main():
    # load full dataset
    dataset = PreprocessedImageDataset(DATA_DIR)

    # compute split sizes
    total = len(dataset)
    print(f"Length of dataset: {total}")

    train_size = int(0.9 * total)
    val_size = int(0.05 * total)
    test_size = total - train_size - val_size

    #Perform Split
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    #test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SimpleCNN(num_classes=NUM_CLASSES)
    train(model, train_loader, epochs=10, lr=1e-3)
    trained_model.save(MODEL_PATH, model, "SimpleCNN", class_names=CLASS_NAMES or None,
                       num_classes=NUM_CLASSES, preprocess_method=PREPROCESS_METHOD)
    print("model saved")
    print("Model_CNN Evaluation")
    print(evaluate(model, val_loader))

    #model_SVM = SVMClassifier(input_dim=input_dim, num_classes=10)
    #train(model_SVM, train_loader, epochs=5, lr=1e-3)
    
    '''model = ASLNet(num_classes= NUM_CLASSES)
    train(model, train_loader, epochs=10, lr=1e-3)
    trained_model.save("cnn_model.pth", model, "ASLNet", class_names=CLASS_NAMES or None, num_classes=NUM_CLASSES, preprocess_method=PREPROCESS_METHOD)
    print("model saved")
    print("Model_ASL Evaluation")
    print(evaluate(model, val_loader))'''

if __name__ == "__main__":
    main()
