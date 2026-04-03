import torch
from device import pick_device
import torch.nn as nn

def train(model, dataloader, epochs=10, lr=1e-3):
    device = pick_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")


def evaluate(model, loader):
    device = pick_device()
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            total_loss += loss.item() * labels.size(0)
            predicted = preds.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"correct: {correct} Total: {total}")

    return avg_loss, accuracy

    
    '''
    Trainning loop:
    1. Forword pass
    2. loss compytation
    3. Backward pass
    4. Parameter update

    Evaluation does not update model

    Cross-entropy loss is the standard for classification, it compares predictied class probabilities to the true label

    
    '''
