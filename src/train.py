# src/train.py
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from dataset import get_dataloaders
from model import get_resnet50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 16
DATA_DIR = "data/chest_xray"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)

    model = get_resnet50(num_classes=num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        train_losses = []
        y_true_train, y_pred_train = [], []

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, preds = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        train_acc = accuracy_score(y_true_train, y_pred_train)

        # ---------- VALIDATION ----------
        model.eval()
        val_losses = []
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_acc = accuracy_score(y_true_val, y_pred_val)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_val, y_pred_val, average="weighted", zero_division=0
        )

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {sum(train_losses)/len(train_losses):.4f} "
              f"Val Loss: {sum(val_losses)/len(val_losses):.4f} "
              f"Val Acc: {val_acc:.4f} "
              f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_resnet50.pth"))
            print("✅ Saved new best model")

if __name__ == "__main__":
    train()
