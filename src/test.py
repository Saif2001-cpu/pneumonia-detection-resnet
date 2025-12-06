import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/chest_xray/test"
MODEL_PATH = "models/best_resnet50.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ----------------------------------------
# 1. Test DataLoader
# ----------------------------------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# ----------------------------------------
# 2. Load model
# ----------------------------------------
model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=DEVICE)

# ----------------------------------------
# 3. Test Evaluation Loop
# ----------------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ----------------------------------------
# 4. Metrics
# ----------------------------------------
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary"
)

print("\n===== Test Set Results =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# ----------------------------------------
# 5. Confusion Matrix
# ----------------------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Test Set")
plt.show()
