# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes: int, pretrained: bool = True):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_model(model_path: str, num_classes: int, device: str = "cpu"):
    model = get_resnet50(num_classes, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
