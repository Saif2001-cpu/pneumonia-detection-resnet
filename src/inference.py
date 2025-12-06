# src/inference.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import load_model
from grad_cam import GradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def prepare_image(pil_image):
    tensor = preprocess(pil_image).unsqueeze(0)
    return tensor.to(DEVICE)

def predict_with_gradcam(pil_image, model_path, class_names, output_cam_path="cam.jpg"):
    num_classes = len(class_names)
    model = load_model(model_path, num_classes=num_classes, device=DEVICE)

    input_tensor = prepare_image(pil_image)

    grad_cam = GradCAM(model, target_layer_name="layer4")
    cam = grad_cam.generate(input_tensor)  # [224, 224]

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        pred_prob = float(probs[pred_idx])

    # Create heatmap overlay
    img_np = np.array(pil_image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.4 * heatmap + 0.6 * img_np)

    cv2.imwrite(output_cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return pred_class, pred_prob, output_cam_path

if __name__ == "__main__":

    TEST_IMAGE = "test_image.jpeg"   # <-- replace with path to an actual X-ray

    image = Image.open(TEST_IMAGE).convert("RGB")

    pred_class, pred_prob, cam_path = predict_with_gradcam(
        image,
        model_path="models/best_resnet50.pth",
        class_names=["NORMAL", "PNEUMONIA"],
        output_cam_path="cam_output.jpg"
    )

    print("Prediction:", pred_class)
    print("Confidence:", pred_prob)
    print("Grad-CAM saved to:", cam_path)
