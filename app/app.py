# app/app.py
import os
import sys
import streamlit as st
from PIL import Image

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.inference import predict_with_gradcam

MODEL_PATH = "models/best_resnet50.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # order should match ImageFolder

st.set_page_config(page_title="Chest X-Ray Pneumonia Detector", layout="centered")

st.title("🩺 Chest X-Ray Pneumonia Detection (ResNet50 + Grad-CAM)")
st.write("Upload a chest X-ray image and the model will predict **Normal vs Pneumonia** and show a heatmap of suspicious regions.")

uploaded_file = st.file_uploader("Upload chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)

    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing with ResNet50..."):
            pred_class, pred_prob, cam_path = predict_with_gradcam(
                image, MODEL_PATH, CLASS_NAMES, output_cam_path="cam_output.jpg"
            )

        st.markdown(f"### Prediction: **{pred_class}**")
        st.markdown(f"**Confidence:** {pred_prob * 100:.2f}%")

        st.markdown("### Grad-CAM Heatmap (model's focus)")
        cam_image = Image.open(cam_path)
        st.image(cam_image, use_column_width=True)
        st.info("Red/yellow regions indicate where the model is focusing when making the prediction.")
