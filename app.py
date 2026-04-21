from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import load_cnn_bundle, load_svm_bundle, predict_with_cnn, predict_with_svm


st.set_page_config(page_title="Acne Image Classifier", page_icon=":microscope:", layout="centered")
st.title("Acne Image Classifier")
st.write("Upload a skin image and classify it using either an SVM or CNN model.")

models_dir = ROOT / "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_choice = st.radio("Select model", ["SVM", "CNN"], horizontal=True)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload an image to run prediction.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded image", use_container_width=True)

if model_choice == "SVM" and not (models_dir / "svm_model.joblib").exists():
    st.error("SVM artifacts not found. Run: python src/train_svm.py")
    st.stop()
if model_choice == "CNN" and not (models_dir / "cnn_model.pt").exists():
    st.error("CNN artifacts not found. Run: python src/train_cnn.py")
    st.stop()

if st.button("Classify image", type="primary"):
    with st.spinner("Running inference..."):
        if model_choice == "SVM":
            bundle = load_svm_bundle(models_dir)
            label, confidence, ranked = predict_with_svm(image, bundle)
        else:
            bundle = load_cnn_bundle(models_dir, device)
            label, confidence, ranked = predict_with_cnn(image, bundle, device)

    st.success(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.subheader("Class probabilities")
    probs_table = {"class": [x[0] for x in ranked], "probability": [round(x[1], 4) for x in ranked]}
    st.dataframe(probs_table, use_container_width=True, hide_index=True)
