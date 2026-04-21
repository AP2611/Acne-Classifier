from __future__ import annotations

import json
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


def load_report_metric(reports_dir: Path, file_name: str, metric_key: str) -> float | None:
    path = reports_dir / file_name
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        value = data.get(metric_key)
        return float(value) if value is not None else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def show_prediction_block(title: str, label: str, confidence: float, ranked: list[tuple[str, float]]) -> None:
    st.subheader(title)
    st.success(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence * 100:.2f}%")
    probs_table = {"class": [x[0] for x in ranked], "probability": [round(x[1], 4) for x in ranked]}
    st.dataframe(probs_table, width="stretch", hide_index=True)


st.set_page_config(page_title="Acne Image Classifier", page_icon=":microscope:", layout="centered")
st.title("Acne Image Classifier")
st.write("Upload a skin image and classify it with a baseline (SVM) or advanced (CNN) model.")

models_dir = ROOT / "models"
reports_dir = ROOT / "reports"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.caption("Model roles: SVM = basic baseline model, CNN = advanced deep learning model.")
model_choice = st.radio("Select mode", ["SVM (Basic)", "CNN (Advanced)", "Compare Both"], horizontal=True)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload an image to run prediction.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded image", width="stretch")

needs_svm = model_choice in {"SVM (Basic)", "Compare Both"}
needs_cnn = model_choice in {"CNN (Advanced)", "Compare Both"}

if needs_svm and not (models_dir / "svm_model.joblib").exists():
    st.error("SVM artifacts not found. Run: python src/train_svm.py")
    st.stop()
if needs_cnn and not (models_dir / "cnn_model.pt").exists():
    st.error("CNN artifacts not found. Run: python src/train_cnn.py")
    st.stop()

st.subheader("Saved test metrics")
metrics_cols = st.columns(2)
svm_acc = load_report_metric(reports_dir, "svm_metrics.json", "test_accuracy")
cnn_acc = load_report_metric(reports_dir, "cnn_metrics.json", "test_accuracy")
metrics_cols[0].metric("SVM Test Accuracy", f"{svm_acc * 100:.2f}%" if svm_acc is not None else "N/A")
metrics_cols[1].metric("CNN Test Accuracy", f"{cnn_acc * 100:.2f}%" if cnn_acc is not None else "N/A")

if st.button("Classify image", type="primary"):
    with st.spinner("Running inference..."):
        svm_result = None
        cnn_result = None
        if needs_svm:
            svm_bundle = load_svm_bundle(models_dir)
            svm_result = predict_with_svm(image, svm_bundle)
        if needs_cnn:
            cnn_bundle = load_cnn_bundle(models_dir, device)
            cnn_result = predict_with_cnn(image, cnn_bundle, device)

    if model_choice == "SVM (Basic)":
        label, confidence, ranked = svm_result
        show_prediction_block("SVM (Basic) Prediction", label, confidence, ranked)
    elif model_choice == "CNN (Advanced)":
        label, confidence, ranked = cnn_result
        show_prediction_block("CNN (Advanced) Prediction", label, confidence, ranked)
    else:
        left_col, right_col = st.columns(2)
        with left_col:
            label, confidence, ranked = svm_result
            show_prediction_block("SVM (Basic)", label, confidence, ranked)
        with right_col:
            label, confidence, ranked = cnn_result
            show_prediction_block("CNN (Advanced)", label, confidence, ranked)
