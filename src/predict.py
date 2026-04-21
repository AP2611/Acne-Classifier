from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import build_resnet18_classifier, get_cnn_transform, pil_to_svm_vector


def _pil_to_feature_vector(image: Image.Image) -> np.ndarray:
    vec = pil_to_svm_vector(image)
    return vec.reshape(1, -1)


def load_rf_bundle(models_dir: Path) -> Dict:
    model = joblib.load(models_dir / "rf_model.joblib")
    classes = joblib.load(models_dir / "label_encoder.joblib")
    return {"model": model, "classes": classes}


def predict_with_rf(image: Image.Image, bundle: Dict) -> Tuple[str, float, List[Tuple[str, float]]]:
    vec = _pil_to_feature_vector(image)
    probs = bundle["model"].predict_proba(vec)[0]
    top_idx = int(np.argmax(probs))
    top_label = bundle["classes"][top_idx]
    top_conf = float(probs[top_idx])
    ranked = sorted(
        [(bundle["classes"][idx], float(prob)) for idx, prob in enumerate(probs)],
        key=lambda x: x[1],
        reverse=True,
    )
    return top_label, top_conf, ranked


def load_cnn_bundle(models_dir: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(models_dir / "cnn_model.pt", map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    model = build_resnet18_classifier(num_classes=len(class_to_idx), pretrained=False, dropout_p=0.35)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return {"model": model, "idx_to_class": idx_to_class, "transform": get_cnn_transform(train=False)}


def predict_with_cnn(image: Image.Image, bundle: Dict, device: torch.device) -> Tuple[str, float, List[Tuple[str, float]]]:
    tensor = bundle["transform"](image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = bundle["model"](tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_idx = int(np.argmax(probs))
    top_label = bundle["idx_to_class"][top_idx]
    top_conf = float(probs[top_idx])
    ranked = sorted(
        [(bundle["idx_to_class"][idx], float(prob)) for idx, prob in enumerate(probs)],
        key=lambda x: x[1],
        reverse=True,
    )
    return top_label, top_conf, ranked
