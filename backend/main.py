from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import load_cnn_bundle, load_rf_bundle, predict_with_cnn, predict_with_rf

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_dir = ROOT / "models"
reports_dir = ROOT / "reports"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_report_metric(file_name: str, metric_key: str) -> float | None:
    path = reports_dir / file_name
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        value = data.get(metric_key)
        return float(value) if value is not None else None
    except Exception:
        return None

@app.get("/metrics")
async def get_metrics():
    rf_acc = load_report_metric("rf_metrics.json", "test_accuracy")
    cnn_acc = load_report_metric("cnn_metrics.json", "test_accuracy")
    return {
        "rf_accuracy": rf_acc,
        "cnn_accuracy": cnn_acc
    }

@app.post("/classify")
async def classify_image(file: UploadFile = File(...), model_type: str = "both"):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = {}
    
    if model_type in ["rf", "both"]:
        if not (models_dir / "rf_model.joblib").exists():
            results["rf"] = {"error": "RF model not found"}
        else:
            rf_bundle = load_rf_bundle(models_dir)
            label, confidence, ranked = predict_with_rf(image, rf_bundle)
            results["rf"] = {
                "label": label,
                "confidence": confidence,
                "ranked": [{"class": r[0], "probability": r[1]} for r in ranked]
            }

    if model_type in ["cnn", "both"]:
        if not (models_dir / "cnn_model.pt").exists():
            results["cnn"] = {"error": "CNN model not found"}
        else:
            cnn_bundle = load_cnn_bundle(models_dir, device)
            label, confidence, ranked = predict_with_cnn(image, cnn_bundle, device)
            results["cnn"] = {
                "label": label,
                "confidence": confidence,
                "ranked": [{"class": r[0], "probability": r[1]} for r in ranked]
            }

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
