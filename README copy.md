# Acne Image Classification (SVM + CNN)

This project trains two image classification models on folder-labeled acne images:
- SVM baseline (`scikit-learn`)
- CNN (`PyTorch ResNet18 transfer learning`)

It also includes a Streamlit app where users upload an image and choose which model to use for classification.

## Dataset

Expected layout:

```text
acne_dataset/
  acne-cystic/
  closed_comedones/
  perioral_dermatitis/
  rosacea/
```

Each folder name is treated as the label.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train SVM

```bash
python src/train_svm.py
```

Outputs:
- `models/svm_model.joblib`
- `models/svm_scaler.joblib`
- `models/label_encoder.joblib`
- `reports/svm_metrics.json`
- `reports/svm_classification_report.json`
- `reports/svm_confusion_matrix.png`

## Train CNN

```bash
python src/train_cnn.py
```

Outputs:
- `models/cnn_model.pt`
- `reports/cnn_metrics.json`
- `reports/cnn_classification_report.json`
- `reports/cnn_confusion_matrix.png`

## Run Streamlit App

```bash
streamlit run app.py
```

In the UI:
- Upload an image
- Switch between `SVM` and `CNN`
- Click **Classify image** to view predicted class and confidence/probabilities
