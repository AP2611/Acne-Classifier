from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import IMAGE_SIZE, build_svm_features, discover_dataset, make_splits, save_class_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Random Forest acne classifier.")
    parser.add_argument("--dataset", type=Path, default=Path("acne_dataset"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels, class_to_idx = discover_dataset(args.dataset)
    split = make_splits(image_paths, labels, class_to_idx)
    class_names = [split.idx_to_class[i] for i in range(len(split.idx_to_class))]

    # Preprocess raw images into hand-crafted feature vectors (HOG + color histograms).
    X_train = build_svm_features(split.X_train, image_size=IMAGE_SIZE)
    X_val = build_svm_features(split.X_val, image_size=IMAGE_SIZE)
    X_test = build_svm_features(split.X_test, image_size=IMAGE_SIZE)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, split.y_train)
    base_model = search.best_estimator_
    print(f"Best RF params: {search.best_params_}")

    # Calibrate class probabilities for more reliable confidence scores.
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train, split.y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    val_acc = accuracy_score(split.y_val, val_preds)
    test_acc = accuracy_score(split.y_test, test_preds)

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    report = classification_report(split.y_test, test_preds, target_names=class_names, output_dict=True)
    report_path = args.reports_dir / "rf_classification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(split.y_test, test_preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax,
        xticks_rotation=30,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(args.reports_dir / "rf_confusion_matrix.png", dpi=150)
    plt.close(fig)

    joblib.dump(model, args.models_dir / "rf_model.joblib")
    joblib.dump(class_names, args.models_dir / "label_encoder.joblib")
    save_class_mapping(args.models_dir / "class_mapping.json", class_to_idx)

    metrics = {
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "feature_dimension": int(X_train.shape[1]),
        "best_params": search.best_params_,
        "probability_calibration": "sigmoid_cv3",
        "train_class_distribution": np.bincount(split.y_train).tolist(),
    }
    with (args.reports_dir / "rf_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved Random Forest artifacts and reports.")


if __name__ == "__main__":
    main()
