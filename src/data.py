from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage import exposure
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
import torch.nn as nn

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetSplit:
    X_train: List[Path]
    y_train: np.ndarray
    X_val: List[Path]
    y_val: np.ndarray
    X_test: List[Path]
    y_test: np.ndarray
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]


def discover_dataset(dataset_dir: Path) -> Tuple[List[Path], np.ndarray, Dict[str, int]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    class_names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found under: {dataset_dir}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_paths: List[Path] = []
    labels: List[int] = []

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for image_path in sorted(class_dir.glob(ext)):
                image_paths.append(image_path)
                labels.append(class_to_idx[class_name])

    if not image_paths:
        raise ValueError(f"No images found under class folders in {dataset_dir}")

    return image_paths, np.asarray(labels, dtype=np.int64), class_to_idx


def make_splits(
    image_paths: Sequence[Path],
    labels: np.ndarray,
    class_to_idx: Dict[str, int],
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> DatasetSplit:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        list(image_paths),
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # val_size is relative to train+val, so convert to fraction of trainval.
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_trainval,
    )

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
    )


def get_cnn_transform(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_pil_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def pil_to_svm_vector(image: Image.Image, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = image.convert("RGB").resize(image_size)
    rgb_arr = np.asarray(image, dtype=np.float32) / 255.0

    # Local contrast normalization improves robustness to uneven lighting.
    gray = np.mean(rgb_arr, axis=2)
    gray = exposure.equalize_adapthist(gray, clip_limit=0.03)

    # Capture both coarse and fine texture patterns.
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    hog_features_fine = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # Uniform LBP histogram keeps local micro-texture cues.
    gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, P=8, R=1.0, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)
    lbp_features = lbp_hist.astype(np.float32)

    # Add low-dimensional color histograms to keep color cues for acne classes.
    color_hist_parts: List[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(rgb_arr[:, :, channel], bins=24, range=(0.0, 1.0), density=True)
        color_hist_parts.append(hist.astype(np.float32))

    hsv_arr = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    for channel in range(3):
        hist, _ = np.histogram(hsv_arr[:, :, channel], bins=24, range=(0.0, 1.0), density=True)
        color_hist_parts.append(hist.astype(np.float32))

    color_features = np.concatenate(color_hist_parts)
    return np.concatenate(
        [
            hog_features.astype(np.float32),
            hog_features_fine.astype(np.float32),
            lbp_features,
            color_features,
        ],
        axis=0,
    )


def image_to_svm_vector(path: Path, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = load_pil_rgb(path)
    return pil_to_svm_vector(image, image_size=image_size)


def build_svm_features(paths: Sequence[Path], image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    vectors = [image_to_svm_vector(path, image_size=image_size) for path in paths]
    return np.stack(vectors, axis=0)


def build_efficientnet_b0_classifier(num_classes: int, pretrained: bool = True, dropout_p: float = 0.35):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
    return model


def save_class_mapping(path: Path, class_to_idx: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
