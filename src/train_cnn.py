from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data import build_resnet18_classifier, discover_dataset, get_cnn_transform, load_pil_rgb, make_splits, save_class_mapping


class PathImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_pil_rgb(self.paths[idx])
        image = self.transform(image)
        label = int(self.labels[idx])
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN acne classifier.")
    parser.add_argument("--dataset", type=Path, default=Path("acne_dataset"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def fit_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device):
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    wait = 0

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, best_val_acc


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths, labels, class_to_idx = discover_dataset(args.dataset)
    split = make_splits(image_paths, labels, class_to_idx)
    class_names = [split.idx_to_class[i] for i in range(len(split.idx_to_class))]

    train_ds = PathImageDataset(split.X_train, split.y_train, transform=get_cnn_transform(train=True))
    val_ds = PathImageDataset(split.X_val, split.y_val, transform=get_cnn_transform(train=False))
    test_ds = PathImageDataset(split.X_test, split.y_test, transform=get_cnn_transform(train=False))

    class_counts = np.bincount(split.y_train, minlength=len(class_names))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[split.y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_resnet18_classifier(num_classes=len(class_names), pretrained=True, dropout_p=0.35)
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.08,
    )
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=1e-4)

    print("Stage 1: training classifier head.")
    model, best_head_val = fit_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs_head,
        patience=args.patience,
        device=device,
    )

    print("Stage 2: fine-tuning layer4 + fc.")
    for p in model.layer4.parameters():
        p.requires_grad = True
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_finetune,
        weight_decay=1e-4,
    )
    model, best_ft_val = fit_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs_finetune,
        patience=args.patience,
        device=device,
    )

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {test_acc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with (args.reports_dir / "cnn_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax,
        xticks_rotation=30,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(args.reports_dir / "cnn_confusion_matrix.png", dpi=150)
    plt.close(fig)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "image_size": [224, 224],
            "best_head_val_acc": best_head_val,
            "best_finetune_val_acc": best_ft_val,
        },
        args.models_dir / "cnn_model.pt",
    )
    save_class_mapping(args.models_dir / "class_mapping.json", class_to_idx)

    metrics = {
        "best_head_validation_accuracy": best_head_val,
        "best_finetune_validation_accuracy": best_ft_val,
        "test_accuracy": test_acc,
    }
    with (args.reports_dir / "cnn_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved CNN artifacts and reports.")


if __name__ == "__main__":
    main()
