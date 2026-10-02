"""
Multi-Class Training Pipeline for Footstep vs Clap vs Knock vs Other.
Trains MultiClassAudioNet with:
1. 3-channel (mel + delta + delta²) 80-band spectrogram input
2. 4-way class-weighted Cross-Entropy loss
3. WeightedRandomSampler for balanced batch distributions
4. Multi-domain acoustic augmentations applied to all classes
5. Learning rate scheduling and early stopping based on validation Macro F1
6. Saves model to models/footstep_multiclass_v4.pth
"""

import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.dataset_multiclass import create_multiclass_dataloaders
from src.models_multiclass import MultiClassAudioNet, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TrainMultiClass")

CLASS_NAMES = ["other", "footstep", "clap", "knock"]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for specs, labels, _ in loader:
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for specs, labels, _ in loader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

    return avg_loss, acc, f1_macro, f1_per_class, cm


def train_multiclass_pipeline(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = config.DEVICE,
    output_model_path: Path = config.MODELS_DIR / "footstep_multiclass_v4.pth"
):
    logger.info("=" * 70)
    logger.info("STARTING MULTI-CLASS AUDIO EVENT MODEL TRAINING")
    logger.info("=" * 70)

    # 1. Dataloaders
    train_loader, val_loader, test_loader, dataset_info = create_multiclass_dataloaders(
        batch_size=batch_size
    )
    logger.info(f"Dataset Windows -> Train: {dataset_info['train_windows']} | Val: {dataset_info['val_windows']} | Test: {dataset_info['test_windows']}")

    # 2. Model
    model = MultiClassAudioNet(in_channels=3, num_classes=4, dropout=0.35).to(device)
    logger.info(f"MultiClassAudioNet initialized with {count_parameters(model):,} trainable parameters.")

    # 3. Class-weighted CrossEntropy
    # Normalizing weights for numerical stability
    weights = torch.tensor(dataset_info["class_weights"], dtype=torch.float32).to(device)
    weights = weights / weights.sum() * 4.0
    criterion = nn.CrossEntropyLoss(weight=weights)
    logger.info(f"CrossEntropyLoss class weights: {[round(w, 3) for w in weights.tolist()]}")

    # 4. Optimizer & Schedulers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "val_f1_other": [], "val_f1_step": [], "val_f1_clap": [], "val_f1_knock": []
    }

    best_val_f1 = 0.0
    best_epoch = 0
    patience = 8
    patience_counter = 0

    print("-" * 90)
    print(f"{'Epoch':7s} | {'Tr Loss':8s} | {'Tr Acc':7s} | {'Val Loss':8s} | {'Val Acc':7s} | {'Macro F1':8s} | {'Step F1':8s} | {'Clap F1':8s} | {'Knock F1':8s}")
    print("-" * 90)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_per_class, cm = evaluate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(val_f1)
        history["val_f1_other"].append(float(val_per_class[0]))
        history["val_f1_step"].append(float(val_per_class[1]))
        history["val_f1_clap"].append(float(val_per_class[2]))
        history["val_f1_knock"].append(float(val_per_class[3]))

        print(
            f"Ep {epoch:02d}/{epochs:02d} | "
            f"{tr_loss:8.4f} | {tr_acc*100:6.1f}% | "
            f"{val_loss:8.4f} | {val_acc*100:6.1f}% | "
            f"{val_f1:8.4f} | "
            f"{val_per_class[1]:8.4f} | {val_per_class[2]:8.4f} | {val_per_class[3]:8.4f}"
        )

        # Checkpoint on best validation macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_per_class_f1": val_per_class.tolist(),
                "class_names": CLASS_NAMES,
                "in_channels": 3,
                "num_classes": 4,
                "history": history
            }, output_model_path)
            logger.info(f"[*] New best model saved to {output_model_path.name} (Val Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

    print("-" * 90)
    logger.info(f"Training Complete! Best Validation Macro F1: {best_val_f1:.4f} at epoch {best_epoch}.")

    # Save training curves plot
    plot_path = config.REPORTS_DIR / "multiclass_training_curves.png"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_multiclass_history(history, plot_path)
    logger.info(f"Saved training curves to {plot_path}")

    return history


def plot_multiclass_history(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="tab:blue", lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="tab:orange", lw=2)
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Macro F1 & Accuracy
    axes[1].plot(history["val_f1"], label="Val Macro F1", color="tab:green", lw=2)
    axes[1].plot(history["val_acc"], label="Val Accuracy", color="tab:purple", lw=2, linestyle="--")
    axes[1].set_title("Overall Validation Performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Per-Class F1
    axes[2].plot(history["val_f1_step"], label="Footstep F1", color="#2b5c8f", lw=2)
    axes[2].plot(history["val_f1_clap"], label="Clap F1", color="#d95f02", lw=2)
    axes[2].plot(history["val_f1_knock"], label="Knock F1", color="#7570b3", lw=2)
    axes[2].plot(history["val_f1_other"], label="Other F1", color="#1b9e77", lw=1.5, linestyle=":")
    axes[2].set_title("Per-Class Validation F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Class Audio Event Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_multiclass_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
