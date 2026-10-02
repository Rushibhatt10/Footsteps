"""
Phase 11: Robust Footstep Detection Model Training.
Trains BaselineFootstepCNN with:
1. Acoustic Domain-Gap Augmentation (speaker EQ, mic EQ, synthetic reverb, noise at various SNRs, gain scaling).
2. Shared preprocessing function (preprocess_audio_window) identical to inference.
3. Class-balanced WeightedRandomSampler to prevent class imbalance issues.
4. Early stopping and learning rate scheduling.
5. Saves robust model checkpoint to models/footstep_detector_robust.pth.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.dataset_loader import create_dataloaders
from src.models import BaselineFootstepCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TrainRobust")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels, _ in loader:
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, f1, prec, rec


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for specs, labels, _ in loader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, f1, prec, rec


def run_training(
    epochs: int = 25,
    batch_size: int = config.BATCH_SIZE,
    lr: float = 1e-3,
    patience: int = 8,
    save_path: Path = config.MODELS_DIR / "footstep_detector_robust.pth"
):
    logger.info("=" * 65)
    logger.info("ROBUST MODEL TRAINING (BaselineFootstepCNN + Domain Augmentations)")
    logger.info(f"Target Output: {save_path}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info("=" * 65)

    # 1. Load data with balanced sampler and acoustic domain augmentations
    logger.info("Loading dataloaders from dataset_v2/manifest.csv...")
    train_loader, val_loader, test_loader, manifest_df = create_dataloaders(
        batch_size=batch_size,
        use_balanced_sampler=True,
        enable_augmentation=True
    )
    logger.info(f"Train windows: {len(train_loader.dataset)} | Val windows: {len(val_loader.dataset)}")

    # 2. Model, Loss, Optimizer, Scheduler
    model = BaselineFootstepCNN(in_channels=1, dropout=0.30).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "train_acc": [], "val_acc": [],
        "train_prec": [], "val_prec": [],
        "train_rec": [], "val_rec": []
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1, tr_prec, tr_rec = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate_epoch(
            model, val_loader, criterion, config.DEVICE
        )

        scheduler.step(val_f1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(val_f1)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_prec"].append(tr_prec)
        history["val_prec"].append(val_prec)
        history["train_rec"].append(tr_rec)
        history["val_rec"].append(val_rec)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss: {tr_loss:.4f}/{val_loss:.4f} | "
            f"F1: {tr_f1:.3f}/{val_f1:.3f} | "
            f"Rec: {tr_rec:.3f}/{val_rec:.3f} | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpoint best model on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "model_type": "BaselineFootstepCNN_Robust",
            }, save_path)
            logger.info(f"  [+] Saved new best robust model (Val F1: {val_f1:.4f}) to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    history_path = config.REPORTS_DIR / "robust_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved history to {history_path}")

    # Plot curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Robust Model Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_f1"], label="Train F1")
    plt.plot(history["val_f1"], label="Val F1")
    plt.title("Robust Model F1 Score")
    plt.xlabel("Epoch")
    plt.legend()

    plot_path = config.REPORTS_DIR / "robust_learning_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved learning curves to {plot_path}")
    logger.info(f"[SUCCESS] Robust Model Training Complete. Best Val F1 = {best_val_f1:.4f}")
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train robust footstep detector")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    run_training(epochs=args.epochs, lr=args.lr)
