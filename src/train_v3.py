"""
Phase 10: Model V3 Training — EfficientFootstepNet + Dataset V3.

Runs 4 controlled experiments:
  V3-A: EfficientFootstepNet + Dataset V3 + standard BCE
  V3-B: EfficientFootstepNet + Dataset V3 + weighted BCE (pos_weight=5.0)
  V3-C: EfficientFootstepNet + Dataset V3 + focal loss + label smoothing 0.05
  V3-D: Best-from-ABC + SpecAugment + Mixup

Each experiment:
- Cosine annealing LR schedule
- Balanced WeightedRandomSampler
- Per-epoch validation metrics
- Checkpoint saved on best validation F1
- Full threshold sweep on validation set
- Selects best model overall
- Reports final test benchmark (untouched test set)
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio
from src.preprocessing import pad_or_trim
from src.feature_extraction_v3 import extract_log_mel_v3, V3_N_MELS
from src.models import EfficientFootstepNet, count_parameters
from src.augment import random_time_shift, random_gain, add_subtle_noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainV3")

# =====================================================================
# Paths
# =====================================================================
DATASET_V3_DIR = config.PROJECT_ROOT / "dataset_v3"
V3_MANIFEST    = DATASET_V3_DIR / "manifest.csv"

# =====================================================================
# Hyperparameters
# =====================================================================
EPOCHS         = 40
BATCH_SIZE     = 32
LR             = 5e-4
WEIGHT_DECAY   = 2e-4
PATIENCE       = 10         # early stopping
RANDOM_SEED    = config.RANDOM_SEED
DEVICE         = config.DEVICE
THRESHOLDS     = np.arange(0.25, 0.85, 0.05).tolist()


# =====================================================================
# SpecAugment (frequency + time masking applied on spectrogram tensor)
# =====================================================================
def spec_augment(
    spec: np.ndarray,
    freq_mask_max: int = 8,
    time_mask_max: int = 6,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> np.ndarray:
    """
    Applies SpecAugment to a (C, F, T) spectrogram:
    - freq_mask: zeros out up to freq_mask_max consecutive mel bands
    - time_mask: zeros out up to time_mask_max consecutive time frames
    """
    s = spec.copy()
    _, n_mels, n_frames = s.shape

    for _ in range(n_freq_masks):
        f = np.random.randint(0, min(freq_mask_max, n_mels))
        f0 = np.random.randint(0, max(1, n_mels - f))
        s[:, f0:f0 + f, :] = 0.0

    for _ in range(n_time_masks):
        t = np.random.randint(0, min(time_mask_max, n_frames))
        t0 = np.random.randint(0, max(1, n_frames - t))
        s[:, :, t0:t0 + t] = 0.0

    return s


# =====================================================================
# Dataset V3 AudioWindowDataset (3-channel)
# =====================================================================
class AudioWindowDatasetV3(Dataset):
    """
    Dataset loader for Dataset V3 using 3-channel (mel+delta+delta²) features.
    Supports SpecAugment and Mixup at the batch level.
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        is_train: bool = False,
        use_spec_augment: bool = False,
    ):
        self.is_train = is_train
        self.use_spec_augment = use_spec_augment and is_train
        self.window_size = config.WINDOW_SIZE
        # Footstep: denser hop (0.25s) for more positive windows
        self.pos_hop = config.SAMPLE_RATE // 4   # 4000 samples = 0.25s
        self.neg_hop = config.HOP_SIZE            # 8000 samples = 0.50s

        df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.samples: List[Dict] = []
        self._index_samples(df)

    def _index_samples(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            fpath = config.PROJECT_ROOT / row["relative_path"]
            if not fpath.exists():
                continue
            try:
                audio, _ = load_audio(fpath, target_sr=config.SAMPLE_RATE, mono=True)
            except Exception:
                continue

            label = int(row["binary_label"])
            hop = self.pos_hop if (self.is_train and label == 1) else self.neg_hop

            # Sliding window
            start = 0
            while start + self.window_size <= len(audio):
                win = audio[start:start + self.window_size]
                rms = float(np.sqrt(np.mean(win ** 2)))
                if rms >= 0.0005:
                    self.samples.append({
                        "audio": win.astype(np.float32),
                        "label": label,
                        "original_class": row["original_class"],
                        "filename": row["filename"],
                        "source_id": row["source_id"],
                    })
                start += hop

            # If audio shorter than window, pad and include
            if len(audio) < self.window_size and len(audio) > 0:
                padded = pad_or_trim(audio, self.window_size)
                rms = float(np.sqrt(np.mean(padded ** 2)))
                if rms >= 0.0005:
                    self.samples.append({
                        "audio": padded.astype(np.float32),
                        "label": label,
                        "original_class": row["original_class"],
                        "filename": row["filename"],
                        "source_id": row["source_id"],
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        s = self.samples[idx]
        audio = s["audio"].copy()

        if self.is_train:
            # Audio-level augmentation (conservative)
            if np.random.random() < 0.5:
                audio = random_time_shift(audio, config.SAMPLE_RATE, max_shift_sec=0.3)
            if np.random.random() < 0.4:
                audio = random_gain(audio, 0.75, 1.25)
            if np.random.random() < 0.3:
                audio = add_subtle_noise(audio, noise_factor=np.random.uniform(0.002, 0.01))

        audio = pad_or_trim(audio, self.window_size)
        spec = extract_log_mel_v3(audio)  # (3, 80, T)

        if self.use_spec_augment:
            spec = spec_augment(spec)

        tensor = torch.from_numpy(spec).float()
        label  = torch.tensor(s["label"], dtype=torch.float32)
        meta   = {k: s[k] for k in ("filename", "source_id", "original_class")}
        return tensor, label, meta

    def get_sampler(self) -> WeightedRandomSampler:
        labels = [s["label"] for s in self.samples]
        pos = sum(labels)
        neg = len(labels) - pos
        if pos == 0 or neg == 0:
            weights = [1.0] * len(labels)
        else:
            w_pos = 1.0 / pos
            w_neg = 1.0 / neg
            weights = [w_pos if l == 1 else w_neg for l in labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_loaders(
    manifest_df: pd.DataFrame,
    use_spec_augment: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = AudioWindowDatasetV3(manifest_df, "train", is_train=True, use_spec_augment=use_spec_augment)
    val_ds   = AudioWindowDatasetV3(manifest_df, "validation", is_train=False)
    test_ds  = AudioWindowDatasetV3(manifest_df, "test",  is_train=False)

    sampler = train_ds.get_sampler()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos = sum(s["label"] for s in train_ds.samples)
    neg = len(train_ds.samples) - pos
    logger.info(f"  Train windows:  {len(train_ds.samples)} ({pos} pos, {neg} neg, ratio 1:{neg/max(pos,1):.1f})")
    logger.info(f"  Val windows:    {len(val_ds.samples)}")
    logger.info(f"  Test windows:   {len(test_ds.samples)}")
    return train_loader, val_loader, test_loader


# =====================================================================
# Loss Functions
# =====================================================================
class FocalLoss(nn.Module):
    """Binary focal loss with optional label smoothing."""
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def get_loss_fn(experiment: str) -> nn.Module:
    """Returns loss function for the given experiment name."""
    if experiment == "V3-A":
        return nn.BCEWithLogitsLoss()
    elif experiment == "V3-B":
        pos_weight = torch.tensor([5.0], device=DEVICE)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif experiment == "V3-C":
        pos_weight = torch.tensor([4.0], device=DEVICE)
        return FocalLoss(gamma=2.0, pos_weight=pos_weight, label_smoothing=0.05)
    elif experiment == "V3-D":
        pos_weight = torch.tensor([4.0], device=DEVICE)
        return FocalLoss(gamma=2.0, pos_weight=pos_weight, label_smoothing=0.05)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


# =====================================================================
# Mixup (audio-level)
# =====================================================================
def mixup_batch(
    specs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies mixup to a batch of (3, H, W) spectrograms."""
    if alpha <= 0:
        return specs, labels
    lam = float(np.random.beta(alpha, alpha))
    batch_size = specs.size(0)
    idx = torch.randperm(batch_size)
    mixed = lam * specs + (1 - lam) * specs[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed, mixed_labels


# =====================================================================
# Train / Eval Loops
# =====================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    use_mixup: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels, _ in loader:
        specs  = specs.to(DEVICE)
        labels = labels.to(DEVICE)

        if use_mixup:
            specs, labels = mixup_batch(specs, labels, alpha=0.2)

        optimizer.zero_grad()
        logits = model(specs)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        hard_labels = labels.detach().cpu().numpy()
        if use_mixup:
            hard_labels = (hard_labels >= 0.5).astype(float)
        all_preds.extend((probs >= 0.5).astype(int).tolist())
        all_labels.extend(hard_labels.astype(int).tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return {"loss": avg_loss, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.5,
    collect_meta: bool = False,
) -> Dict:
    model.eval()
    all_probs, all_labels, all_meta = [], [], []

    for specs, labels, meta in loader:
        specs = specs.to(DEVICE)
        logits = model(specs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
        if collect_meta:
            for i in range(len(labels)):
                all_meta.append({k: meta[k][i] for k in meta})

    preds = (np.array(all_probs) >= threshold).astype(int)
    labels_arr = np.array(all_labels, dtype=int)

    acc = accuracy_score(labels_arr, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_arr, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels_arr, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    result = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "all_probs": all_probs,
        "all_labels": list(labels_arr),
    }
    if collect_meta:
        result["meta"] = all_meta
    return result


def threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    thresholds: List[float] = THRESHOLDS,
) -> pd.DataFrame:
    """Sweeps over thresholds and returns per-threshold metrics on the given split."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for specs, labels, _ in loader:
            logits = model(specs.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    labels_arr = np.array(all_labels, dtype=int)
    rows = []
    for t in thresholds:
        preds = (np.array(all_probs) >= t).astype(int)
        acc = accuracy_score(labels_arr, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels_arr, preds, average="binary", zero_division=0)
        cm = confusion_matrix(labels_arr, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        rows.append({
            "threshold": round(t, 2),
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positive_rate": round(float(fp / max(fp + tn, 1)), 4),
        })
    return pd.DataFrame(rows)


# =====================================================================
# Recording-Level Evaluation
# =====================================================================
def recording_level_eval(
    model: nn.Module,
    loader: DataLoader,
    threshold: float,
) -> Dict:
    """
    Aggregate window predictions per source file → recording-level metrics.
    A recording is FOOTSTEP if ANY window exceeds threshold.
    """
    result = evaluate(model, loader, threshold=threshold, collect_meta=True)
    meta = result["meta"]
    probs = result["all_probs"]
    labels = result["all_labels"]

    # Group by source_id
    file_groups: Dict[str, Dict] = {}
    for prob, label, m in zip(probs, labels, meta):
        sid = m["source_id"]
        if sid not in file_groups:
            file_groups[sid] = {"probs": [], "label": label, "original_class": m["original_class"]}
        file_groups[sid]["probs"].append(prob)

    y_true, y_pred = [], []
    fp_classes: Dict[str, int] = {}
    fn_classes: Dict[str, int] = {}

    for sid, grp in file_groups.items():
        true = int(grp["label"])
        pred = int(any(p >= threshold for p in grp["probs"]))
        y_true.append(true)
        y_pred.append(pred)
        if pred == 1 and true == 0:
            cat = grp["original_class"]
            fp_classes[cat] = fp_classes.get(cat, 0) + 1
        if pred == 0 and true == 1:
            cat = grp["original_class"]
            fn_classes[cat] = fn_classes.get(cat, 0) + 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "total_files": len(file_groups),
        "fp_by_class": fp_classes,
        "fn_by_class": fn_classes,
    }


# =====================================================================
# Single Experiment Run
# =====================================================================
def run_experiment(
    experiment: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_mixup: bool = False,
) -> Tuple[EfficientFootstepNet, Dict, float]:
    """
    Trains one experiment. Returns (best_model, history, best_val_f1).
    """
    logger.info(f"\n{'='*65}")
    logger.info(f"  EXPERIMENT {experiment}")
    logger.info(f"{'='*65}")

    torch.manual_seed(RANDOM_SEED)
    model = EfficientFootstepNet(in_channels=3, dropout=0.35).to(DEVICE)
    logger.info(f"  Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = get_loss_fn(experiment)

    best_val_f1 = 0.0
    best_state  = None
    patience_cnt = 0
    history = {"train": [], "val": []}

    ckpt_path = config.MODELS_DIR / f"model_v3_{experiment.lower().replace('-', '_')}_ckpt.pth"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, use_mixup=use_mixup)
        val_metrics   = evaluate(model, val_loader, threshold=0.50)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"  [{experiment}] Ep {epoch:02d}/{EPOCHS} | "
            f"Loss {train_metrics['loss']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f} | "
            f"Val Prec {val_metrics['precision']:.4f} | "
            f"Val Rec {val_metrics['recall']:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            torch.save({
                "model_state_dict": best_state,
                "epoch": epoch,
                "val_f1": best_val_f1,
                "experiment": experiment,
            }, ckpt_path)
            logger.info(f"  ↑ New best Val F1: {best_val_f1:.4f}  (saved checkpoint)")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold sweep on validation
    logger.info(f"  Running threshold sweep on validation ...")
    sweep_df = threshold_sweep(model, val_loader)
    best_t_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    best_threshold = float(best_t_row["threshold"])
    logger.info(f"  Best val threshold: {best_threshold:.2f}  (F1={best_t_row['f1']:.4f})")

    return model, history, best_val_f1, best_threshold, sweep_df


# =====================================================================
# Main Pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Model V3 Training Pipeline")
    parser.add_argument(
        "--experiments", nargs="+",
        default=["V3-A", "V3-B", "V3-C", "V3-D"],
        help="Experiments to run (e.g. V3-A V3-B V3-C V3-D)"
    )
    parser.add_argument("--manifest", type=str, default=str(V3_MANIFEST))
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    logger.info("=" * 65)
    logger.info("MODEL V3 TRAINING — EfficientFootstepNet + Dataset V3")
    logger.info("=" * 65)
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Features: 3-channel Log-Mel (80 bands) + Delta + Delta²")
    logger.info(f"  Architecture: EfficientFootstepNet")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run build_dataset_v3.py first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    logger.info(f"  Manifest loaded: {len(df)} files")
    logger.info("\n" + pd.crosstab(df["split"], df["binary_class"], margins=True).to_string())

    # Build loaders (V3-D gets SpecAugment)
    logger.info("\nIndexing Dataset V3 windows (this may take a minute)...")
    # Pre-build standard loaders once — reuse for V3-A/B/C
    train_loader_std, val_loader, test_loader = make_loaders(df, use_spec_augment=False)
    # SpecAugment loader for V3-D
    train_loader_aug, _, _ = make_loaders(df, use_spec_augment=True)

    results = {}
    best_overall_name = None
    best_overall_f1   = 0.0
    best_overall_model = None
    best_overall_threshold = 0.50

    for exp in args.experiments:
        use_mixup = (exp == "V3-D")
        t_loader  = train_loader_aug if exp == "V3-D" else train_loader_std

        model, history, val_f1, best_t, sweep_df = run_experiment(
            exp, t_loader, val_loader, use_mixup=use_mixup
        )

        # Save history
        hist_path = config.REPORTS_DIR / f"model_v3_{exp.lower().replace('-', '_')}_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save threshold sweep
        sweep_path = config.REPORTS_DIR / f"model_v3_{exp.lower().replace('-', '_')}_threshold.csv"
        sweep_df.to_csv(sweep_path, index=False)

        # Save model
        save_path = config.MODELS_DIR / f"model_v3_{exp.lower().replace('-', '_')}.pth"
        torch.save({
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "experiment": exp,
            "val_f1": val_f1,
            "selected_threshold": best_t,
            "feature_type": "log_mel_v3_3ch",
            "n_mels": V3_N_MELS,
        }, save_path)
        logger.info(f"  Saved {exp} → {save_path.name}")

        results[exp] = {
            "val_f1": val_f1,
            "selected_threshold": best_t,
            "model_path": str(save_path),
        }

        if val_f1 > best_overall_f1:
            best_overall_f1 = val_f1
            best_overall_name = exp
            best_overall_model = model
            best_overall_threshold = best_t

    logger.info(f"\n{'='*65}")
    logger.info(f"  BEST EXPERIMENT: {best_overall_name}  (Val F1={best_overall_f1:.4f})")
    logger.info(f"  SELECTED THRESHOLD: {best_overall_threshold:.2f}")
    logger.info(f"{'='*65}")

    # ---------------------------------------------------------------
    # FINAL EVALUATION ON UNTOUCHED TEST SET
    # ---------------------------------------------------------------
    logger.info("\n>>> FINAL TEST EVALUATION (UNTOUCHED TEST SET) <<<")
    window_test = evaluate(best_overall_model, test_loader, threshold=best_overall_threshold, collect_meta=True)
    rec_test    = recording_level_eval(best_overall_model, test_loader, threshold=best_overall_threshold)

    # Inference speed
    dummy = torch.zeros(1, 3, V3_N_MELS, 48).to(DEVICE)
    best_overall_model.eval()
    with torch.no_grad():
        _ = best_overall_model(dummy)
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = best_overall_model(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    avg_inf_ms = float(np.mean(times[50:]))   # discard warmup

    final_report = {
        "best_experiment": best_overall_name,
        "selected_threshold": best_overall_threshold,
        "parameters": count_parameters(best_overall_model),
        "avg_inference_ms": round(avg_inf_ms, 3),
        "window_metrics": {
            k: window_test[k]
            for k in ("accuracy", "precision", "recall", "f1", "false_positives",
                       "false_positive_rate", "false_negatives", "true_positives", "true_negatives")
            if k in window_test
        },
        "recording_metrics": {
            k: rec_test[k]
            for k in ("accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "total_files")
        },
        "test_fp_by_category": rec_test["fp_by_class"],
        "test_fn_by_category": rec_test["fn_by_class"],
        "all_experiments": results,
    }

    report_path = config.REPORTS_DIR / "model_v3_final_test.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    # Classification report
    preds_arr = (np.array(window_test["all_probs"]) >= best_overall_threshold).astype(int)
    clf_report = classification_report(
        window_test["all_labels"], preds_arr,
        target_names=["NON_FOOTSTEP", "FOOTSTEP"]
    )
    clf_path = config.REPORTS_DIR / "model_v3_final_classification_report.txt"
    clf_path.write_text(clf_report)

    # Print summary
    logger.info(f"\n{'─'*65}")
    logger.info(f"  MODEL V3 FINAL TEST RESULTS")
    logger.info(f"{'─'*65}")
    logger.info(f"  Experiment:           {best_overall_name}")
    logger.info(f"  Threshold:            {best_overall_threshold:.2f}")
    logger.info(f"  Parameters:           {count_parameters(best_overall_model):,}")
    logger.info(f"  Avg CPU Inference:    {avg_inf_ms:.2f} ms")
    logger.info(f"\n  Window-Level Metrics:")
    logger.info(f"    Accuracy:           {window_test['accuracy']:.4f}")
    logger.info(f"    Precision:          {window_test['precision']:.4f}")
    logger.info(f"    Recall:             {window_test['recall']:.4f}")
    logger.info(f"    F1:                 {window_test['f1']:.4f}")
    logger.info(f"    False Positives:    {window_test['fp']}")
    logger.info(f"\n  Recording-Level Metrics:")
    logger.info(f"    Accuracy:           {rec_test['accuracy']:.4f}")
    logger.info(f"    Precision:          {rec_test['precision']:.4f}")
    logger.info(f"    Recall:             {rec_test['recall']:.4f}")
    logger.info(f"    F1:                 {rec_test['f1']:.4f}")
    logger.info(f"    TP={rec_test['tp']} FP={rec_test['fp']} TN={rec_test['tn']} FN={rec_test['fn']}")
    logger.info(f"\n  FP by category:  {rec_test['fp_by_class']}")
    logger.info(f"  FN by category:  {rec_test['fn_by_class']}")
    logger.info(f"{'─'*65}")
    logger.info(f"\n  Final report:  {report_path}")
    logger.info(f"  Clf report:    {clf_path}")
    logger.info("  Done.")

    # Comparison vs V2
    logger.info("\n>>> V2 vs V3 COMPARISON <<<")
    v2_path = config.REPORTS_DIR / "model_v2_final_test.json"
    if v2_path.exists():
        with open(v2_path) as f:
            v2 = json.load(f)
        v2r = v2.get("recording_level_metrics", {})
        logger.info(f"  {'Metric':<22} {'V2':>10} {'V3':>10} {'Δ':>10}")
        logger.info(f"  {'-'*52}")
        for k in ["precision", "recall", "f1", "accuracy"]:
            v2v = float(v2r.get(k, 0))
            v3v = float(rec_test.get(k, 0))
            delta = v3v - v2v
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            logger.info(f"  {k:<22} {v2v:>10.4f} {v3v:>10.4f} {arrow}{abs(delta):.4f}")


if __name__ == "__main__":
    main()
