"""
Comprehensive Multi-Class Evaluation Suite.
Evaluates both the Previous Binary Robust Model and the New Multi-Class Model
across 6 critical evaluation axes:
1. Clean Test Set (ESC-50 Fold 5 + unpolluted test clips)
2. Speaker-to-Microphone Acoustic Gap Simulation
3. Clap False Positive Rejection
4. Knock False Positive Rejection
5. Footstep Sensitivity & Recall
6. Silence, Speech & Ambient Noise Rejection

Outputs:
- Confusion Matrix (4x4) & PNG plot
- Before-vs-After Comparison Table
- reports/multiclass_evaluation_summary.json
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio, peak_normalize
from src.preprocessing import preprocess_audio_window, segment_audio
from src.feature_extraction import extract_log_mel_spectrogram
from src.feature_extraction_v3 import extract_log_mel_v3
from src.models import BaselineFootstepCNN
from src.models_multiclass import MultiClassAudioNet
from src.evaluate import simulate_speaker_microphone_channel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("EvaluateMultiClass")

CLASS_NAMES = ["other", "footstep", "clap", "knock"]


def evaluate_binary_model_on_multiclass(
    model_path: Path,
    test_df: pd.DataFrame,
    threshold: float = 0.40,
    simulate_mic: bool = False
) -> Dict[str, Any]:
    """
    Evaluates previous binary model (BaselineFootstepCNN) against multi-class test set.
    Binary model output: >= threshold -> predicted 1 (FOOTSTEP), else 0 (OTHER).
    """
    device = "cpu"
    model = BaselineFootstepCNN(in_channels=1, dropout=0.0).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true_binary = []  # 1 if footstep, 0 otherwise
    y_pred_binary = []  # 1 if predicted footstep, 0 otherwise
    cat_preds = {"footstep": [], "clap": [], "knock": [], "other": []}

    for _, row in test_df.iterrows():
        fpath = config.PROJECT_ROOT / row["relative_path"]
        if not fpath.exists():
            continue
        audio, sr = load_audio(fpath, target_sr=config.SAMPLE_RATE, mono=True)
        if simulate_mic:
            audio = simulate_speaker_microphone_channel(audio, sr=sr)

        true_label = int(row["multiclass_label"])
        cname = row["class_name"]

        windows = segment_audio(audio, window_size=config.WINDOW_SIZE, hop_size=config.HOP_SIZE, min_rms_threshold=0.0001)
        win_probs = []
        for win, _, _ in windows:
            norm_win = preprocess_audio_window(win, target_length=config.WINDOW_SIZE, apply_gain_norm=True)
            spec = extract_log_mel_spectrogram(norm_win, sr=config.SAMPLE_RATE)
            tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                prob = float(torch.sigmoid(model(tensor)).item())
            win_probs.append(prob)

        peak_p = max(win_probs) if win_probs else 0.0
        pred_is_step = 1 if peak_p >= threshold else 0

        is_true_step = 1 if true_label == 1 else 0
        y_true_binary.append(is_true_step)
        y_pred_binary.append(pred_is_step)
        cat_preds[cname].append(pred_is_step)

    # Metrics
    rec = float(np.mean(cat_preds["footstep"])) if cat_preds["footstep"] else 0.0
    clap_fpr = float(np.mean(cat_preds["clap"])) if cat_preds["clap"] else 0.0
    knock_fpr = float(np.mean(cat_preds["knock"])) if cat_preds["knock"] else 0.0
    other_fpr = float(np.mean(cat_preds["other"])) if cat_preds["other"] else 0.0

    return {
        "model_name": model_path.name,
        "mode": "simulated_speaker_mic" if simulate_mic else "clean",
        "footstep_recall": round(rec, 4),
        "clap_fpr": round(clap_fpr, 4),
        "knock_fpr": round(knock_fpr, 4),
        "other_fpr": round(other_fpr, 4),
        "total_test_files": len(y_true_binary)
    }


def evaluate_multiclass_model(
    model_path: Path,
    test_df: pd.DataFrame,
    simulate_mic: bool = False
) -> Dict[str, Any]:
    """
    Evaluates new MultiClassAudioNet against multi-class test set.
    Outputs full 4x4 confusion matrix, per-class metrics, and False Positive Rates.
    """
    device = "cpu"
    model = MultiClassAudioNet(in_channels=3, num_classes=4, dropout=0.0).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_y_true = []
    all_y_pred = []
    cat_preds = {"footstep": [], "clap": [], "knock": [], "other": []}

    for _, row in test_df.iterrows():
        fpath = config.PROJECT_ROOT / row["relative_path"]
        if not fpath.exists():
            continue
        audio, sr = load_audio(fpath, target_sr=config.SAMPLE_RATE, mono=True)
        if simulate_mic:
            audio = simulate_speaker_microphone_channel(audio, sr=sr)

        true_label = int(row["multiclass_label"])
        cname = row["class_name"]

        windows = segment_audio(audio, window_size=config.WINDOW_SIZE, hop_size=config.HOP_SIZE, min_rms_threshold=0.0001)
        win_probs = []
        for win, _, _ in windows:
            norm_win = preprocess_audio_window(win, target_length=config.WINDOW_SIZE, apply_gain_norm=True)
            feat = extract_log_mel_v3(norm_win, sr=config.SAMPLE_RATE)
            tensor = torch.from_numpy(feat).unsqueeze(0).float().to(device)
            with torch.no_grad():
                probs = model.predict_proba(tensor).cpu().numpy()[0]
            win_probs.append(probs)

        if not win_probs:
            pred_class = 0
        else:
            # Aggregate window probabilities
            avg_probs = np.mean(win_probs, axis=0)
            max_probs = np.max(win_probs, axis=0)
            # A transient sound (clap, knock, step) is flagged if its peak probability dominates
            pred_class = int(np.argmax(max_probs))

        all_y_true.append(true_label)
        all_y_pred.append(pred_class)

        # Record whether it was falsely flagged as footstep (class 1)
        flagged_as_step = 1 if pred_class == 1 else 0
        cat_preds[cname].append(flagged_as_step)

    # Per-class metrics
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=[0, 1, 2, 3], zero_division=0
    )
    acc = accuracy_score(y_true_arr, y_pred_arr)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3])

    step_rec = float(rec[1])
    clap_fpr = float(np.mean(cat_preds["clap"])) if cat_preds["clap"] else 0.0
    knock_fpr = float(np.mean(cat_preds["knock"])) if cat_preds["knock"] else 0.0
    other_fpr = float(np.mean(cat_preds["other"])) if cat_preds["other"] else 0.0

    per_class_summary = {}
    for i, c in enumerate(CLASS_NAMES):
        per_class_summary[c] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    return {
        "model_name": model_path.name,
        "mode": "simulated_speaker_mic" if simulate_mic else "clean",
        "accuracy": round(float(acc), 4),
        "footstep_recall": round(step_rec, 4),
        "clap_fpr": round(clap_fpr, 4),
        "knock_fpr": round(knock_fpr, 4),
        "other_fpr": round(other_fpr, 4),
        "per_class": per_class_summary,
        "confusion_matrix": cm.tolist()
    }


def plot_multiclass_confusion_matrix(cm: np.ndarray, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def run_full_comparison():
    manifest_p = config.PROJECT_ROOT / "dataset_multiclass_v4" / "manifest.csv"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")
    
    df = pd.read_csv(manifest_p)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(f"Loaded test split: {len(df_test)} recordings ({dict(df_test['class_name'].value_counts())})")

    binary_model_p = config.MODELS_DIR / "footstep_detector_robust.pth"
    multi_model_p = config.MODELS_DIR / "footstep_multiclass_v4.pth"

    # Evaluate Before (Binary Model)
    logger.info("Evaluating BEFORE: Binary Footstep Detector...")
    bin_clean = evaluate_binary_model_on_multiclass(binary_model_p, df_test, threshold=0.40, simulate_mic=False)
    bin_mic = evaluate_binary_model_on_multiclass(binary_model_p, df_test, threshold=0.40, simulate_mic=True)

    # Evaluate After (Multi-Class Model)
    logger.info("Evaluating AFTER: Multi-Class Audio Event Model...")
    multi_clean = evaluate_multiclass_model(multi_model_p, df_test, simulate_mic=False)
    multi_mic = evaluate_multiclass_model(multi_model_p, df_test, simulate_mic=True)

    # Save Confusion Matrix Plot
    cm_clean = np.array(multi_clean["confusion_matrix"])
    cm_plot_p = config.REPORTS_DIR / "confusion_matrix_multiclass_clean.png"
    plot_multiclass_confusion_matrix(cm_clean, cm_plot_p, "Multi-Class Confusion Matrix (Clean Test Set)")
    logger.info(f"Saved confusion matrix plot to: {cm_plot_p}")

    cm_mic = np.array(multi_mic["confusion_matrix"])
    cm_mic_plot_p = config.REPORTS_DIR / "confusion_matrix_multiclass_simulated_mic.png"
    plot_multiclass_confusion_matrix(cm_mic, cm_mic_plot_p, "Multi-Class Confusion Matrix (Speaker-to-Mic Gap)")

    # Comparison summary JSON
    summary = {
        "before_binary_model": {
            "clean": bin_clean,
            "simulated_mic": bin_mic
        },
        "after_multiclass_model": {
            "clean": multi_clean,
            "simulated_mic": multi_mic
        }
    }
    json_path = config.REPORTS_DIR / "multiclass_evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved comprehensive evaluation summary to: {json_path}")

    # Print Side-by-Side Comparison Table
    print("\n" + "=" * 95)
    print("BEFORE vs AFTER: IMPACT SOUND CLASSIFICATION & FALSE POSITIVE SUPPRESSION")
    print("=" * 95)
    print(f"{'Condition':25s} | {'Model':22s} | {'Footstep Recall':15s} | {'Clap False Alarm':16s} | {'Knock False Alarm':17s}")
    print("-" * 95)
    print(f"{'Clean Test Set':25s} | {'Before (Binary)':22s} | {bin_clean['footstep_recall']*100:13.1f}% | {bin_clean['clap_fpr']*100:14.1f}% | {bin_clean['knock_fpr']*100:15.1f}%")
    print(f"{'Clean Test Set':25s} | {'After (Multi-Class)':22s} | {multi_clean['footstep_recall']*100:13.1f}% | {multi_clean['clap_fpr']*100:14.1f}% | {multi_clean['knock_fpr']*100:15.1f}%")
    print("-" * 95)
    print(f"{'Speaker-to-Mic Gap':25s} | {'Before (Binary)':22s} | {bin_mic['footstep_recall']*100:13.1f}% | {bin_mic['clap_fpr']*100:14.1f}% | {bin_mic['knock_fpr']*100:15.1f}%")
    print(f"{'Speaker-to-Mic Gap':25s} | {'After (Multi-Class)':22s} | {multi_mic['footstep_recall']*100:13.1f}% | {multi_mic['clap_fpr']*100:14.1f}% | {multi_mic['knock_fpr']*100:15.1f}%")
    print("=" * 95 + "\n")

    print("MULTI-CLASS PER-CLASS PERFORMANCE (CLEAN TEST SET):")
    print("-" * 65)
    print(f"{'Class':12s} | {'Precision':11s} | {'Recall':10s} | {'F1-Score':10s} | {'Support':8s}")
    print("-" * 65)
    for c, metrics in multi_clean["per_class"].items():
        print(f"{c:12s} | {metrics['precision']:11.4f} | {metrics['recall']:10.4f} | {metrics['f1']:10.4f} | {metrics['support']:8d}")
    print("-" * 65 + "\n")

    return summary


if __name__ == "__main__":
    run_full_comparison()
