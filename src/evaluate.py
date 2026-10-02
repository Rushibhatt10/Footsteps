"""
Comprehensive Evaluation Suite for Footstep Detection.
Reports Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC,
evaluates on BOTH Clean Test Set and Speaker-to-Microphone Acoustic Gap,
checks for data leakage, and exports misclassified audio clips for manual acoustic inspection.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio, peak_normalize
from src.models import BaselineFootstepCNN
from src.preprocessing import preprocess_audio_window, segment_audio
from src.feature_extraction import extract_log_mel_spectrogram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Evaluator")


def simulate_speaker_microphone_channel(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    bass_cutoff: float = 250.0,
    treble_cutoff: float = 6500.0,
    reverb_delay_ms: float = 35.0,
    gain_scale: float = 0.40,
    snr_db: float = 18.0
) -> np.ndarray:
    """
    Physically grounded acoustic simulation of audio played through a small speaker
    and recorded by a laptop/phone microphone across a room.
    """
    out = audio.copy()
    nyquist = sr * 0.5

    # 1. Speaker bass roll-off (highpass)
    b_hp, a_hp = signal.butter(2, min(bass_cutoff / nyquist, 0.95), btype="highpass")
    out = signal.lfilter(b_hp, a_hp, out)

    # 2. Microphone high-frequency roll-off (lowpass)
    b_lp, a_lp = signal.butter(2, min(treble_cutoff / nyquist, 0.95), btype="lowpass")
    out = signal.lfilter(b_lp, a_lp, out)

    # 3. Room impulse response (early reflections + decay)
    delay_samples = int(reverb_delay_ms * 1e-3 * sr)
    rir_len = int(0.12 * sr)
    rir = np.zeros(rir_len, dtype=np.float32)
    rir[0] = 1.0
    if delay_samples < rir_len:
        rir[delay_samples] = 0.25
        rir[int(delay_samples * 1.7)] = 0.12
    t = np.arange(rir_len) / sr
    rir += np.exp(-t * 28.0) * np.random.normal(0, 0.08, rir_len)
    rir /= np.sum(np.abs(rir))

    out = signal.convolve(out, rir, mode="same")

    # 4. Distance / volume scaling
    out = out * gain_scale

    # 5. Ambient room noise at given SNR
    sig_power = float(np.mean(out ** 2))
    if sig_power > 1e-9:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(out))
        out = out + noise

    return out.astype(np.float32)


def load_model(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Loads BaselineFootstepCNN from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    model = BaselineFootstepCNN(in_channels=1, dropout=0.0).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path.name} (Val F1: {checkpoint.get('val_f1', 0.0):.4f})")
    return model


def check_data_leakage(manifest_path: Path) -> Dict[str, Any]:
    """Verifies strict source-aware split separation and zero leakage."""
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    train_sources = set(train_df["source_id"])
    val_sources = set(val_df["source_id"])
    test_sources = set(test_df["source_id"])

    leak_tr_val = train_sources.intersection(val_sources)
    leak_tr_test = train_sources.intersection(test_sources)
    leak_val_test = val_sources.intersection(test_sources)

    is_clean = (len(leak_tr_val) == 0 and len(leak_tr_test) == 0 and len(leak_val_test) == 0)
    result = {
        "is_leak_free": is_clean,
        "train_sources": len(train_sources),
        "val_sources": len(val_sources),
        "test_sources": len(test_sources),
        "train_files": len(train_df),
        "val_files": len(val_df),
        "test_files": len(test_df)
    }
    logger.info(f"Data leakage verification: {'PASSED (Zero Leakage)' if is_clean else 'FAILED'}")
    return result


def evaluate_split(
    model: torch.nn.Module,
    df_split: pd.DataFrame,
    mode: str = "clean",
    threshold: float = 0.40,
    device: str = "cpu",
    save_misclassified: bool = True,
    output_dir: Path = config.REPORTS_DIR
) -> Dict[str, Any]:
    """
    Evaluates model across audio files in a split.
    Reports both Window-Level and Recording-Level metrics.
    Optionally exports misclassified audio clips.
    """
    all_window_probs = []
    all_window_labels = []
    all_window_meta = []

    recording_labels = []
    recording_probs = []
    recording_preds = []

    misclassified_dir = output_dir / "misclassified_clips" / mode
    if save_misclassified:
        (misclassified_dir / "false_negatives").mkdir(parents=True, exist_ok=True)
        (misclassified_dir / "false_positives").mkdir(parents=True, exist_ok=True)

    for _, row in df_split.iterrows():
        fpath = config.PROJECT_ROOT / row["relative_path"]
        if not fpath.exists():
            continue

        audio, sr = load_audio(fpath, target_sr=config.SAMPLE_RATE, mono=True)
        label = int(row["binary_label"])
        filename = row["filename"]
        orig_cat = row["original_class"]

        # Apply speaker-mic channel if requested
        if mode == "simulated_speaker_mic":
            audio = simulate_speaker_microphone_channel(audio, sr=sr)

        windows = segment_audio(
            audio,
            window_size=config.WINDOW_SIZE,
            hop_size=config.HOP_SIZE,
            min_rms_threshold=0.0001
        )

        file_window_probs = []
        for win_audio, s_sec, e_sec in windows:
            # Shared preprocessing function
            norm_win = preprocess_audio_window(win_audio, target_length=config.WINDOW_SIZE, apply_gain_norm=True)
            spec = extract_log_mel_spectrogram(norm_win, sr=config.SAMPLE_RATE)
            tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                prob = float(torch.sigmoid(model(tensor)).item())

            file_window_probs.append(prob)
            all_window_probs.append(prob)
            all_window_labels.append(label)
            all_window_meta.append({
                "filename": filename,
                "original_class": orig_cat,
                "start_sec": s_sec,
                "end_sec": e_sec,
                "prob": prob,
                "label": label,
                "window_audio": win_audio
            })

        max_prob = max(file_window_probs) if file_window_probs else 0.0
        rec_pred = 1 if max_prob >= threshold else 0

        recording_labels.append(label)
        recording_probs.append(max_prob)
        recording_preds.append(rec_pred)

        # Save misclassified clips at recording level
        if save_misclassified:
            if label == 1 and rec_pred == 0:
                # False Negative: Missed footstep
                out_path = misclassified_dir / "false_negatives" / f"FN_{max_prob:.3f}_{filename}"
                sf.write(str(out_path), audio, config.SAMPLE_RATE)
            elif label == 0 and rec_pred == 1:
                # False Positive: Look-alike triggered footstep
                out_path = misclassified_dir / "false_positives" / f"FP_{max_prob:.3f}_{orig_cat}_{filename}"
                sf.write(str(out_path), audio, config.SAMPLE_RATE)

    # 1. Window-Level Metrics
    win_y_true = np.array(all_window_labels)
    win_y_prob = np.array(all_window_probs)
    win_y_pred = (win_y_prob >= threshold).astype(int)

    w_acc = accuracy_score(win_y_true, win_y_pred)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(win_y_true, win_y_pred, average="binary", zero_division=0)
    cm_w = confusion_matrix(win_y_true, win_y_pred, labels=[0, 1])
    tn_w, fp_w, fn_w, tp_w = cm_w.ravel()

    fpr_w = fp_w / (tn_w + fp_w) if (tn_w + fp_w) > 0 else 0.0
    fnr_w = fn_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0

    # ROC & PR AUC
    fpr_curve, tpr_curve, _ = roc_curve(win_y_true, win_y_prob)
    roc_auc = auc(fpr_curve, tpr_curve)
    prec_curve, rec_curve, _ = precision_recall_curve(win_y_true, win_y_prob)
    pr_auc = average_precision_score(win_y_true, win_y_prob)

    # 2. Recording-Level Metrics
    rec_y_true = np.array(recording_labels)
    rec_y_pred = np.array(recording_preds)
    rec_y_prob = np.array(recording_probs)

    r_acc = accuracy_score(rec_y_true, rec_y_pred)
    r_prec, r_rec, r_f1, _ = precision_recall_fscore_support(rec_y_true, rec_y_pred, average="binary", zero_division=0)
    cm_r = confusion_matrix(rec_y_true, rec_y_pred, labels=[0, 1])
    tn_r, fp_r, fn_r, tp_r = cm_r.ravel()

    # Plot Curves
    plot_evaluation_curves(
        win_y_true, win_y_prob, cm_w,
        fpr_curve, tpr_curve, roc_auc,
        prec_curve, rec_curve, pr_auc,
        mode=mode, threshold=threshold, output_dir=output_dir
    )

    results = {
        "mode": mode,
        "threshold": threshold,
        "window_metrics": {
            "accuracy": round(float(w_acc), 4),
            "precision": round(float(w_prec), 4),
            "recall": round(float(w_rec), 4),
            "f1": round(float(w_f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "fpr": round(float(fpr_w), 4),
            "fnr": round(float(fnr_w), 4),
            "tp": int(tp_w),
            "fp": int(fp_w),
            "tn": int(tn_w),
            "fn": int(fn_w),
            "total_windows": len(win_y_true)
        },
        "recording_metrics": {
            "accuracy": round(float(r_acc), 4),
            "precision": round(float(r_prec), 4),
            "recall": round(float(r_rec), 4),
            "f1": round(float(r_f1), 4),
            "tp": int(tp_r),
            "fp": int(fp_r),
            "tn": int(tn_r),
            "fn": int(fn_r),
            "total_files": len(rec_y_true)
        }
    }
    return results


def plot_evaluation_curves(
    y_true, y_prob, cm,
    fpr, tpr, roc_auc,
    precision, recall, pr_auc,
    mode: str,
    threshold: float,
    output_dir: Path
):
    """Generates and saves Confusion Matrix, ROC curve, and PR curve."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NON_FOOTSTEP", "FOOTSTEP"])
    disp.plot(cmap="Blues", ax=axes[0], values_format="d")
    axes[0].set_title(f"Confusion Matrix ({mode.upper()})\nThreshold = {threshold:.2f}")

    # 2. ROC Curve
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title(f"ROC Curve ({mode.upper()})")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    axes[2].plot(recall, precision, color="green", lw=2, label=f"PR (AP = {pr_auc:.3f})")
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel("Recall (Footstep Detection Rate)")
    axes[2].set_ylabel("Precision")
    axes[2].set_title(f"Precision-Recall Curve ({mode.upper()})")
    axes[2].legend(loc="lower left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"eval_curves_{mode}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved evaluation curves to {plot_path}")


def run_full_evaluation(
    model_path: Path,
    threshold: float = 0.40,
    compare_baseline: bool = True
):
    reports_dir = config.REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = config.PROJECT_ROOT / "dataset_v2" / "manifest.csv"
    if not manifest_path.exists():
        manifest_path = config.PROCESSED_DIR / "manifest.csv"

    # Leakage check
    leak_info = check_data_leakage(manifest_path)

    df_manifest = pd.read_csv(manifest_path)
    df_test = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)
    logger.info(f"Test split contains {len(df_test)} files ({int(sum(df_test['binary_label']==1))} Footsteps, {int(sum(df_test['binary_label']==0))} Non-Footsteps).")

    # Evaluate target model
    model = load_model(model_path, device=config.DEVICE)

    logger.info("Evaluating on CLEAN test set...")
    res_clean = evaluate_split(model, df_test, mode="clean", threshold=threshold, device=config.DEVICE, output_dir=reports_dir)

    logger.info("Evaluating on SIMULATED SPEAKER-TO-MICROPHONE test set...")
    res_mic = evaluate_split(model, df_test, mode="simulated_speaker_mic", threshold=threshold, device=config.DEVICE, output_dir=reports_dir)

    comparison_data = {
        "model": model_path.name,
        "threshold": threshold,
        "clean_test_set": res_clean,
        "simulated_speaker_mic_test_set": res_mic,
        "leakage_verification": leak_info
    }

    # If requested and baseline exists, also run baseline for before/after comparison
    baseline_path = config.BASELINE_MODEL_PATH
    if compare_baseline and baseline_path.exists() and baseline_path != model_path:
        logger.info("\nEvaluating BASELINE model for before/after comparison...")
        b_model = load_model(baseline_path, device=config.DEVICE)
        b_clean = evaluate_split(b_model, df_test, mode="clean", threshold=0.45, device=config.DEVICE, save_misclassified=False, output_dir=reports_dir)
        b_mic = evaluate_split(b_model, df_test, mode="simulated_speaker_mic", threshold=0.45, device=config.DEVICE, save_misclassified=False, output_dir=reports_dir)
        comparison_data["baseline_comparison"] = {
            "model": baseline_path.name,
            "threshold": 0.45,
            "clean_test_set": b_clean,
            "simulated_speaker_mic_test_set": b_mic
        }

    # Save summary JSON
    summary_json_path = reports_dir / "evaluation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    logger.info(f"Saved evaluation summary to {summary_json_path}")

    # Print Table
    print("\n" + "=" * 90)
    print("EVALUATION & BENCHMARK REPORT")
    print("=" * 90)
    print(f"Model Under Test: {model_path.name} (Decision Threshold = {threshold:.2f})")
    print(f"Test Split:       {len(df_test)} audio recordings (ESC-50 Fold 5, untouched)")
    print("-" * 90)
    print(f"{'Condition':30s} | {'Window Recall':13s} | {'Window Prec':11s} | {'Window F1':9s} | {'Rec Recall':10s} | {'Rec F1':7s}")
    print("-" * 90)
    print(f"{'Clean Test Set':30s} | {res_clean['window_metrics']['recall']:13.4f} | {res_clean['window_metrics']['precision']:11.4f} | {res_clean['window_metrics']['f1']:9.4f} | {res_clean['recording_metrics']['recall']:10.4f} | {res_clean['recording_metrics']['f1']:7.4f}")
    print(f"{'Speaker-to-Mic Simulation':30s} | {res_mic['window_metrics']['recall']:13.4f} | {res_mic['window_metrics']['precision']:11.4f} | {res_mic['window_metrics']['f1']:9.4f} | {res_mic['recording_metrics']['recall']:10.4f} | {res_mic['recording_metrics']['f1']:7.4f}")

    if "baseline_comparison" in comparison_data:
        b = comparison_data["baseline_comparison"]
        print("-" * 90)
        print(f"BASELINE COMPARISON ({b['model']} at threshold 0.45):")
        print(f"{'Clean Test Set (Baseline)':30s} | {b['clean_test_set']['window_metrics']['recall']:13.4f} | {b['clean_test_set']['window_metrics']['precision']:11.4f} | {b['clean_test_set']['window_metrics']['f1']:9.4f} | {b['clean_test_set']['recording_metrics']['recall']:10.4f} | {b['clean_test_set']['recording_metrics']['f1']:7.4f}")
        print(f"{'Speaker-to-Mic (Baseline)':30s} | {b['simulated_speaker_mic_test_set']['window_metrics']['recall']:13.4f} | {b['simulated_speaker_mic_test_set']['window_metrics']['precision']:11.4f} | {b['simulated_speaker_mic_test_set']['window_metrics']['f1']:9.4f} | {b['simulated_speaker_mic_test_set']['recording_metrics']['recall']:10.4f} | {b['simulated_speaker_mic_test_set']['recording_metrics']['f1']:7.4f}")

    print("=" * 90 + "\n")
    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate footstep detection model")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--threshold", "-t", type=float, default=0.40, help="Decision threshold")
    parser.add_argument("--no-compare", action="store_true", help="Do not compare against baseline model")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else config.MODELS_DIR / "footstep_detector_robust.pth"
    if not model_path.exists():
        model_path = config.MODELS_DIR / "model_v2_balanced_sampler.pth"

    run_full_evaluation(model_path=model_path, threshold=args.threshold, compare_baseline=not args.no_compare)


if __name__ == "__main__":
    main()
