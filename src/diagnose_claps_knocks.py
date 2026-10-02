"""
Diagnostic Script: Evaluates current footstep model specifically on Claps and Knocks.
Measures False Positive Rates (FPR), exports misclassified audio clips,
and plots acoustic features (waveforms, mel-spectrograms, decay profiles, spectral centroids).
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio, peak_normalize
from src.models import BaselineFootstepCNN
from src.preprocessing import preprocess_audio_window, segment_audio
from src.feature_extraction import extract_log_mel_spectrogram


def compute_acoustic_features(audio: np.ndarray, sr: int = config.SAMPLE_RATE):
    """
    Computes measurable acoustic discriminators:
    - Attack time (ms to reach peak)
    - Decay time (ms from peak to -20dB)
    - Low-frequency energy ratio (< 300 Hz)
    - High-frequency energy ratio (> 2000 Hz)
    - Spectral centroid (Hz)
    - Spectral flatness
    """
    eps = 1e-9
    norm = audio / (np.max(np.abs(audio)) + eps)
    
    # 1. Temporal envelope
    analytic_signal = signal.hilbert(norm)
    envelope = np.abs(analytic_signal)
    peak_idx = int(np.argmax(envelope))
    peak_val = envelope[peak_idx]
    
    # Attack time
    attack_samples = peak_idx
    attack_ms = (attack_samples / sr) * 1000.0
    
    # Decay time: time from peak until envelope falls below 10% (-20dB)
    decay_threshold = peak_val * 0.10
    decay_samples = len(envelope) - peak_idx
    for i in range(peak_idx, len(envelope)):
        if envelope[i] < decay_threshold:
            decay_samples = i - peak_idx
            break
    decay_ms = (decay_samples / sr) * 1000.0
    
    # 2. Spectral metrics
    freqs, psd = signal.welch(norm, fs=sr, nperseg=min(len(norm), 1024))
    total_power = np.sum(psd) + eps
    
    low_band = (freqs <= 300.0)
    high_band = (freqs >= 2000.0)
    
    low_ratio = np.sum(psd[low_band]) / total_power
    high_ratio = np.sum(psd[high_band]) / total_power
    
    # Spectral centroid
    spectral_centroid = np.sum(freqs * psd) / total_power
    
    # Spectral flatness (geometric mean / arithmetic mean)
    geo_mean = np.exp(np.mean(np.log(psd + eps)))
    arith_mean = np.mean(psd) + eps
    flatness = float(geo_mean / arith_mean)
    
    return {
        "attack_ms": round(float(attack_ms), 1),
        "decay_ms": round(float(decay_ms), 1),
        "low_energy_ratio": round(float(low_ratio), 4),
        "high_energy_ratio": round(float(high_ratio), 4),
        "spectral_centroid_hz": round(float(spectral_centroid), 1),
        "spectral_flatness": round(float(flatness), 5)
    }


def run_diagnostics(
    model_path: Path,
    output_dir: Path = config.REPORTS_DIR / "diagnostics"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    misclass_dir = output_dir / "misclassified_claps_knocks"
    (misclass_dir / "false_positive_claps").mkdir(parents=True, exist_ok=True)
    (misclass_dir / "false_positive_knocks").mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    device = config.DEVICE
    model = BaselineFootstepCNN(in_channels=1, dropout=0.0).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded model: {model_path.name}")

    # 2. Gather audio samples from ESC-50 and dataset_v2
    # ESC-50 meta
    esc_meta = pd.read_csv(config.ESC50_META_FILE)
    esc_audio_dir = config.ESC50_AUDIO_DIR

    # Target test categories
    test_files = []
    
    # Clapping (40 files)
    claps = esc_meta[esc_meta["category"] == "clapping"]
    for _, row in claps.iterrows():
        p = esc_audio_dir / row["filename"]
        if p.exists():
            test_files.append({"path": p, "true_cat": "clapping", "is_footstep": 0, "fold": row["fold"]})

    # Door wood knock (40 files)
    knocks = esc_meta[esc_meta["category"] == "door_wood_knock"]
    for _, row in knocks.iterrows():
        p = esc_audio_dir / row["filename"]
        if p.exists():
            test_files.append({"path": p, "true_cat": "knocking", "is_footstep": 0, "fold": row["fold"]})

    # Footsteps (40 files)
    steps = esc_meta[esc_meta["category"] == "footsteps"]
    for _, row in steps.iterrows():
        p = esc_audio_dir / row["filename"]
        if p.exists():
            test_files.append({"path": p, "true_cat": "footstep", "is_footstep": 1, "fold": row["fold"]})

    # Also add other negatives for context (e.g. ambient, speech, typing)
    others = esc_meta[esc_meta["category"].isin(["keyboard_typing", "mouse_click", "rain", "wind"])]
    for _, row in others.iterrows():
        p = esc_audio_dir / row["filename"]
        if p.exists():
            test_files.append({"path": p, "true_cat": "other_ambient", "is_footstep": 0, "fold": row["fold"]})

    print(f"[INFO] Total diagnostic test files gathered: {len(test_files)}")

    # 3. Inference on each file using the sliding-window engine
    records = []
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.65]

    for item in test_files:
        fpath = item["path"]
        true_cat = item["true_cat"]
        is_step = item["is_footstep"]
        fold = item["fold"]

        audio, sr = load_audio(fpath, target_sr=config.SAMPLE_RATE, mono=True)
        windows = segment_audio(audio, window_size=config.WINDOW_SIZE, hop_size=config.HOP_SIZE, min_rms_threshold=0.0001)

        win_probs = []
        for win, s_sec, e_sec in windows:
            norm_win = preprocess_audio_window(win, target_length=config.WINDOW_SIZE, apply_gain_norm=True)
            spec = extract_log_mel_spectrogram(norm_win, sr=config.SAMPLE_RATE)
            tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                prob = float(torch.sigmoid(model(tensor)).item())
            win_probs.append(prob)

        max_p = max(win_probs) if win_probs else 0.0
        mean_p = float(np.mean(win_probs)) if win_probs else 0.0

        rec = {
            "filename": fpath.name,
            "category": true_cat,
            "fold": fold,
            "is_footstep": is_step,
            "peak_probability": max_p,
            "mean_probability": mean_p,
            "num_windows": len(win_probs)
        }
        for th in thresholds:
            rec[f"pred_{int(th*100)}"] = 1 if max_p >= th else 0
        records.append(rec)

        # Save misclassified clips at threshold 0.40
        if is_step == 0 and max_p >= 0.40:
            if true_cat == "clapping":
                sf.write(str(misclass_dir / "false_positive_claps" / f"FP_clap_{max_p:.3f}_{fpath.name}"), audio, config.SAMPLE_RATE)
            elif true_cat == "knocking":
                sf.write(str(misclass_dir / "false_positive_knocks" / f"FP_knock_{max_p:.3f}_{fpath.name}"), audio, config.SAMPLE_RATE)

    df_diag = pd.DataFrame(records)
    csv_out = output_dir / "clap_knock_diagnosis.csv"
    df_diag.to_csv(csv_out, index=False)
    print(f"[INFO] Diagnosis CSV saved to {csv_out}")

    # 4. Report per-category False Positive Rates
    print("\n" + "=" * 80)
    print("CURRENT MODEL FALSE POSITIVE RATE ANALYSIS ON TRANSIENT IMPACTS")
    print("=" * 80)
    print(f"{'Category':20s} | {'Count':5s} | {'FPR @ 0.35':10s} | {'FPR @ 0.40':10s} | {'FPR @ 0.45':10s} | {'FPR @ 0.50':10s} | {'Avg Peak Prob':13s}")
    print("-" * 80)
    
    for cat in ["clapping", "knocking", "other_ambient"]:
        sub = df_diag[df_diag["category"] == cat]
        cnt = len(sub)
        fpr35 = (sub["pred_35"] == 1).mean() * 100
        fpr40 = (sub["pred_40"] == 1).mean() * 100
        fpr45 = (sub["pred_45"] == 1).mean() * 100
        fpr50 = (sub["pred_50"] == 1).mean() * 100
        avg_p = sub["peak_probability"].mean()
        print(f"{cat:20s} | {cnt:5d} | {fpr35:9.1f}% | {fpr40:9.1f}% | {fpr45:9.1f}% | {fpr50:9.1f}% | {avg_p:13.4f}")

    step_sub = df_diag[df_diag["category"] == "footstep"]
    rec40 = (step_sub["pred_40"] == 1).mean() * 100
    rec50 = (step_sub["pred_50"] == 1).mean() * 100
    print(f"{'footstep (Recall)':20s} | {len(step_sub):5d} | {'--':10s} | {rec40:9.1f}% | {'--':10s} | {rec50:9.1f}% | {step_sub['peak_probability'].mean():13.4f}")
    print("=" * 80 + "\n")

    # 5. Measure and Plot Acoustic Differences (Footstep vs Clap vs Knock)
    plot_acoustic_comparison(output_dir)

    return df_diag


def plot_acoustic_comparison(output_dir: Path):
    """
    Plots waveforms, spectrograms, and envelope decay curves for:
    1. Footstep (wood / shoe impact)
    2. Clapping (hand clap)
    3. Knocking (door wood knock)
    And prints table of measurable differences.
    """
    esc_meta = pd.read_csv(config.ESC50_META_FILE)
    esc_audio = config.ESC50_AUDIO_DIR

    # Pick representative examples from Fold 5 (test set)
    step_row = esc_meta[(esc_meta["category"] == "footsteps") & (esc_meta["fold"] == 5)].iloc[0]
    clap_row = esc_meta[(esc_meta["category"] == "clapping") & (esc_meta["fold"] == 5)].iloc[0]
    knock_row = esc_meta[(esc_meta["category"] == "door_wood_knock") & (esc_meta["fold"] == 5)].iloc[0]

    sounds = [
        ("Footstep", esc_audio / step_row["filename"], "#2b5c8f"),
        ("Clapping", esc_audio / clap_row["filename"], "#d95f02"),
        ("Door Knock", esc_audio / knock_row["filename"], "#7570b3")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    feature_table = []

    for col_idx, (name, path, color) in enumerate(sounds):
        audio, sr = load_audio(path, target_sr=config.SAMPLE_RATE, mono=True)
        # Take first 1.5s window
        win = audio[:int(config.WINDOW_DURATION_SEC * sr)]
        feats = compute_acoustic_features(win, sr=sr)
        feats["name"] = name
        feats["filename"] = path.name
        feature_table.append(feats)

        time_axis = np.linspace(0, len(win) / sr, len(win))

        # 1. Waveform + Hilbert Envelope
        ax_w = axes[0, col_idx]
        ax_w.plot(time_axis, win, alpha=0.5, color=color, label="Waveform")
        env = np.abs(signal.hilbert(win))
        ax_w.plot(time_axis, env, color="black", lw=1.2, label="Envelope")
        ax_w.set_title(f"{name}\nAttack: {feats['attack_ms']}ms | Decay: {feats['decay_ms']}ms", fontsize=11, fontweight="bold")
        ax_w.set_xlabel("Time (s)")
        ax_w.set_ylabel("Amplitude")
        ax_w.set_ylim([-1.05, 1.05])
        ax_w.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_w.legend(loc="upper right", fontsize=8)

        # 2. Log-Mel Spectrogram
        ax_s = axes[1, col_idx]
        norm_w = preprocess_audio_window(win, target_length=len(win), apply_gain_norm=True)
        spec = extract_log_mel_spectrogram(norm_w, sr=sr)
        im = ax_s.imshow(spec, aspect="auto", origin="lower", cmap="magma", extent=[0, len(win)/sr, 20, 8000])
        ax_s.set_title(f"Centroid: {feats['spectral_centroid_hz']:.0f} Hz\nLow (<300Hz): {feats['low_energy_ratio']*100:.1f}% | High (>2kHz): {feats['high_energy_ratio']*100:.1f}%", fontsize=10)
        ax_s.set_xlabel("Time (s)")
        ax_s.set_ylabel("Freq (Hz)")

        # 3. Power Spectral Density (PSD)
        ax_p = axes[2, col_idx]
        freqs, psd = signal.welch(norm_w, fs=sr, nperseg=1024)
        ax_p.semilogy(freqs, psd, color=color, lw=1.8)
        ax_p.axvline(300, color="red", linestyle="--", alpha=0.7, label="300 Hz Cutoff")
        ax_p.axvline(2000, color="blue", linestyle=":", alpha=0.7, label="2000 Hz")
        ax_p.set_title(f"PSD & Energy Distribution\nFlatness: {feats['spectral_flatness']:.4f}", fontsize=10)
        ax_p.set_xlabel("Frequency (Hz)")
        ax_p.set_ylabel("PSD")
        ax_p.set_xlim([0, 8000])
        ax_p.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_p.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "acoustic_comparison_footstep_clap_knock.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"[INFO] Acoustic comparison plot saved to: {plot_path}")

    # Print summary table of features
    df_feat = pd.DataFrame(feature_table)
    feat_csv = output_dir / "acoustic_feature_metrics.csv"
    df_feat.to_csv(feat_csv, index=False)

    print("\n" + "=" * 80)
    print("MEASURABLE ACOUSTIC DISCRIMINATORS: FOOTSTEP vs CLAP vs KNOCK")
    print("=" * 80)
    print(f"{'Sound Class':12s} | {'Decay (ms)':10s} | {'Low <300Hz (%)':14s} | {'High >2kHz (%)':14s} | {'Centroid (Hz)':13s} | {'Flatness':8s}")
    print("-" * 80)
    for f in feature_table:
        print(f"{f['name']:12s} | {f['decay_ms']:10.1f} | {f['low_energy_ratio']*100:14.1f}% | {f['high_energy_ratio']*100:14.1f}% | {f['spectral_centroid_hz']:13.0f} | {f['spectral_flatness']:8.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    model_file = config.MODELS_DIR / "footstep_detector_robust.pth"
    if not model_file.exists():
        model_file = config.MODELS_DIR / "model_v2_balanced_sampler.pth"
    run_diagnostics(model_file)
