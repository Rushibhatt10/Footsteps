"""
Multi-Class Audio Dataset & DataLoader for Footstep vs Clap vs Knock vs Other.
Includes:
1. Audio window segmentation (1.5s windows with 0.25s hop for minority classes)
2. Domain-Gap and Acoustic Augmentations applied uniformly across ALL classes
3. 3-Channel Log-Mel + Delta + Delta-Delta extraction
4. 4-Way Class-Balanced WeightedRandomSampler to prevent majority-class bias
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.audio_utils import load_audio, peak_normalize
from src.preprocessing import segment_audio, preprocess_audio_window
from src.feature_extraction_v3 import extract_log_mel_v3
from src.augment import apply_augmentations


def apply_multiclass_augmentations(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    prob: float = 0.65
) -> np.ndarray:
    """
    Applies diverse acoustic domain augmentations:
    - Pitch shifting
    - Time shifting
    - Gain / volume jitter
    - Ambient room noise injection
    - Lowpass / Highpass acoustic roll-off (simulating speaker/mic channel)
    - Synthetic room reverb
    """
    if np.random.rand() > prob:
        return audio

    out = audio.copy()

    # 1. Gain jitter (0.4x to 1.3x)
    if np.random.rand() < 0.6:
        gain = np.random.uniform(0.4, 1.3)
        out = out * gain

    # 2. Time Shift (-0.2s to +0.2s)
    if np.random.rand() < 0.5:
        max_shift = int(0.2 * sr)
        shift = np.random.randint(-max_shift, max_shift)
        out = np.roll(out, shift)
        if shift > 0:
            out[:shift] = 0
        elif shift < 0:
            out[shift:] = 0

    # 3. Additive noise (SNR 12 to 28 dB)
    if np.random.rand() < 0.5:
        sig_power = float(np.mean(out ** 2))
        if sig_power > 1e-9:
            target_snr = np.random.uniform(12.0, 28.0)
            noise_power = sig_power / (10.0 ** (target_snr / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), len(out))
            out = out + noise

    # 4. Acoustic Filter (lowpass or highpass roll-off)
    if np.random.rand() < 0.4:
        nyquist = sr * 0.5
        if np.random.rand() < 0.5:
            # Highpass (speaker bass roll-off 150-350 Hz)
            cutoff = np.random.uniform(150.0, 350.0) / nyquist
            b, a = signal.butter(2, min(cutoff, 0.95), btype="highpass")
            out = signal.lfilter(b, a, out)
        else:
            # Lowpass (mic treble roll-off 4500-7500 Hz)
            cutoff = np.random.uniform(4500.0, 7500.0) / nyquist
            b, a = signal.butter(2, min(cutoff, 0.95), btype="lowpass")
            out = signal.lfilter(b, a, out)

    # 5. Room impulse response / Reverb
    if np.random.rand() < 0.35:
        rir_len = int(0.08 * sr)
        delay_samples = int(np.random.uniform(15, 45) * 1e-3 * sr)
        rir = np.zeros(rir_len, dtype=np.float32)
        rir[0] = 1.0
        if delay_samples < rir_len:
            rir[delay_samples] = np.random.uniform(0.15, 0.35)
        t = np.arange(rir_len) / sr
        rir += np.exp(-t * 30.0) * np.random.normal(0, 0.05, rir_len)
        rir /= (np.sum(np.abs(rir)) + 1e-9)
        out = signal.convolve(out, rir, mode="same")

    return out.astype(np.float32)


class MultiClassAudioWindowDataset(Dataset):
    """
    Dataset extracting 1.5s 3-channel (mel+delta+delta²) windows for 4 classes:
    0: OTHER
    1: FOOTSTEP
    2: CLAP
    3: KNOCK
    """
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str = "train",
        window_size: int = config.WINDOW_SIZE,
        hop_size: int = config.HOP_SIZE,
        is_train: bool = True,
        enable_augmentation: bool = True
    ):
        self.split = split
        self.is_train = is_train
        self.enable_augmentation = enable_augmentation and is_train
        self.window_size = window_size
        self.hop_size = hop_size

        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.samples: List[Dict[str, Any]] = []
        self._build_samples()

    def _build_samples(self):
        """Builds all window samples for this split."""
        for _, row in self.df.iterrows():
            fpath = config.PROJECT_ROOT / row["relative_path"]
            if not fpath.exists():
                continue

            try:
                audio, _ = load_audio(fpath, target_sr=config.SAMPLE_RATE, mono=True)
            except Exception:
                continue

            label = int(row["multiclass_label"])
            cname = row["class_name"]
            orig_cat = row["original_class"]
            fn = row["filename"]
            src_id = row.get("source_id", fn)

            # In training, use dense 0.25s hop for minority classes (footstep, clap, knock)
            # Use standard 0.5s hop for 'other'
            if self.is_train:
                curr_hop = self.hop_size // 2 if label in [1, 2, 3] else self.hop_size
            else:
                curr_hop = self.hop_size

            windows = segment_audio(
                audio,
                window_size=self.window_size,
                hop_size=curr_hop,
                min_rms_threshold=0.0001
            )

            for win_audio, s_sec, e_sec in windows:
                self.samples.append({
                    "audio": win_audio,
                    "label": label,
                    "class_name": cname,
                    "original_class": orig_cat,
                    "filename": fn,
                    "source_id": src_id,
                    "start_sec": s_sec,
                    "end_sec": e_sec
                })

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> List[int]:
        return [s["label"] for s in self.samples]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        sample = self.samples[idx]
        audio = sample["audio"].copy()
        label = sample["label"]

        if self.enable_augmentation:
            audio = apply_multiclass_augmentations(audio, sr=config.SAMPLE_RATE, prob=0.65)

        # Standardize window with gain normalization
        norm_audio = preprocess_audio_window(audio, target_length=self.window_size, apply_gain_norm=True)

        # Extract 3-channel feature: (3, 80, 47)
        feat = extract_log_mel_v3(norm_audio, sr=config.SAMPLE_RATE)
        tensor = torch.from_numpy(feat).float()

        meta = {
            "filename": sample["filename"],
            "class_name": sample["class_name"],
            "original_class": sample["original_class"],
            "start_sec": sample["start_sec"],
            "end_sec": sample["end_sec"]
        }
        return tensor, label, meta


def create_multiclass_dataloaders(
    manifest_path: Path = config.PROJECT_ROOT / "dataset_multiclass_v4" / "manifest.csv",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    df = pd.read_csv(manifest_path)

    train_ds = MultiClassAudioWindowDataset(df, split="train", is_train=True, enable_augmentation=True)
    val_ds = MultiClassAudioWindowDataset(df, split="validation", is_train=False, enable_augmentation=False)
    test_ds = MultiClassAudioWindowDataset(df, split="test", is_train=False, enable_augmentation=False)

    train_labels = train_ds.get_labels()
    class_counts = np.bincount(train_labels, minlength=4)
    total_samples = len(train_labels)

    # Class weights for loss & sampling: inversely proportional to frequency
    class_weights = total_samples / (4.0 * np.maximum(class_counts, 1).astype(np.float32))
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    dataset_info = {
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "train_class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist()
    }

    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    t_ldr, v_ldr, te_ldr, info = create_multiclass_dataloaders()
    print("Multi-Class DataLoader initialized successfully:")
    print("Dataset info:", info)
    for specs, labels, meta in t_ldr:
        print("Batch specs shape:", specs.shape)
        print("Batch labels shape:", labels.shape)
        print("Batch label distribution:", torch.bincount(labels, minlength=4).numpy())
        break
