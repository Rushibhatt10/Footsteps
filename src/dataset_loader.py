"""
Dataset Loader for Footstep Detection.
Extracts audio windows, applies train-only augmentations, computes Log-Mel Spectrograms,
and provides balanced sampling for training.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src import config
from src.audio_utils import load_audio
from src.preprocessing import segment_audio, pad_or_trim, preprocess_audio_window
from src.feature_extraction import extract_log_mel_spectrogram
from src.augment import apply_augmentations


class AudioWindowDataset(Dataset):
    """
    Dataset that extracts 1.5s audio windows from standardized audio clips.
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

        # Filter by split
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        
        # Build window index
        self.samples: List[Dict[str, Any]] = []
        self._build_samples()

    def _build_samples(self):
        """Indexes all windows from all audio files for this split."""
        for _, row in self.df.iterrows():
            filepath = config.PROJECT_ROOT / row["relative_path"]
            if not filepath.exists():
                continue

            try:
                audio, _ = load_audio(filepath, target_sr=config.SAMPLE_RATE, mono=config.MONO)
            except Exception:
                continue

            binary_label = int(row["binary_label"])
            orig_class = row["original_class"]
            filename = row["filename"]
            source_id = row.get("source_id", filename)

            # For footsteps in train split, use smaller hop (0.25s) to generate dense positive windows
            curr_hop = self.hop_size // 2 if (self.is_train and binary_label == 1) else self.hop_size

            windows = segment_audio(
                audio,
                window_size=self.window_size,
                hop_size=curr_hop,
                min_rms_threshold=0.0005
            )

            for win_audio, start_sec, end_sec in windows:
                self.samples.append({
                    "audio": win_audio,
                    "label": binary_label,
                    "original_class": orig_class,
                    "filename": filename,
                    "source_id": source_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]
        audio = sample["audio"].copy()

        # Apply augmentation only during training if enabled
        if self.enable_augmentation:
            audio = apply_augmentations(audio, sr=config.SAMPLE_RATE)

        # Common standardized preprocessing (DC removal, soft-gain norm, exact length)
        audio = preprocess_audio_window(audio, target_length=self.window_size, apply_gain_norm=True)
        spec = extract_log_mel_spectrogram(audio, sr=config.SAMPLE_RATE)

        # Shape: [1, n_mels, time_frames]
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()
        label_tensor = torch.tensor(sample["label"], dtype=torch.float32)

        meta = {
            "filename": sample["filename"],
            "source_id": sample.get("source_id", sample["filename"]),
            "original_class": sample["original_class"],
            "start_sec": sample["start_sec"],
            "end_sec": sample["end_sec"]
        }

        return spec_tensor, label_tensor, meta

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Creates a WeightedRandomSampler to balance positive and negative classes evenly in training batches.
        """
        labels = [s["label"] for s in self.samples]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count

        if pos_count == 0 or neg_count == 0:
            weights = [1.0] * len(labels)
        else:
            weight_pos = 1.0 / pos_count
            weight_neg = 1.0 / neg_count
            weights = [weight_pos if l == 1 else weight_neg for l in labels]

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    manifest_path: Optional[Path | str] = None,
    use_balanced_sampler: bool = False,
    enable_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Creates train, validation, and test dataloaders from the specified manifest.
    Defaults to dataset_v2/manifest.csv if present.
    """
    if manifest_path is None:
        v2_path = config.PROJECT_ROOT / "dataset_v2" / "manifest.csv"
        manifest_path = v2_path if v2_path.exists() else config.PROCESSED_DIR / "manifest.csv"

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)

    train_ds = AudioWindowDataset(df_manifest, split="train", is_train=True, enable_augmentation=enable_augmentation)
    val_ds = AudioWindowDataset(df_manifest, split="validation", is_train=False, enable_augmentation=False)
    test_ds = AudioWindowDataset(df_manifest, split="test", is_train=False, enable_augmentation=False)

    if use_balanced_sampler:
        sampler = train_ds.get_sampler()
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, df_manifest
