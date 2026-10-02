"""
Phase 10: Dataset V3 Builder.

Strategy:
- Preserves the untouched final test set (ESC-50 Fold 5) EXACTLY as in Dataset V2.
- Copies all existing Dataset V2 train + validation splits.
- Extracts windowed hard-negative clips from the exact categories that caused
  false positives in Model V2:
    keyboard_typing, water_drops, door_wood_knock, glass_breaking, chainsaw
- Generates augmented footstep variants (pitch shift, time stretch, noise) to
  increase footstep training diversity from ~61 to ~120+ window-level positives.
- Applies SpecAugment at DataLoader level (NOT here — augmentation is in the loader).

GUARANTEES:
- Untouched test set: 100% identical to Dataset V2 test split.
- Zero source-ID leakage between train / validation / test.
- Hard negatives from Fold 5 files are NEVER added (test fold protected).
"""

import sys
import shutil
import random
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio, peak_normalize
from src.preprocessing import pad_or_trim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetV3Builder")

# =====================================================================
# Paths
# =====================================================================
DATASET_V2_DIR = config.PROJECT_ROOT / "dataset_v2"
DATASET_V3_DIR = config.PROJECT_ROOT / "dataset_v3"
V3_TRAIN_DIR   = DATASET_V3_DIR / "train"
V3_VAL_DIR     = DATASET_V3_DIR / "validation"
V3_TEST_DIR    = DATASET_V3_DIR / "test"
V3_MANIFEST    = DATASET_V3_DIR / "manifest.csv"

# Hard-negative source categories and how many windowed clips to extract per file
# (only from Folds 1-4 files in ESC-50 — Fold 5 is protected test set)
HARD_NEGATIVE_CATEGORIES: Dict[str, int] = {
    "keyboard_typing": 6,    # 3 FPs with 0.99 confidence
    "water_drops":     6,    # 6 window-level FPs
    "door_wood_knock": 5,    # 3 FPs at ~0.90
    "glass_breaking":  5,    # 1 FP but very high confidence
    "chainsaw":        4,    # 1 FP — industrial rhythm pattern
}

# Protected fold (test set — never touch)
PROTECTED_FOLD = 5

# Footstep augmentation variants to generate per recording
FOOTSTEP_AUG_VARIANTS = 3   # pitch-up, pitch-down, time-stretch
AUG_PITCH_STEPS = [-2, +2, -3]   # semitones


def _make_dirs():
    for d in [V3_TRAIN_DIR / "FOOTSTEP", V3_TRAIN_DIR / "NON_FOOTSTEP",
              V3_VAL_DIR   / "FOOTSTEP", V3_VAL_DIR   / "NON_FOOTSTEP",
              V3_TEST_DIR  / "FOOTSTEP", V3_TEST_DIR  / "NON_FOOTSTEP"]:
        d.mkdir(parents=True, exist_ok=True)


def _extract_hard_negative_windows(
    audio: np.ndarray,
    sr: int,
    window_size: int,
    hop_size: int,
    n_clips: int,
    rng: random.Random,
) -> List[np.ndarray]:
    """
    Extracts up to n_clips non-overlapping 1.5s windows from a hard-negative audio file.
    Selects windows with the highest RMS energy (most challenging segments).
    """
    windows = []
    starts = list(range(0, len(audio) - window_size + 1, hop_size))
    if not starts:
        # File shorter than window — pad and return once
        windows.append(pad_or_trim(audio, window_size))
        return windows

    # Score windows by RMS
    scored = []
    for s in starts:
        win = audio[s:s + window_size]
        rms = float(np.sqrt(np.mean(win ** 2)))
        scored.append((rms, s))

    scored.sort(reverse=True)  # Highest-energy first = hardest negatives

    # Take top-n_clips (ensure non-overlapping by spacing)
    chosen_starts = []
    for rms, s in scored:
        # Ensure no overlap with already chosen windows
        overlap = any(abs(s - cs) < window_size // 2 for cs in chosen_starts)
        if not overlap:
            chosen_starts.append(s)
            if len(chosen_starts) >= n_clips:
                break

    for s in chosen_starts:
        win = audio[s:s + window_size]
        win = pad_or_trim(win, window_size)
        win = peak_normalize(win, target_peak=0.90)
        windows.append(win)

    return windows


def _augment_footstep(
    audio: np.ndarray,
    sr: int,
    window_size: int,
    variant_idx: int,
) -> np.ndarray:
    """Generates one augmented footstep variant."""
    aug = audio.copy()
    if variant_idx < len(AUG_PITCH_STEPS):
        n_steps = AUG_PITCH_STEPS[variant_idx]
        try:
            aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=n_steps)
        except Exception:
            pass
    else:
        # Time-stretch
        rate = random.uniform(0.85, 1.15)
        try:
            aug = librosa.effects.time_stretch(aug, rate=rate)
        except Exception:
            pass

    aug = pad_or_trim(aug, window_size)
    aug = peak_normalize(aug, target_peak=0.90)
    return aug.astype(np.float32)


def build_dataset_v3():
    logger.info("=" * 65)
    logger.info("PHASE 10: CONSTRUCTING DATASET V3")
    logger.info("  Hard negatives + augmented footsteps + V2 base")
    logger.info("=" * 65)

    _make_dirs()
    rng = random.Random(config.RANDOM_SEED)
    records: List[Dict] = []
    seen_source_ids: Dict[str, str] = {}  # source_id → split

    # -----------------------------------------------------------------
    # STEP 1: Copy V2 manifest entries directly (test is 100% preserved)
    # -----------------------------------------------------------------
    logger.info("Step 1: Copying Dataset V2 files into dataset_v3/ ...")
    df_v2 = pd.read_csv(DATASET_V2_DIR / "manifest.csv")

    split_dir_map = {"train": V3_TRAIN_DIR, "validation": V3_VAL_DIR, "test": V3_TEST_DIR}

    for _, row in tqdm(df_v2.iterrows(), total=len(df_v2), desc="Copying V2"):
        src = config.PROJECT_ROOT / row["relative_path"]
        split = row["split"]
        binary_class = row["binary_class"]
        dest_dir = split_dir_map[split] / binary_class
        dest_file = dest_dir / row["filename"]

        if src.exists():
            shutil.copy2(src, dest_file)

        source_id = row["source_id"]
        if source_id in seen_source_ids:
            if seen_source_ids[source_id] != split:
                raise ValueError(f"Leakage in V2 copy! {source_id}: {seen_source_ids[source_id]} vs {split}")
        else:
            seen_source_ids[source_id] = split

        records.append({
            "filename":       row["filename"],
            "source":         row["source"],
            "source_id":      source_id,
            "original_file":  row["original_file"],
            "original_class": row["original_class"],
            "binary_class":   binary_class,
            "binary_label":   int(row["binary_label"]),
            "split":          split,
            "fold":           int(row["fold"]),
            "duration":       float(row["duration"]),
            "sample_rate":    int(row["sample_rate"]),
            "relative_path":  str(dest_file.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
            "v3_augmented":   False,
        })

    logger.info(f"  Copied {len(df_v2)} files from Dataset V2.")

    # -----------------------------------------------------------------
    # STEP 2: Extract targeted hard-negative windows from ESC-50 folds 1-4
    # -----------------------------------------------------------------
    logger.info("Step 2: Mining targeted hard-negative windows from ESC-50 ...")
    meta = pd.read_csv(config.ESC50_META_FILE)
    hn_added = 0

    for category, n_clips in HARD_NEGATIVE_CATEGORIES.items():
        cat_files = meta[
            (meta["category"] == category) &
            (meta["fold"] != PROTECTED_FOLD)
        ]
        # Prioritise Fold 3 files (not in V2 test, not in val if possible)
        cat_files = cat_files.sort_values("fold", ascending=False)

        for _, frow in cat_files.iterrows():
            esc_fname = frow["filename"]
            esc_audio_path = config.ESC50_AUDIO_DIR / esc_fname
            source_id = f"esc50_{esc_fname.split('-')[1]}"
            fold = int(frow["fold"])

            # Determine split from original V2 assignment
            if source_id in seen_source_ids:
                split = seen_source_ids[source_id]
                if split == "test":
                    continue  # Never touch test sources
            else:
                # Not in V2 → assign to train (fold ≤ 3) or validation (fold 4)
                split = "train" if fold <= 3 else "validation"
                seen_source_ids[source_id] = split

            if not esc_audio_path.exists():
                continue

            try:
                audio, sr = load_audio(esc_audio_path, target_sr=config.SAMPLE_RATE, mono=True)
            except Exception as e:
                logger.warning(f"  Could not load {esc_fname}: {e}")
                continue

            windows = _extract_hard_negative_windows(
                audio, sr,
                window_size=config.WINDOW_SIZE,
                hop_size=config.HOP_SIZE,
                n_clips=n_clips,
                rng=rng,
            )

            for i, win in enumerate(windows):
                out_fname = f"v3_hn_{category}_{esc_fname.replace('.wav', '')}_{i:02d}.wav"
                dest_dir = (V3_TRAIN_DIR if split == "train" else V3_VAL_DIR) / "NON_FOOTSTEP"
                dest_file = dest_dir / out_fname

                sf.write(str(dest_file), win, config.SAMPLE_RATE, subtype="PCM_16")

                hn_clip_source_id = f"hn_v3_{source_id}_clip{i}"
                seen_source_ids[hn_clip_source_id] = split

                records.append({
                    "filename":       out_fname,
                    "source":         "esc50_hard_negative",
                    "source_id":      hn_clip_source_id,
                    "original_file":  esc_fname,
                    "original_class": category,
                    "binary_class":   "NON_FOOTSTEP",
                    "binary_label":   0,
                    "split":          split,
                    "fold":           fold,
                    "duration":       config.WINDOW_DURATION_SEC,
                    "sample_rate":    config.SAMPLE_RATE,
                    "relative_path":  str(dest_file.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
                    "v3_augmented":   False,
                })
                hn_added += 1

    logger.info(f"  Added {hn_added} targeted hard-negative clips.")

    # -----------------------------------------------------------------
    # STEP 3: Augment footstep training recordings
    # -----------------------------------------------------------------
    logger.info("Step 3: Generating augmented footstep variants for train split ...")
    aug_added = 0

    # Only augment FOOTSTEP files from train split
    footstep_train = [r for r in records
                      if r["binary_class"] == "FOOTSTEP" and r["split"] == "train"]

    for rec in tqdm(footstep_train, desc="Augmenting footsteps"):
        orig_file = config.PROJECT_ROOT / rec["relative_path"]
        if not orig_file.exists():
            continue

        try:
            audio, _ = load_audio(orig_file, target_sr=config.SAMPLE_RATE, mono=True)
        except Exception:
            continue

        audio = pad_or_trim(audio, config.WINDOW_SIZE)

        for variant_idx in range(FOOTSTEP_AUG_VARIANTS):
            aug_audio = _augment_footstep(
                audio, config.SAMPLE_RATE, config.WINDOW_SIZE, variant_idx
            )
            aug_fname = f"v3_aug_{rec['filename'].replace('.wav', '')}_v{variant_idx}.wav"
            dest_file = V3_TRAIN_DIR / "FOOTSTEP" / aug_fname

            sf.write(str(dest_file), aug_audio, config.SAMPLE_RATE, subtype="PCM_16")

            aug_source_id = f"aug_{rec['source_id']}_v{variant_idx}"
            seen_source_ids[aug_source_id] = "train"

            records.append({
                "filename":       aug_fname,
                "source":         "augmented",
                "source_id":      aug_source_id,
                "original_file":  rec["original_file"],
                "original_class": rec["original_class"],
                "binary_class":   "FOOTSTEP",
                "binary_label":   1,
                "split":          "train",
                "fold":           rec["fold"],
                "duration":       config.WINDOW_DURATION_SEC,
                "sample_rate":    config.SAMPLE_RATE,
                "relative_path":  str(dest_file.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
                "v3_augmented":   True,
            })
            aug_added += 1

    logger.info(f"  Added {aug_added} augmented footstep variants.")

    # -----------------------------------------------------------------
    # STEP 4: Save manifest + verify leakage
    # -----------------------------------------------------------------
    df_v3 = pd.DataFrame(records)
    df_v3.to_csv(V3_MANIFEST, index=False)
    logger.info(f"Saved Dataset V3 manifest to {V3_MANIFEST}")

    # Verify no leakage at the source_id level for original (non-augmented) sources
    orig_records = df_v3[~df_v3["source_id"].str.startswith("aug_")]
    train_ids = set(orig_records[orig_records["split"] == "train"]["source_id"])
    val_ids   = set(orig_records[orig_records["split"] == "validation"]["source_id"])
    test_ids  = set(orig_records[orig_records["split"] == "test"]["source_id"])

    leak_tr_val  = train_ids & val_ids
    leak_tr_test = train_ids & test_ids
    leak_val_test = val_ids & test_ids

    if leak_tr_val or leak_tr_test or leak_val_test:
        logger.error(f"[ERROR] Leakage! TV={leak_tr_val}, TT={leak_tr_test}, VT={leak_val_test}")
        return False
    else:
        logger.info("[VERIFIED] Zero source-ID leakage between train / validation / test.")

    # Summary
    logger.info("-" * 65)
    logger.info("DATASET V3 SUMMARY:")
    logger.info(f"  Total files:       {len(df_v3)}")
    ct = pd.crosstab(df_v3["split"], df_v3["binary_class"], margins=True)
    logger.info("\n" + ct.to_string())
    aug_fs = df_v3[(df_v3["v3_augmented"]) & (df_v3["binary_class"] == "FOOTSTEP")]
    logger.info(f"\n  Augmented footstep variants: {len(aug_fs)}")
    hn_rows = df_v3[df_v3["source"] == "esc50_hard_negative"]
    logger.info(f"  Hard-negative clips:         {len(hn_rows)}")
    logger.info(
        "\n  Hard negatives by category:\n"
        + hn_rows["original_class"].value_counts().to_string()
    )
    logger.info("-" * 65)
    return True


if __name__ == "__main__":
    success = build_dataset_v3()
    if success:
        logger.info("Dataset V3 build complete.")
    else:
        logger.error("Dataset V3 build FAILED.")
        sys.exit(1)
