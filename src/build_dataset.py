"""
Phase 3: Dataset Builder Script.
Processes ESC-50, standardizes audio to 16 kHz Mono WAV, separates into fold-aware
train/validation/test sets, preserves detailed category metadata, and generates manifest.csv.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio, save_audio, peak_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DatasetBuilder")


# Broad selection of ambient/human/mechanical classes to accompany percussive negatives
GENERAL_AMBIENT_NEGATIVES = [
    "coughing",
    "breathing",
    "crying_baby",
    "rain",
    "wind",
    "engine",
    "train",
    "chainsaw",
    "chirping_birds",
    "toilet_flush",
    "vacuum_cleaner",
]


def build_dataset(force_rebuild: bool = False):
    logger.info("=" * 60)
    logger.info("PHASE 3: BUILDING STANDARDIZED DATASET")
    logger.info("=" * 60)

    manifest_path = config.PROCESSED_DIR / "manifest.csv"
    if manifest_path.exists() and not force_rebuild:
        logger.warning(f"Manifest already exists at {manifest_path}!")
        logger.warning("Use --force to rebuild. Reading existing manifest summary...")
        df_existing = pd.read_csv(manifest_path)
        logger.info(f"Existing dataset contains {len(df_existing)} standardized audio files.")
        logger.info(df_existing["split"].value_counts().to_string())
        return True

    # 1. Load ESC-50 meta
    if not config.ESC50_META_FILE.exists():
        logger.error(f"Cannot find ESC-50 meta at {config.ESC50_META_FILE}")
        return False

    df_meta = pd.read_csv(config.ESC50_META_FILE)
    logger.info(f"Loaded ESC-50 metadata: {len(df_meta)} rows.")

    # 2. Select target categories: Footsteps + Percussive Negatives + Diverse Ambient Negatives
    selected_categories = set(["footsteps"] + config.PERCUSSIVE_NEGATIVES + GENERAL_AMBIENT_NEGATIVES)
    df_selected = df_meta[df_meta["category"].isin(selected_categories)].copy()
    logger.info(f"Selected {len(df_selected)} clips across {len(selected_categories)} categories.")

    # 3. Create destination directories
    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR, config.PROCESSED_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)
        # Subdirectories for binary classes
        (split_dir / "FOOTSTEP").mkdir(parents=True, exist_ok=True)
        (split_dir / "NON_FOOTSTEP").mkdir(parents=True, exist_ok=True)

    manifest_records = []

    logger.info("Standardizing audio to 16 kHz Mono WAV and populating splits...")
    for _, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Processing audio"):
        src_path = config.ESC50_AUDIO_DIR / row["filename"]
        if not src_path.exists():
            logger.warning(f"File not found: {src_path}")
            continue

        fold = int(row["fold"])
        orig_cat = row["category"]
        is_footstep = (orig_cat == "footsteps")
        binary_class = "FOOTSTEP" if is_footstep else "NON_FOOTSTEP"

        # Split determination (Fold-aware)
        if fold in config.TRAIN_FOLDS:
            split = "train"
            dest_dir = config.TRAIN_DIR / binary_class
        elif fold in config.VAL_FOLDS:
            split = "validation"
            dest_dir = config.VAL_DIR / binary_class
        elif fold in config.TEST_FOLDS:
            split = "test"
            dest_dir = config.TEST_DIR / binary_class
        else:
            continue

        dest_filename = f"esc50_f{fold}_{orig_cat}_{row['filename']}"
        dest_path = dest_dir / dest_filename

        # Load, resample to 16000Hz, mono, peak normalize
        audio, sr = load_audio(src_path, target_sr=config.SAMPLE_RATE, mono=config.MONO)
        audio = peak_normalize(audio, target_peak=0.95)
        save_audio(dest_path, audio, sr=config.SAMPLE_RATE)

        duration = len(audio) / config.SAMPLE_RATE

        manifest_records.append({
            "filename": dest_filename,
            "source": "esc50",
            "original_file": row["filename"],
            "original_class": orig_cat,
            "binary_class": binary_class,
            "binary_label": 1 if is_footstep else 0,
            "fold": fold,
            "split": split,
            "duration": round(duration, 2),
            "sample_rate": config.SAMPLE_RATE,
            "relative_path": str(dest_path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
        })

    # Save manifest
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"Saved dataset manifest to {manifest_path}")

    # Summary
    logger.info("-" * 60)
    logger.info("DATASET BUILD SUMMARY:")
    logger.info(f"Total processed audio files: {len(df_manifest)}")
    logger.info("\nDistribution by split and class:")
    summary_table = pd.crosstab(df_manifest["split"], df_manifest["binary_class"], margins=True)
    logger.info("\n" + summary_table.to_string())

    logger.info("\nOriginal category representation in training set:")
    train_cats = df_manifest[df_manifest["split"] == "train"]["original_class"].value_counts()
    for cat, cnt in train_cats.items():
        logger.info(f"  {cat:20s}: {cnt} clips")

    logger.info("-" * 60)
    logger.info("[SUCCESS] Phase 3 Dataset Builder Complete! Ready for Phase 4 & 5.")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    force = "--force" in sys.argv
    build_dataset(force_rebuild=force)
