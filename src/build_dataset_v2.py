"""
Phase 8: Balanced Dataset V2 Builder.
Constructs dataset_v2/ by integrating ESC-50 source folds, imported environmental footstep recordings,
and mined hard negatives.
Guarantees:
- The untouched test set (ESC-50 Fold 5) remains 100% identical and separate.
- Strict source-aware splitting: all audio derived from source_id stays in a single split.
- Zero source ID leakage between train, validation, and test.
"""

import sys
import shutil
import random
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
logger = logging.getLogger("DatasetV2Builder")

DATASET_V2_DIR = config.PROJECT_ROOT / "dataset_v2"
V2_TRAIN_DIR = DATASET_V2_DIR / "train"
V2_VAL_DIR = DATASET_V2_DIR / "validation"
V2_TEST_DIR = DATASET_V2_DIR / "test"
V2_MANIFEST_PATH = DATASET_V2_DIR / "manifest.csv"


def build_dataset_v2():
    logger.info("=" * 60)
    logger.info("PHASE 8: CONSTRUCTING DATASET V2 (WITH SOURCE-AWARE SPLITTING)")
    logger.info("=" * 60)

    # 1. Setup directories
    for split_dir in [V2_TRAIN_DIR, V2_VAL_DIR, V2_TEST_DIR]:
        (split_dir / "FOOTSTEP").mkdir(parents=True, exist_ok=True)
        (split_dir / "NON_FOOTSTEP").mkdir(parents=True, exist_ok=True)

    manifest_v2_records = []
    seen_source_ids = {}  # source_id -> split

    # 2. STEP 1: PRESERVE UNTOUCHED TEST SET EXACTLY
    logger.info("Step 1: Copying untouched original test set into dataset_v2/test/...")
    manifest_v1_path = config.PROCESSED_DIR / "manifest.csv"
    if not manifest_v1_path.exists():
        raise FileNotFoundError(f"Manifest V1 not found at {manifest_v1_path}")

    df_v1 = pd.read_csv(manifest_v1_path)
    df_test_v1 = df_v1[df_v1["split"] == "test"].copy()

    for _, row in df_test_v1.iterrows():
        src_file = config.PROJECT_ROOT / row["relative_path"]
        binary_class = row["binary_class"]
        dest_file = V2_TEST_DIR / binary_class / row["filename"]

        shutil.copy2(src_file, dest_file)

        # Derive source_id for ESC-50
        orig_fname = row["original_file"]
        source_id = f"esc50_{orig_fname.split('-')[1]}"

        record = {
            "filename": row["filename"],
            "source": "esc50",
            "source_id": source_id,
            "original_file": orig_fname,
            "original_class": row["original_class"],
            "binary_class": binary_class,
            "binary_label": int(row["binary_label"]),
            "split": "test",
            "fold": int(row["fold"]),
            "duration": float(row["duration"]),
            "sample_rate": int(row["sample_rate"]),
            "relative_path": str(dest_file.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
        }
        manifest_v2_records.append(record)
        seen_source_ids[source_id] = "test"

    logger.info(f"Preserved {len(df_test_v1)} untouched test files (Source IDs: {len(seen_source_ids)}).")

    # 3. STEP 2: COPY ESC-50 TRAIN & VALIDATION SPLITS
    logger.info("Step 2: Processing ESC-50 train (folds 1-3) and validation (fold 4)...")
    df_train_val = df_v1[df_v1["split"].isin(["train", "validation"])].copy()

    for _, row in df_train_val.iterrows():
        src_file = config.PROJECT_ROOT / row["relative_path"]
        binary_class = row["binary_class"]
        split = row["split"]
        dest_dir = V2_TRAIN_DIR if split == "train" else V2_VAL_DIR
        dest_file = dest_dir / binary_class / row["filename"]

        shutil.copy2(src_file, dest_file)

        orig_fname = row["original_file"]
        source_id = f"esc50_{orig_fname.split('-')[1]}"

        # Leakage guard check
        if source_id in seen_source_ids:
            if seen_source_ids[source_id] != split:
                raise ValueError(f"Leakage detected! {source_id} in {seen_source_ids[source_id]} and {split}")
        else:
            seen_source_ids[source_id] = split

        record = {
            "filename": row["filename"],
            "source": "esc50",
            "source_id": source_id,
            "original_file": orig_fname,
            "original_class": row["original_class"],
            "binary_class": binary_class,
            "binary_label": int(row["binary_label"]),
            "split": split,
            "fold": int(row["fold"]),
            "duration": float(row["duration"]),
            "sample_rate": int(row["sample_rate"]),
            "relative_path": str(dest_file.relative_path if hasattr(dest_file, "relative_path") else str(dest_file.relative_to(config.PROJECT_ROOT))).replace("\\", "/")
        }
        manifest_v2_records.append(record)

    # 4. STEP 3: INTEGRATE IMPORTED PUBLIC / ENVIRONMENTAL AUDIO
    logger.info("Step 3: Integrating imported environmental recordings into train and validation...")
    import_catalog_path = config.RAW_DIR / "public" / "import_catalog.csv"

    if import_catalog_path.exists():
        df_import = pd.read_csv(import_catalog_path)
        # Deduplicate against ESC-50 original files
        esc50_files = set(df_v1["original_file"].values)

        new_imported = []
        for _, row in df_import.iterrows():
            orig_fn = row["original_filename"]
            # Skip if this was an ESC-50 copy
            is_esc_copy = False
            for ef in esc50_files:
                if ef in orig_fn:
                    is_esc_copy = True
                    break
            if not is_esc_copy:
                new_imported.append(row)

        logger.info(f"Found {len(new_imported)} genuinely new external recordings to integrate.")

        # Randomize with fixed seed for reproducibility
        rng = random.Random(config.RANDOM_SEED)
        rng.shuffle(new_imported)

        # Assign 80% train, 20% validation. NONE to test.
        for idx, row in enumerate(new_imported):
            split = "train" if (idx % 5 != 0) else "validation"
            binary_class = row["binary_class"]
            dest_dir = V2_TRAIN_DIR if split == "train" else V2_VAL_DIR

            src_file = config.PROJECT_ROOT / row["relative_path"]
            if not src_file.exists():
                continue

            source_id = f"ext_{row['source_id']}"
            dest_filename = f"v2_{source_id}.wav"
            dest_file = dest_dir / binary_class / dest_filename

            shutil.copy2(src_file, dest_file)

            if source_id in seen_source_ids and seen_source_ids[source_id] != split:
                raise ValueError(f"Leakage detected! {source_id} in {seen_source_ids[source_id]} and {split}")
            seen_source_ids[source_id] = split

            manifest_v2_records.append({
                "filename": dest_filename,
                "source": row["source"],
                "source_id": source_id,
                "original_file": row["original_filename"],
                "original_class": row["original_class"],
                "binary_class": binary_class,
                "binary_label": 1 if binary_class == "FOOTSTEP" else 0,
                "split": split,
                "fold": -1,  # external
                "duration": float(row["duration"]),
                "sample_rate": int(row["sample_rate"]),
                "relative_path": str(dest_file.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
            })

    # 5. Save dataset_v2/manifest.csv
    df_manifest_v2 = pd.DataFrame(manifest_v2_records)
    df_manifest_v2.to_csv(V2_MANIFEST_PATH, index=False)
    logger.info(f"Saved Dataset V2 manifest to {V2_MANIFEST_PATH}")

    # 6. Verify zero leakage programmatically
    train_ids = set(df_manifest_v2[df_manifest_v2["split"] == "train"]["source_id"])
    val_ids = set(df_manifest_v2[df_manifest_v2["split"] == "validation"]["source_id"])
    test_ids = set(df_manifest_v2[df_manifest_v2["split"] == "test"]["source_id"])

    leak_tr_val = train_ids.intersection(val_ids)
    leak_tr_test = train_ids.intersection(test_ids)
    leak_val_test = val_ids.intersection(test_ids)

    if leak_tr_val or leak_tr_test or leak_val_test:
        logger.error(f"[ERROR] Source leakage detected! Train/Val: {leak_tr_val}, Train/Test: {leak_tr_test}, Val/Test: {leak_val_test}")
        return False
    else:
        logger.info("[VERIFIED] Zero source ID leakage between train, validation, and test splits!")

    # Summary table
    logger.info("-" * 60)
    logger.info("DATASET V2 SUMMARY:")
    logger.info(f"Total recordings in Dataset V2: {len(df_manifest_v2)}")
    logger.info(f"Unique source recordings: {df_manifest_v2['source_id'].nunique()}")
    logger.info("\nDistribution by split and class:")
    logger.info("\n" + pd.crosstab(df_manifest_v2["split"], df_manifest_v2["binary_class"], margins=True).to_string())
    logger.info("-" * 60)
    return True


if __name__ == "__main__":
    build_dataset_v2()
