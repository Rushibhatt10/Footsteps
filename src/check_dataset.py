"""
Dataset Quality Assurance & Integrity Checker for Dataset V2.
Validates file existence, class distributions, category diversity,
and strictly checks for zero source ID leakage across train, validation, and test splits.
Generates reports/dataset_v2_summary.json and reports/dataset_v2_summary.txt.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DatasetV2Checker")

V2_MANIFEST_PATH = config.PROJECT_ROOT / "dataset_v2" / "manifest.csv"


def check_dataset_v2():
    logger.info("=" * 60)
    logger.info("PHASE 15: DATASET V2 QUALITY AUDIT & LEAKAGE CHECK")
    logger.info("=" * 60)

    if not V2_MANIFEST_PATH.exists():
        logger.error(f"Manifest not found at {V2_MANIFEST_PATH}")
        sys.exit(1)

    df = pd.read_csv(V2_MANIFEST_PATH)
    logger.info(f"Loaded manifest with {len(df)} records.")

    # 1. Check missing files
    missing_files = []
    for _, row in df.iterrows():
        p = config.PROJECT_ROOT / row["relative_path"]
        if not p.exists():
            missing_files.append(row["relative_path"])

    if missing_files:
        logger.error(f"FAIL: Found {len(missing_files)} missing audio files on disk!")
        for mf in missing_files[:5]:
            logger.error(f"  Missing: {mf}")
        sys.exit(1)
    else:
        logger.info("[PASS] All audio files exist on disk.")

    # 2. Check duplicate filenames
    duplicates = df["filename"].duplicated().sum()
    if duplicates > 0:
        logger.error(f"FAIL: Found {duplicates} duplicate filenames in manifest!")
        sys.exit(1)
    else:
        logger.info("[PASS] Zero duplicate filenames.")

    # 3. Source ID Leakage check (Mandatory)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    train_sources = set(train_df["source_id"])
    val_sources = set(val_df["source_id"])
    test_sources = set(test_df["source_id"])

    leak_tr_val = train_sources.intersection(val_sources)
    leak_tr_test = train_sources.intersection(test_sources)
    leak_val_test = val_sources.intersection(test_sources)

    total_leaked_ids = len(leak_tr_val) + len(leak_tr_test) + len(leak_val_test)

    if total_leaked_ids > 0:
        logger.error("=" * 60)
        logger.error("VALIDATION FAILED: SOURCE ID LEAKAGE DETECTED ACROSS SPLITS!")
        logger.error(f"Train <-> Val overlaps: {leak_tr_val}")
        logger.error(f"Train <-> Test overlaps: {leak_tr_test}")
        logger.error(f"Val <-> Test overlaps: {leak_val_test}")
        logger.error("=" * 60)
        sys.exit(1)
    else:
        logger.info("[PASS] SOURCE-AWARE INTEGRITY: Zero source ID leakage across all splits!")

    # 4. Class counts and category breakdown
    total_recordings = len(df)
    unique_sources = df["source_id"].nunique()
    footstep_cnt = len(df[df["binary_class"] == "FOOTSTEP"])
    non_footstep_cnt = len(df[df["binary_class"] == "NON_FOOTSTEP"])

    train_cnt = len(train_df)
    val_cnt = len(val_df)
    test_cnt = len(test_df)

    neg_category_counts = df[df["binary_class"] == "NON_FOOTSTEP"]["original_class"].value_counts().to_dict()
    pos_category_counts = df[df["binary_class"] == "FOOTSTEP"]["original_class"].value_counts().to_dict()

    # 5. Generate reports
    summary_data = {
        "dataset_version": "V2",
        "total_recordings": total_recordings,
        "unique_source_recordings": unique_sources,
        "footstep_count": footstep_cnt,
        "non_footstep_count": non_footstep_cnt,
        "split_counts": {
            "train": {
                "total": train_cnt,
                "footstep": len(train_df[train_df["binary_class"] == "FOOTSTEP"]),
                "non_footstep": len(train_df[train_df["binary_class"] == "NON_FOOTSTEP"])
            },
            "validation": {
                "total": val_cnt,
                "footstep": len(val_df[val_df["binary_class"] == "FOOTSTEP"]),
                "non_footstep": len(val_df[val_df["binary_class"] == "NON_FOOTSTEP"])
            },
            "test": {
                "total": test_cnt,
                "footstep": len(test_df[test_df["binary_class"] == "FOOTSTEP"]),
                "non_footstep": len(test_df[test_df["binary_class"] == "NON_FOOTSTEP"])
            }
        },
        "negative_category_counts": neg_category_counts,
        "footstep_surface_counts": pos_category_counts,
        "missing_files_count": len(missing_files),
        "duplicate_files_count": int(duplicates),
        "source_leakage_detected": False,
        "duplicate_source_ids_across_splits": total_leaked_ids
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "dataset_v2_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved dataset summary JSON to {json_path}")

    txt_path = config.REPORTS_DIR / "dataset_v2_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DATASET V2 AUDIT & INTEGRITY SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total recordings:             {total_recordings}\n")
        f.write(f"Unique source recordings:     {unique_sources}\n")
        f.write(f"FOOTSTEP recordings:          {footstep_cnt}\n")
        f.write(f"NON_FOOTSTEP recordings:      {non_footstep_cnt}\n\n")
        f.write("Splits:\n")
        f.write(f"  Train:      {train_cnt} ({summary_data['split_counts']['train']['footstep']} footsteps, {summary_data['split_counts']['train']['non_footstep']} non-footsteps)\n")
        f.write(f"  Validation: {val_cnt} ({summary_data['split_counts']['validation']['footstep']} footsteps, {summary_data['split_counts']['validation']['non_footstep']} non-footsteps)\n")
        f.write(f"  Test:       {test_cnt} ({summary_data['split_counts']['test']['footstep']} footsteps, {summary_data['split_counts']['test']['non_footstep']} non-footsteps)\n\n")
        f.write(f"Missing files:                {len(missing_files)}\n")
        f.write(f"Duplicate files:              {duplicates}\n")
        f.write(f"Source ID Leakage Across Splits: NO (0 duplicates)\n\n")
        f.write("Footstep Surface Varieties:\n")
        for cat, cnt in pos_category_counts.items():
            f.write(f"  - {cat:20s}: {cnt}\n")
        f.write("\nNegative Sound Categories (Look-Alikes & Ambient):\n")
        for cat, cnt in neg_category_counts.items():
            f.write(f"  - {cat:20s}: {cnt}\n")

    logger.info(f"Saved dataset summary TXT to {txt_path}")

    # Console display
    print("\n" + "=" * 60)
    print("DATASET V2 STATUS")
    print("=" * 60)
    print(f"FOOTSTEP recordings:     {footstep_cnt}")
    print(f"NON_FOOTSTEP recordings: {non_footstep_cnt}")
    print("\nHard-negative categories:")
    for cat in ["door_wood_knock", "knocking", "clapping", "mouse_click", "keyboard_typing", "can_opening", "glass_breaking", "door", "table_hit"]:
        if cat in neg_category_counts:
            print(f"  - {cat:20s}: {neg_category_counts[cat]} recordings")
    print(f"\nTrain:      {train_cnt} recordings ({summary_data['split_counts']['train']['footstep']} footsteps, {summary_data['split_counts']['train']['non_footstep']} non-footsteps)")
    print(f"Validation: {val_cnt} recordings ({summary_data['split_counts']['validation']['footstep']} footsteps, {summary_data['split_counts']['validation']['non_footstep']} non-footsteps)")
    print(f"Test:       {test_cnt} recordings ({summary_data['split_counts']['test']['footstep']} footsteps, {summary_data['split_counts']['test']['non_footstep']} non-footsteps)")
    print(f"\nSource leakage:                      NO")
    print(f"Missing files:                       {len(missing_files)}")
    print(f"Duplicate source IDs across splits:  {total_leaked_ids}")
    print(f"Public datasets imported:            ['ESC-50', 'environmental_footstep_surfaces (boots, carpet, tile, wood, hallway)']")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    check_dataset_v2()
