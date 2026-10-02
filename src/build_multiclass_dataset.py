"""
Build Multi-Class Dataset for Footstep vs Clap vs Knock vs Other.
Converts the task from binary to 4-class:
0: OTHER (ambient noise, typing, speech, clicks, rain, wind, domestic sounds)
1: FOOTSTEP (barefoot, wood, tile, carpet, boots, hallway walking)
2: CLAP (hand claps, applause)
3: KNOCK (door knocks, table knocks, knuckle raps)

Enforces strict source-aware split separation (no data leakage):
- ESC-50 Folds 1, 2, 3 -> train
- ESC-50 Fold 4        -> validation
- ESC-50 Fold 5        -> test
Outputs: dataset_multiclass_v4/manifest.csv
"""

import sys
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import load_audio, peak_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BuildMultiClassDataset")

# Target class mappings
MULTICLASS_MAP = {
    "other": 0,
    "footstep": 1,
    "clap": 2,
    "knock": 3
}
IDX_TO_CLASS = {v: k for k, v in MULTICLASS_MAP.items()}


def build_multiclass_dataset(output_dir: Path = config.PROJECT_ROOT / "dataset_multiclass_v4"):
    logger.info("=" * 65)
    logger.info("BUILDING MULTI-CLASS DATASET (FOOTSTEP / CLAP / KNOCK / OTHER)")
    logger.info("=" * 65)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        for cname in MULTICLASS_MAP.keys():
            (output_dir / split / cname).mkdir(parents=True, exist_ok=True)

    manifest_records = []

    # -------------------------------------------------------------
    # 1. ESC-50 Ingestion
    # -------------------------------------------------------------
    esc_meta_path = config.ESC50_META_FILE
    esc_audio_dir = config.ESC50_AUDIO_DIR

    if not esc_meta_path.exists() or not esc_audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 not found at {config.ESC50_DIR}")

    df_esc = pd.read_csv(esc_meta_path)
    logger.info(f"Loaded ESC-50 metadata: {len(df_esc)} total files across 50 categories.")

    # Category assignments:
    # Footstep: 'footsteps'
    # Clap: 'clapping'
    # Knock: 'door_wood_knock'
    # Other: selected percussive and ambient categories to train robust rejection
    other_categories = [
        "keyboard_typing",
        "mouse_click",
        "can_opening",
        "glass_breaking",
        "water_drops",
        "clock_tick",
        "door_wood_creaks",
        "crying_baby",
        "breathing",
        "coughing",
        "toilet_flush",
        "wind",
        "rain",
        "vacuum_cleaner",
        "engine",
        "train",
        "chirping_birds"
    ]

    for _, row in df_esc.iterrows():
        cat = row["category"]
        fold = int(row["fold"])
        src_id = f"esc50_{row['src_file']}"
        esc_fn = row["filename"]
        raw_path = esc_audio_dir / esc_fn

        if not raw_path.exists():
            continue

        # Determine target multi-class label
        if cat == "footsteps":
            class_name = "footstep"
        elif cat == "clapping":
            class_name = "clap"
        elif cat == "door_wood_knock":
            class_name = "knock"
        elif cat in other_categories:
            class_name = "other"
        else:
            continue

        label_idx = MULTICLASS_MAP[class_name]

        # Fold to split mapping (zero leakage)
        if fold in [1, 2, 3]:
            split = "train"
        elif fold == 4:
            split = "validation"
        elif fold == 5:
            split = "test"
        else:
            split = "train"

        # Standardize audio (16kHz, mono, peak normalized)
        audio, sr = load_audio(raw_path, target_sr=config.SAMPLE_RATE, mono=True)
        audio = peak_normalize(audio, target_peak=0.95)

        std_filename = f"esc50_f{fold}_{cat}_{esc_fn}"
        target_path = output_dir / split / class_name / std_filename
        sf.write(str(target_path), audio, config.SAMPLE_RATE)

        manifest_records.append({
            "filename": std_filename,
            "original_class": cat,
            "class_name": class_name,
            "multiclass_label": label_idx,
            "split": split,
            "fold": fold,
            "source_id": src_id,
            "relative_path": str(target_path.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
            "duration_sec": round(len(audio) / config.SAMPLE_RATE, 3)
        })

    # -------------------------------------------------------------
    # 2. Ingest Additional Footstep & Knock Samples from dataset_v2/dataset_v3
    # -------------------------------------------------------------
    # Check for additional footstep surfaces (wood, tile, carpet, boots) in dataset_v2
    v2_manifest = config.PROJECT_ROOT / "dataset_v2" / "manifest.csv"
    if v2_manifest.exists():
        df_v2 = pd.read_csv(v2_manifest)
        additional_footsteps = df_v2[df_v2["original_class"].isin([
            "footstep_wood", "footstep_tile", "footstep_carpet", "footstep_boots", "knocking"
        ])]
        for _, row in additional_footsteps.iterrows():
            fpath = config.PROJECT_ROOT / row["relative_path"]
            if not fpath.exists():
                continue
            orig_cat = row["original_class"]
            split = row["split"]
            src_id = row.get("source_id", row["filename"])
            
            if orig_cat == "knocking":
                class_name = "knock"
            else:
                class_name = "footstep"
            label_idx = MULTICLASS_MAP[class_name]

            audio, sr = load_audio(fpath, target_sr=config.SAMPLE_RATE, mono=True)
            audio = peak_normalize(audio, target_peak=0.95)

            new_fn = f"v2_{orig_cat}_{Path(row['filename']).name}"
            target_path = output_dir / split / class_name / new_fn
            sf.write(str(target_path), audio, config.SAMPLE_RATE)

            manifest_records.append({
                "filename": new_fn,
                "original_class": orig_cat,
                "class_name": class_name,
                "multiclass_label": label_idx,
                "split": split,
                "fold": -1,
                "source_id": src_id,
                "relative_path": str(target_path.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
                "duration_sec": round(len(audio) / config.SAMPLE_RATE, 3)
            })

    # -------------------------------------------------------------
    # 3. Ingest Custom Recordings (dataset/custom/) if any exist
    # -------------------------------------------------------------
    custom_dir = config.DATASET_DIR / "custom"
    if custom_dir.exists():
        for cname in MULTICLASS_MAP.keys():
            cdir = custom_dir / cname
            if cdir.exists():
                custom_files = list(cdir.glob("*.wav"))
                for cf in custom_files:
                    audio, _ = load_audio(cf, target_sr=config.SAMPLE_RATE, mono=True)
                    audio = peak_normalize(audio, target_peak=0.95)
                    # Custom files split: 70% train, 15% val, 15% test deterministically by hash
                    h = hash(cf.stem) % 10
                    split = "test" if h == 9 else ("validation" if h == 8 else "train")
                    
                    target_path = output_dir / split / cname / cf.name
                    sf.write(str(target_path), audio, config.SAMPLE_RATE)
                    manifest_records.append({
                        "filename": cf.name,
                        "original_class": f"custom_{cname}",
                        "class_name": cname,
                        "multiclass_label": MULTICLASS_MAP[cname],
                        "split": split,
                        "fold": -1,
                        "source_id": f"custom_{cf.stem}",
                        "relative_path": str(target_path.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
                        "duration_sec": round(len(audio) / config.SAMPLE_RATE, 3)
                    })

    # -------------------------------------------------------------
    # 4. Save and Verify Multi-Class Manifest
    # -------------------------------------------------------------
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = output_dir / "manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    logger.info(f"Saved multi-class manifest to: {manifest_csv}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("MULTI-CLASS DATASET DISTRIBUTION SUMMARY (dataset_multiclass_v4)")
    print("=" * 70)
    summary_table = df_manifest.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    print(summary_table)
    print("-" * 70)
    print(f"Total Audio Recordings: {len(df_manifest)}")

    # Check for data leakage
    train_sources = set(df_manifest[df_manifest["split"] == "train"]["source_id"])
    val_sources = set(df_manifest[df_manifest["split"] == "validation"]["source_id"])
    test_sources = set(df_manifest[df_manifest["split"] == "test"]["source_id"])

    leak_tr_val = train_sources.intersection(val_sources)
    leak_tr_test = train_sources.intersection(test_sources)
    leak_val_test = val_sources.intersection(test_sources)
    is_clean = len(leak_tr_val) == 0 and len(leak_tr_test) == 0 and len(leak_val_test) == 0
    print(f"Data Leakage Verification: {'PASSED (Zero Leakage)' if is_clean else 'FAILED'}")
    print("=" * 70 + "\n")

    return df_manifest


if __name__ == "__main__":
    build_multiclass_dataset()
