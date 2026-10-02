"""
Phase 7: Hard-Negative Mining Pipeline.
Loads Model V1, runs inference on negative audio from training and validation splits ONLY
(NEVER final test set), identifies false-positive triggers, and ranks them by footstep probability.
Saves results to reports/hard_negative_candidates.csv and organizes into dataset/hard_negatives/.
"""

import sys
import shutil
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.dataset_loader import AudioWindowDataset
from src.models import BaselineFootstepCNN
from src.audio_utils import save_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HardNegativeMiner")


CATEGORY_MAPPING = {
    "door_wood_knock": "knocking",
    "clapping": "clapping",
    "mouse_click": "mouse_click",
    "can_opening": "can_opening",
    "keyboard_typing": "tapping",
    "door_wood_creaks": "door",
    "clock_tick": "tapping",
    "glass_breaking": "object_drop",
    "water_drops": "tapping",
    "chair": "chair",
    "table_hit": "table_hit",
    "hand_hit": "hand_hit",
}


def mine_hard_negatives(threshold: float = 0.40):
    logger.info("=" * 60)
    logger.info("PHASE 7: HARD-NEGATIVE MINING WITH MODEL V1")
    logger.info("=" * 60)

    # 1. Load Model V1
    if not config.BASELINE_MODEL_PATH.exists():
        logger.error(f"Baseline model not found at {config.BASELINE_MODEL_PATH}")
        return

    device = config.DEVICE
    model = BaselineFootstepCNN(in_channels=1, dropout=0.0).to(device)
    checkpoint = torch.load(config.BASELINE_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded Model V1 ({checkpoint.get('model_type', 'BaselineFootstepCNN')})")

    # 2. Load manifest and filter for TRAIN and VALIDATION splits ONLY
    manifest_path = config.PROCESSED_DIR / "manifest.csv"
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}")
        return

    df_manifest = pd.read_csv(manifest_path)
    # Exclude test split strictly
    df_dev = df_manifest[df_manifest["split"].isin(["train", "validation"])].reset_index(drop=True)
    df_dev_neg = df_dev[df_dev["binary_class"] == "NON_FOOTSTEP"].reset_index(drop=True)
    logger.info(f"Auditing {len(df_dev_neg)} negative audio clips from train + validation splits.")

    # 3. Create dataset for dev negative windows
    dev_ds = AudioWindowDataset(df_dev, split="train", is_train=False)
    # Also get validation windows
    val_ds = AudioWindowDataset(df_dev, split="validation", is_train=False)

    hard_candidates = []

    for ds_name, ds in [("train", dev_ds), ("validation", val_ds)]:
        logger.info(f"Scanning {ds_name} split ({len(ds)} windows)...")
        for i in range(len(ds)):
            spec, label, meta = ds[i]
            # Only test negative samples
            if int(label.item()) != 0:
                continue

            with torch.no_grad():
                spec_tensor = spec.unsqueeze(0).to(device)
                logit = model(spec_tensor)
                prob = float(torch.sigmoid(logit).cpu().item())

            # If predicted above mining threshold
            if prob >= threshold:
                orig_cat = meta["original_class"]
                target_subfolder = CATEGORY_MAPPING.get(orig_cat, "other")
                
                hard_candidates.append({
                    "filename": meta["filename"],
                    "source": "esc50",
                    "true_class": "NON_FOOTSTEP",
                    "original_class": orig_cat,
                    "target_subfolder": target_subfolder,
                    "predicted_class": "FOOTSTEP" if prob >= 0.50 else "UNCERTAIN_IMPULSE",
                    "footstep_probability": round(prob, 4),
                    "split": ds_name,
                    "start_sec": meta["start_sec"],
                    "end_sec": meta["end_sec"]
                })

    df_hard = pd.DataFrame(hard_candidates)

    if df_hard.empty:
        logger.info(f"No negative samples exceeded the mining threshold {threshold}.")
        df_hard = pd.DataFrame(columns=[
            "filename", "source", "true_class", "original_class",
            "target_subfolder", "predicted_class", "footstep_probability", "split"
        ])
    else:
        df_hard = df_hard.sort_values(by="footstep_probability", ascending=False).reset_index(drop=True)

    # 4. Save report
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_csv = config.REPORTS_DIR / "hard_negative_candidates.csv"
    df_hard.to_csv(report_csv, index=False)
    logger.info(f"Saved {len(df_hard)} hard-negative candidates to {report_csv}")

    # 5. Populate dataset/hard_negatives/
    for _, row in df_hard.iterrows():
        subfolder = config.PROJECT_ROOT / "dataset" / "hard_negatives" / row["target_subfolder"]
        subfolder.mkdir(parents=True, exist_ok=True)
        # Find standardized source file
        src_file = config.DATASET_DIR / row["split"] / "NON_FOOTSTEP" / row["filename"]
        if src_file.exists():
            dest_file = subfolder / f"hard_neg_{row['split']}_{row['filename']}"
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)

    logger.info("-" * 60)
    logger.info("HARD-NEGATIVE CANDIDATES SUMMARY:")
    logger.info(f"Total candidates flagged (prob >= {threshold}): {len(df_hard)}")
    if not df_hard.empty:
        logger.info("\nBreakdown by original category:")
        cat_counts = df_hard["original_class"].value_counts()
        for cat, cnt in cat_counts.items():
            logger.info(f"  {cat:20s}: {cnt} instances")
        logger.info("\nTop 5 hardest negative windows:")
        for idx, row in df_hard.head(5).iterrows():
            logger.info(f"  {idx+1}. {row['original_class']} ({row['filename']}) -> Prob: {row['footstep_probability']}")

    logger.info("=" * 60)
    return len(df_hard)


if __name__ == "__main__":
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40
    mine_hard_negatives(threshold=thresh)
