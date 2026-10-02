"""
Phase 4: Public Audio Dataset Importer.
Imports and standardizes external public datasets (e.g. AudioSet, local environment recordings,
footstep surface packs) into dataset/raw/public/ with comprehensive metadata tracking.
"""

import sys
import argparse
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
logger = logging.getLogger("PublicAudioImporter")

# Known keyword heuristic mapping for classifying imported files
POSITIVE_KEYWORDS = ["footstep", "walk", "step", "boots", "carpet", "tile", "wood_step", "shoe", "hallway"]
NEGATIVE_LOOKALIKE_KEYWORDS = {
    "knock": "knocking",
    "table": "table_hit",
    "hit": "hand_hit",
    "clap": "clapping",
    "drop": "object_drop",
    "click": "mouse_click",
    "typing": "keyboard_typing",
    "keyboard": "keyboard_typing",
    "door": "door",
    "chair": "chair",
    "cough": "coughing",
    "fan": "background_noise",
    "speech": "speech",
}


def infer_classes(filename: str) -> tuple[str, str]:
    """Infers original_class and binary_class based on filename keywords."""
    fn_lower = filename.lower()
    
    # Check positive
    for kw in POSITIVE_KEYWORDS:
        if kw in fn_lower:
            # Check specific surface if available
            if "boots" in fn_lower:
                return "footstep_boots", "FOOTSTEP"
            elif "carpet" in fn_lower:
                return "footstep_carpet", "FOOTSTEP"
            elif "tile" in fn_lower:
                return "footstep_tile", "FOOTSTEP"
            elif "wood" in fn_lower and "door" not in fn_lower:
                return "footstep_wood", "FOOTSTEP"
            return "footsteps", "FOOTSTEP"

    # Check negative lookalikes
    for kw, orig_cat in NEGATIVE_LOOKALIKE_KEYWORDS.items():
        if kw in fn_lower:
            return orig_cat, "NON_FOOTSTEP"

    return "ambient_other", "NON_FOOTSTEP"


def import_audio_directory(
    source_dir: Path | str,
    source_name: str = "public_import",
    target_dir: Path | str = None
):
    source_dir = Path(source_dir)
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return []

    target_dir = Path(target_dir) if target_dir else config.RAW_DIR / "public" / source_name
    target_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = config.RAW_DIR / "public" / "import_catalog.csv"
    existing_records = []
    if catalog_path.exists():
        df_ex = pd.read_csv(catalog_path)
        existing_records = df_ex.to_dict("records")

    audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    source_files = [f for f in source_dir.rglob("*") if f.suffix.lower() in audio_extensions]

    logger.info(f"Found {len(source_files)} audio files in {source_dir}")
    new_records = []

    for src_file in tqdm(source_files, desc=f"Importing {source_name}"):
        orig_class, binary_class = infer_classes(src_file.name)
        # Unique source_id derived from filename stem
        source_id = f"{source_name}_{src_file.stem}"
        dest_filename = f"{source_id}.wav"
        dest_path = target_dir / dest_filename

        try:
            audio, sr = load_audio(src_file, target_sr=config.SAMPLE_RATE, mono=config.MONO)
            audio = peak_normalize(audio, target_peak=0.95)
            save_audio(dest_path, audio, sr=config.SAMPLE_RATE)
            duration = round(len(audio) / config.SAMPLE_RATE, 2)

            record = {
                "source": source_name,
                "source_id": source_id,
                "original_filename": src_file.name,
                "original_class": orig_class,
                "binary_class": binary_class,
                "duration": duration,
                "sample_rate": config.SAMPLE_RATE,
                "relative_path": str(dest_path.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
            }
            new_records.append(record)
        except Exception as e:
            logger.warning(f"Error importing {src_file.name}: {e}")

    all_records = existing_records + new_records
    df_catalog = pd.DataFrame(all_records)
    # Deduplicate by source_id
    df_catalog = df_catalog.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    df_catalog.to_csv(catalog_path, index=False)
    logger.info(f"Updated import catalog at {catalog_path} with {len(df_catalog)} total entries.")
    return new_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import external audio dataset")
    parser.add_argument("--source-dir", type=str, required=True, help="Path to audio folder")
    parser.add_argument("--source-name", type=str, default="public_dataset", help="Identifier for data source")
    args = parser.parse_args()

    import_audio_directory(args.source_dir, args.source_name)
