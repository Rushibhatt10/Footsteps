"""
Dataset Downloader & Preparer for Public Audio Corpora.
Provides utilities to fetch and prepare external audio clips for Claps, Knocks, and Footsteps:
- ESC-50 (already integrated)
- UrbanSound8K / FSD50K / Freesound links and pre-processing
"""

import sys
import os
import argparse
from pathlib import Path
import urllib.request
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def check_local_dataset():
    """Checks the status of available datasets in the local workspace."""
    esc_audio = config.ESC50_AUDIO_DIR
    esc_meta = config.ESC50_META_FILE
    
    print("\n" + "=" * 60)
    print("DATASET STATUS REPORT")
    print("=" * 60)
    if esc_audio.exists() and esc_meta.exists():
        df = pd.read_csv(esc_meta)
        print(f"[OK] ESC-50 Dataset is PRESENT:")
        print(f"     - Total audio files: {len(list(esc_audio.glob('*.wav')))}")
        print(f"     - Clapping clips:    {len(df[df['category'] == 'clapping'])}")
        print(f"     - Door knock clips:  {len(df[df['category'] == 'door_wood_knock'])}")
        print(f"     - Footstep clips:    {len(df[df['category'] == 'footsteps'])}")
    else:
        print("[!] ESC-50 is NOT found at:", esc_audio)

    custom_dir = config.DATASET_DIR / "custom"
    if custom_dir.exists():
        custom_wavs = list(custom_dir.rglob("*.wav"))
        print(f"\n[OK] Custom Recorded Samples:")
        print(f"     - Total custom files: {len(custom_wavs)}")
    else:
        print("\n[-] No custom recordings found yet. Use `python src/record_custom_samples.py` to record via mic.")

    print("=" * 60 + "\n")


def print_public_source_guide():
    """Prints instructions and sources for additional public datasets."""
    print("""
PUBLIC DATASET RECOMMENDATIONS FOR TRANSIENT IMPACTS:
----------------------------------------------------
1. ESC-50 (Environmental Sound Classification) - ALREADY INTEGRATED
   - 40 footsteps, 40 clapping, 40 door_wood_knock, 40 keyboard, 40 mouse click
   - Location: esc50/

2. FSD50K (Freesound Dataset 50K)
   - Category tags: 'Hands', 'Clapping', 'Knock', 'Door', 'Footsteps', 'Walk_and_footsteps'
   - URL: https://zenodo.org/record/4060432

3. UrbanSound8K
   - 8,732 sound clips across 10 urban sound classes
   - URL: https://urbansounddataset.oxfordmartin.ox.ac.uk/

4. Freesound API / Web
   - Download individual WAV samples of table knocks, applause, shoes on tile/concrete
   - Save directly into: dataset/custom/knock/ or dataset/custom/clap/
""")


if __name__ == "__main__":
    check_local_dataset()
    print_public_source_guide()
