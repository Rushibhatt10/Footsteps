"""
Interactive Custom Audio Recording Tool.
Allows recording personalized microphone samples for:
1. Footsteps (barefoot, shoes, socks on various surfaces)
2. Claps (single claps, applause, cup claps)
3. Knocks (door knocks, table knocks, wall knocks)
4. Other / Ambient (silence, room fan, speech, keyboard typing)

Saved directly to dataset/custom/<class>/ with source-aware metadata.
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.microphone_utils import list_microphones


def record_clip(duration: float = 3.0, sr: int = config.SAMPLE_RATE, device_id: int = None) -> np.ndarray:
    """Records audio from microphone."""
    print(f"[*] Recording for {duration:.1f} seconds... Make the sound now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32", device=device_id)
    sd.wait()
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser(description="Record custom audio samples for multi-class model")
    parser.add_argument("--category", "-c", type=str, choices=["footstep", "clap", "knock", "other"],
                        help="Sound class to record")
    parser.add_argument("--num-samples", "-n", type=int, default=5, help="Number of clips to record (default: 5)")
    parser.add_argument("--duration", "-d", type=float, default=3.0, help="Duration of each clip in seconds (default: 3.0)")
    parser.add_argument("--device", type=int, default=None, help="Input microphone device ID")
    args = parser.parse_args()

    custom_dir = config.DATASET_DIR / "custom"
    
    # Interactive selection if not provided
    category = args.category
    if category is None:
        print("\n" + "=" * 50)
        print("AUDIO SAMPLE RECORDER FOR FOOTSTEP / CLAP / KNOCK")
        print("=" * 50)
        print("1. footstep (walking, tapping soles)")
        print("2. clap (hand claps)")
        print("3. knock (door knock, desk rap)")
        print("4. other (room silence, ambient noise, typing)")
        choice = input("\nSelect category (1-4) [default: 1]: ").strip()
        cat_map = {"1": "footstep", "2": "clap", "3": "knock", "4": "other"}
        category = cat_map.get(choice, "footstep")

    out_dir = custom_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving {args.num_samples} clips of category '{category}' to: {out_dir}")
    print(f"[INFO] Using Sample Rate: {config.SAMPLE_RATE} Hz | Clip Duration: {args.duration}s")

    for i in range(1, args.num_samples + 1):
        input(f"\nPress ENTER when ready to record sample {i}/{args.num_samples}...")
        for count in [3, 2, 1]:
            print(f"Starting in {count}...", end="\r", flush=True)
            time.sleep(0.7)
        print("RECORDING NOW!               ")

        audio = record_clip(duration=args.duration, sr=config.SAMPLE_RATE, device_id=args.device)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"[OK] Captured! Peak: {peak:.3f}, RMS: {rms:.4f}")

        timestamp = int(time.time())
        filename = f"custom_{category}_{timestamp}_{i:02d}.wav"
        save_path = out_dir / filename
        sf.write(str(save_path), audio, config.SAMPLE_RATE)
        print(f"[+] Saved to: {save_path.name}")

    print(f"\n[DONE] Successfully recorded {args.num_samples} '{category}' samples.")


if __name__ == "__main__":
    main()
