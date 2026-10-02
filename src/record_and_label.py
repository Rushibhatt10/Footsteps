"""
Interactive Microphone Recorder & Labeling Utility.
Enables quick recording of real audio samples through the user's physical speaker-and-mic setup,
standardizes to 16 kHz Mono WAV, applies labeling, and registers into the custom dataset catalog.
"""

import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.audio_utils import peak_normalize
from src.microphone_utils import list_microphones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RecordAndLabel")


def record_sample(
    duration_sec: float = 5.0,
    device_id: int | None = None,
    sample_rate: int = config.SAMPLE_RATE
) -> np.ndarray:
    """Records audio from microphone and returns 16kHz float32 mono array."""
    devices = sd.query_devices()
    chosen_device = device_id
    if chosen_device is None:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            chosen_device = default_in

    dev_name = devices[chosen_device]["name"] if chosen_device is not None else "Default"
    dev_info = sd.query_devices(chosen_device, "input")
    native_sr = int(dev_info["default_samplerate"])

    logger.info(f"Using Microphone: {dev_name} (ID: {chosen_device})")

    # Check 16kHz support
    use_native = False
    rec_sr = sample_rate
    try:
        sd.check_input_settings(device=chosen_device, channels=1, samplerate=sample_rate, dtype="float32")
    except Exception:
        rec_sr = native_sr
        use_native = True
        logger.info(f"Microphone requires native rate: {rec_sr} Hz. Will resample to {sample_rate} Hz.")

    print(f"\nRecording {duration_sec:.1f} seconds of audio in:")
    for count in [3, 2, 1]:
        print(f"  {count}...")
        time.sleep(1.0)
    print("  >>> RECORDING NOW! Play or make the sound now! <<<")

    total_frames = int(duration_sec * rec_sr)
    recording = sd.rec(total_frames, samplerate=rec_sr, channels=1, dtype="float32", device=chosen_device)
    sd.wait()
    print("  >>> RECORDING COMPLETE! <<<\n")

    audio = recording[:, 0].flatten()

    # Resample if needed
    if use_native and rec_sr != sample_rate:
        target_len = int(len(audio) * (sample_rate / rec_sr))
        audio = signal.resample(audio, target_len).astype(np.float32)

    return audio


def save_and_catalog(
    audio: np.ndarray,
    label: str,
    descriptor: str,
    duration_sec: float,
    notes: str = ""
) -> Path:
    """Saves standardized recording and appends to custom manifest."""
    custom_dir = config.DATASET_DIR / "custom" / "speaker_mic"
    dest_dir = custom_dir / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in descriptor)
    filename = f"mic_{label.lower()}_{safe_name}_{timestamp}.wav"
    filepath = dest_dir / filename

    # Standardize peak
    audio_norm = peak_normalize(audio, target_peak=0.95)
    sf.write(str(filepath), audio_norm, config.SAMPLE_RATE, subtype="PCM_16")

    # Catalog
    catalog_path = config.DATASET_DIR / "custom" / "custom_manifest.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "filename": filename,
        "source": "speaker_mic_live",
        "source_id": f"mic_{timestamp}",
        "original_file": filename,
        "original_class": descriptor,
        "binary_class": label,
        "binary_label": 1 if label == "FOOTSTEP" else 0,
        "split": "train",
        "duration": round(duration_sec, 2),
        "sample_rate": config.SAMPLE_RATE,
        "relative_path": str(filepath.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
        "notes": notes
    }

    df_new = pd.DataFrame([record])
    if catalog_path.exists():
        df_existing = pd.read_csv(catalog_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(catalog_path, index=False)
    logger.info(f"Saved recording to: {filepath}")
    logger.info(f"Cataloged in: {catalog_path}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Record and label real microphone/speaker audio")
    parser.add_argument("--device", "-d", type=int, default=None, help="Microphone device ID")
    parser.add_argument("--duration", "-t", type=float, default=5.0, help="Recording duration in seconds (default: 5.0)")
    parser.add_argument("--label", "-l", type=str, choices=["FOOTSTEP", "NON_FOOTSTEP"], default=None, help="Class label")
    parser.add_argument("--name", "-n", type=str, default=None, help="Acoustic descriptor (e.g. wood_step_speaker, phone_mic_tap)")
    parser.add_argument("--notes", type=str, default="", help="Optional notes about acoustic environment or setup")
    parser.add_argument("--list-devices", action="store_true", help="List available microphones and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_microphones()
        return

    # Interactive prompts if arguments not supplied
    duration = args.duration
    label = args.label
    name = args.name

    if label is None:
        print("Choose class label:")
        print("  [1] FOOTSTEP")
        print("  [2] NON_FOOTSTEP")
        choice = input("Enter choice (1 or 2): ").strip()
        label = "FOOTSTEP" if choice == "1" else "NON_FOOTSTEP"

    if name is None:
        name = input("Enter a short description (e.g. sneaker_tile_speaker, desk_tap, ambient_hum): ").strip()
        if not name:
            name = "sample"

    audio = record_sample(duration_sec=duration, device_id=args.device)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"Captured audio stats: RMS={rms:.4f}, Peak={peak:.4f}")

    if peak < 0.001:
        print("[WARNING] Audio energy is extremely low! Check your microphone volume.")

    save_and_catalog(audio, label=label, descriptor=name, duration_sec=duration, notes=args.notes)
    print("\n[SUCCESS] Sample recorded and cataloged successfully.")


if __name__ == "__main__":
    main()
