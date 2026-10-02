"""
Microphone Discovery & Diagnostics Utility.
Lists audio input devices, queries device capabilities, and tests signal health (RMS, Peak, Silence).
"""

import sys
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import sounddevice as sd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MicrophoneUtils")


def list_microphones() -> List[Dict[str, Any]]:
    """
    Queries sounddevice for all available audio input devices.
    Returns:
        List of dictionaries with device details.
    """
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    input_devices = []

    for idx, dev in enumerate(devices):
        max_in = dev.get("max_input_channels", 0)
        if max_in > 0:
            is_default = (idx == default_input)
            input_devices.append({
                "index": idx,
                "name": dev["name"],
                "hostapi": dev.get("hostapi", 0),
                "channels": max_in,
                "default_samplerate": int(dev.get("default_samplerate", 44100)),
                "is_default": is_default
            })

    return input_devices


def print_microphones():
    """Prints formatted table of available input devices."""
    devices = list_microphones()
    print("\n" + "=" * 65)
    print("AVAILABLE MICROPHONE INPUT DEVICES")
    print("=" * 65)
    if not devices:
        print("  [ERROR] No microphone input devices found on system!")
        print("=" * 65 + "\n")
        return

    print(f"{'ID':4s} | {'Default':7s} | {'Channels':8s} | {'Native SR':9s} | {'Device Name'}")
    print("-" * 65)
    for d in devices:
        def_mark = "  YES  " if d["is_default"] else "       "
        print(f"{d['index']:4d} | {def_mark} | {d['channels']:8d} | {d['default_samplerate']:9d} | {d['name']}")
    print("=" * 65 + "\n")


def test_microphone(device_id: Optional[int] = None, duration_sec: float = 2.0) -> Dict[str, Any]:
    """
    Tests whether the specified microphone captures audio.
    Measures RMS level, peak amplitude, and checks for dead/silent stream.
    """
    sr = config.SAMPLE_RATE
    total_samples = int(sr * duration_sec)
    recorded_chunks = []

    logger.info(f"Testing microphone (Device ID: {device_id if device_id is not None else 'DEFAULT'}, Duration: {duration_sec}s)...")

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio stream status: {status}")
        recorded_chunks.append(indata[:, 0].copy())

    try:
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sr,
            dtype="float32",
            callback=audio_callback
        ):
            sd.sleep(int(duration_sec * 1000))
    except Exception as e:
        logger.error(f"Failed to open microphone stream: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "device_id": device_id,
            "rms": 0.0,
            "peak": 0.0
        }

    if not recorded_chunks:
        logger.error("No audio chunks received from microphone!")
        return {"status": "NO_AUDIO", "rms": 0.0, "peak": 0.0}

    audio = np.concatenate(recorded_chunks, axis=0)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    # Diagnosis
    if peak < 1e-4:
        diag = "DEAD_OR_MUTED (Zero amplitude detected)"
    elif rms < 0.001:
        diag = "VERY_QUIET (Low signal or distant microphone)"
    else:
        diag = "HEALTHY (Audio signal actively received)"

    result = {
        "status": "OK" if peak >= 1e-4 else "SILENT",
        "device_id": device_id,
        "sample_rate": sr,
        "duration_sec": duration_sec,
        "total_samples": len(audio),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "diagnosis": diag
    }

    print("\n" + "=" * 55)
    print("MICROPHONE HEALTH TEST RESULTS")
    print("=" * 55)
    print(f"Device:       {device_id if device_id is not None else 'Default'}")
    print(f"Sample Rate:  {sr} Hz")
    print(f"RMS Level:    {rms:.6f}")
    print(f"Peak Level:   {peak:.6f}")
    print(f"Diagnosis:    {diag}")
    print("=" * 55 + "\n")

    return result


if __name__ == "__main__":
    if "--list" in sys.argv or "-l" in sys.argv:
        print_microphones()
    elif "--test" in sys.argv or "-t" in sys.argv:
        dev_id = None
        for i, arg in enumerate(sys.argv):
            if arg in ("--device", "-d") and i + 1 < len(sys.argv):
                dev_id = int(sys.argv[i + 1])
        test_microphone(device_id=dev_id)
    else:
        print_microphones()
        test_microphone()
