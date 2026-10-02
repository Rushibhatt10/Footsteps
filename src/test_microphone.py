"""CLI wrapper to test microphone signal health and levels."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.microphone_utils import test_microphone

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test microphone audio input")
    parser.add_argument("--device", "-d", type=int, default=None, help="Device index")
    parser.add_argument("--duration", "-t", type=float, default=2.0, help="Test duration in seconds")
    args = parser.parse_args()

    test_microphone(device_id=args.device, duration_sec=args.duration)
