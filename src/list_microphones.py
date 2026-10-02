"""CLI wrapper to list all available microphones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.microphone_utils import print_microphones

if __name__ == "__main__":
    print_microphones()
