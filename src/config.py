"""
Centralized Configuration for Footstep Detection System.
All paths, audio specs, training hyperparameters, and detection thresholds are defined here.
"""

from pathlib import Path
import torch

# =====================================================================
# PATHS
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ESC-50 paths
ESC50_DIR = PROJECT_ROOT / "esc50"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_META_FILE = ESC50_DIR / "meta" / "esc50.csv"

# Dataset directories
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
CUSTOM_DIR = DATASET_DIR / "custom"

# Output and storage directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
TEST_AUDIO_DIR = PROJECT_ROOT / "test_audio"
HARD_NEGATIVES_DIR = TEST_AUDIO_DIR / "hard_negatives"

# App & Database
APP_DIR = PROJECT_ROOT / "app"
DATABASE_PATH = PROJECT_ROOT / "footstep_detections.db"

# Model paths
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pth"
BEST_MODEL_PATH = MODELS_DIR / "footstep_detector.pth"

# =====================================================================
# AUDIO STANDARDIZATION & SEGMENTATION
# =====================================================================
SAMPLE_RATE = 16000          # 16 kHz standard
MONO = True
AUDIO_FORMAT = "wav"

# Audio window for impulse & resonance capture
WINDOW_DURATION_SEC = 1.5    # 1.5 seconds gives clear transient + decay context
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION_SEC)  # 24,000 samples
HOP_DURATION_SEC = 0.5       # Sliding window hop
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION_SEC)        # 8,000 samples (66.7% overlap)

# =====================================================================
# FEATURE EXTRACTION (Log-Mel Spectrogram)
# =====================================================================
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 20.0
F_MAX = 8000.0
POWER = 2.0

# Expected spectrogram shape for 1.5s at sr=16000, n_fft=1024, hop_length=512:
# time frames = 1 + int((24000 - 1024) / 512) + 2 = ~48 frames (depending on padding)
# -> (1, 64, 48)

# =====================================================================
# CLASSES & LABELS
# =====================================================================
CLASSES = ["NON_FOOTSTEP", "FOOTSTEP"]
LABEL_TO_IDX = {"NON_FOOTSTEP": 0, "FOOTSTEP": 1}
IDX_TO_LABEL = {0: "NON_FOOTSTEP", 1: "FOOTSTEP"}

# Folds from ESC-50
TRAIN_FOLDS = [1, 2, 3]
VAL_FOLDS = [4]
TEST_FOLDS = [5]

# High-priority hard negative categories from ESC-50
PERCUSSIVE_NEGATIVES = [
    "door_wood_knock",
    "clapping",
    "keyboard_typing",
    "mouse_click",
    "door_wood_creaks",
    "clock_tick",
    "glass_breaking",
    "water_drops",
    "can_opening",
]

# =====================================================================
# TRAINING HYPERPARAMETERS
# =====================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Data Augmentation (Train only)
AUGMENT_PROBABILITY = 0.5
TIME_SHIFT_MAX_SEC = 0.2
PITCH_SHIFT_STEPS = 2
NOISE_FACTOR = 0.005

# =====================================================================
# LIVE DETECTION & TEMPORAL FILTERING
# =====================================================================
CONFIDENCE_THRESHOLD = 0.70  # Min probability to flag footstep candidate
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.70
MIN_CONFIRMATIONS = 2        # Consecutive windows required
DETECTION_COOLDOWN_SEC = 0.8 # Min seconds between declared footstep events
INPUT_DEVICE_INDEX = None    # None = system default
BUFFER_DURATION_SEC = 3.0    # Ring buffer length in seconds for live engine
