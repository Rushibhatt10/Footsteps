# Robust Footstep vs Non-Footstep Detection System

An end-to-end Machine Learning audio classification system designed to distinguish actual footsteps from acoustic look-alikes (knocking, table hits, clapping, doors closing, keyboard typing, chair movement, and ambient noise).

## Architecture Overview
```
MICROPHONE / AUDIO FILE
        ↓
AUDIO CAPTURE & BUFFERING (16 kHz, Mono)
        ↓
PREPROCESSING (Peak Normalization, Transient Preservation)
        ↓
AUDIO SEGMENTATION (1.5s sliding window, 0.5s hop)
        ↓
FEATURE EXTRACTION (Log-Mel Spectrogram, 64 mels, n_fft=1024, hop=512)
        ↓
CONVOLUTIONAL NEURAL NETWORK / AUDIO MODEL
        ↓
BINARY PROBABILITY (FOOTSTEP vs NON_FOOTSTEP)
        ↓
TEMPORAL VALIDATION & CONFIRMATION LOGIC
        ↓
FOOTSTEP EVENT & SQLite DATABASE LOGGING
        ↓
FASTAPI REST API & REAL-TIME DASHBOARD
```

## Quick Start

### 1. Validate Dataset
```bash
python src/check_dataset.py
```

### 2. Build Standardized Dataset
```bash
python src/build_dataset.py
```

### 3. Train Baseline CNN Model
```bash
python src/train_baseline.py
```

### 4. Evaluate Model & Generate Reports
```bash
python src/evaluate.py
```

### 5. Run Offline Audio Prediction
```bash
python src/predict.py --file path/to/audio.wav
```

### 6. Interactive Microphone Recorder
```bash
python src/recorder.py --list-devices
python src/recorder.py --duration 5 --output recordings/test_step.wav
```

### 7. Real-Time Live Detection
```bash
python src/live_detection.py
```

### 8. Real-Time Web Dashboard & API
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
