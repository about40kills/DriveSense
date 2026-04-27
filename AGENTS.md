# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

DriveSense is a real-time driver drowsiness detection system using webcam input, MediaPipe facial landmarks, and a scikit-learn Random Forest classifier. It classifies three states: **AWAKE**, **DROWSY**, and **YAWNING** based on Eye Aspect Ratio (EAR) and Mouth Open Ratio features.

## Dependencies

Python with: opencv-python, mediapipe, pandas, scikit-learn, matplotlib, joblib

No requirements.txt exists yet — install manually:
```
pip install opencv-python mediapipe pandas scikit-learn matplotlib joblib
```

## Key Commands

All scripts must be run from the project root (paths are relative to it).

```bash
# 1. Collect training data via webcam (writes to data/drowsiness_dataset.csv)
python src/save_dataset.py

# 2. Inspect the collected dataset
python src/check_dataset.py

# 3. Train the Random Forest model (saves to models_ml/drowsiness_model.pkl)
python src/train_model.py

# 4. Evaluate the trained model (saves report + confusion matrix to results/)
python src/evaluate_model.py

# 5. Run live detection using the trained ML model
python src/live_ml_app.py

# 6. Run rule-based detection (no ML, uses hardcoded thresholds + face mesh overlay)
python src/drowsiness_warning.py
```

Press **q** to quit any webcam window.

## Architecture

The pipeline has two phases:

1. **Data collection & training** — `save_dataset.py` runs the webcam, computes features per frame using rule-based thresholds, labels each frame, and appends rows to `data/drowsiness_dataset.csv`. Then `train_model.py` trains a Random Forest on that CSV.

2. **Live inference** — `live_ml_app.py` computes the same features in real-time and feeds them to the trained model for prediction instead of using thresholds.

### Feature extraction (shared across scripts, duplicated — not in a shared module)

- **Eye Aspect Ratio (EAR)**: Computed from 6 MediaPipe landmark indices per eye. Ratio of vertical to horizontal distances. Low EAR = eyes closed.
- **Mouth Open Ratio**: Vertical/horizontal distance ratio from 4 mouth landmarks. High ratio = mouth open.
- **Frame counters**: `closed_eye_frames` and `open_mouth_frames` count consecutive frames above/below thresholds — these are features fed to the model.

### Important landmark indices

- Left eye: `[33, 160, 158, 133, 153, 144]`
- Right eye: `[362, 385, 387, 263, 373, 380]`
- Mouth: `[13, 14, 78, 308]`

### Thresholds

`save_dataset.py` and `live_ml_app.py` use: EAR <= 0.40 (closed), mouth > 0.10 (open).
`drowsiness_warning.py` uses stricter thresholds: EAR < 0.20, mouth > 0.07, plus higher frame counts (10/25 vs 1/8).

### Models

- `models/face_landmarker.task` — MediaPipe face landmarker model (binary, tracked via Git LFS / .gitattributes)
- `models_ml/drowsiness_model.pkl` — Trained Random Forest classifier

## Additional Scripts

- `src/webcam_test.py`, `src/landmarks_test.py`, `src/feature_test.py` — standalone test/debug scripts for webcam, landmarks, and feature visualization
- `src/clean_warning.py`, `src/clean_warning2.py` — earlier iterations of the rule-based warning system
