# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

DriveSense is a real-time driver drowsiness detection system using a webcam, MediaPipe facial landmarks, and a scikit-learn Random Forest classifier. It classifies three states — **AWAKE**, **DROWSY**, **YAWNING** — and also detects head-pose distraction, gaze deviation, blink-rate anomalies, and micro-sleep events. A Flask web dashboard (`src/app.py`) provides a live camera feed, real-time biometric charts, per-driver calibration profiles, and a rolling fatigue score.

## Dependencies

Python with dependencies managed in [requirements.txt](file:///Users/davis/Documents/GitHub/DriveSense/requirements.txt):
- opencv-python, mediapipe, pandas, scikit-learn, matplotlib, joblib, sounddevice, numpy, scipy, Flask

Set up a virtual environment and install everything:
```bash
./setup.sh
```
Or manually:
```bash
pip install -r requirements.txt
```

## Key Commands

All scripts must be run from the project root (paths are relative to it).

```bash
# 1. Collect training data via webcam → data/drowsiness_dataset.csv
python src/save_dataset.py

# 2. Inspect the collected dataset
python src/check_dataset.py

# 3. Train the Random Forest model (temporal split) → models_ml/drowsiness_model.pkl
python src/train_model.py

# 4. Evaluate the trained model (hold-out + cross-validation) → results/
python src/evaluate_model.py

# 5a. Live detection — OpenCV window
python src/live_ml_app.py

# 5b. Live detection — web dashboard, http://127.0.0.1:5001
python src/app.py

# 6. Compare rule-based vs ML detection
python src/compare_models.py

# 7. Rule-based detection only (no ML, face mesh overlay)
python src/drowsiness_warning.py
```

Press **q** to quit any OpenCV webcam window.

## Architecture

The pipeline has two phases:

1. **Data collection & training** — `save_dataset.py` runs the webcam, computes features per frame using rule-based thresholds, labels each frame, and appends rows to `data/drowsiness_dataset.csv`. `train_model.py` trains a Random Forest on that CSV using a temporal split (first 80% train, last 20% test) to avoid leakage from sequential frame counters.

2. **Live inference** — `live_ml_app.py` (OpenCV window) and `app.py` (Flask dashboard) compute the same features in real time and feed them to the trained model for prediction, augmented by rule-based checks for head pose, gaze, and micro-sleep.

`app.py` persists state across restarts via two helper modules:
- **`db.py`** — SQLite event log at `data/events.db`. `log_event()` records each non-AWAKE status transition; `get_recent_events()` / `get_event_counts()` back the `/api/events_log` and `/api/session_summary` endpoints.
- **`profiles.py`** — per-driver calibration storage at `data/profiles/{name}.json`. `save_threshold()` / `load_threshold()` persist the EAR threshold from calibration so it's restored when a driver is reselected via `/api/set_driver`.

`app.py` also maintains a **fatigue score** (0–100): a rolling 5-minute weighted sum of DROWSY/YAWNING/MICRO-SLEEP/DISTRACTED events, exposed via the `/events` SSE stream as `fatigue_score`.

### Feature extraction (`src/features.py`, shared across all scripts)

- **Eye Aspect Ratio (EAR)**: 6 MediaPipe landmark indices per eye, ratio of vertical to horizontal distances. Low EAR = closed.
- **Mouth Open Ratio**: vertical/horizontal distance ratio from 4 mouth landmarks. High ratio = open/yawning.
- **Frame counters**: `closed_eye_frames` and `open_mouth_frames` count consecutive frames above/below threshold — fed to the model as features.
- **Iris ratios**: `iris_h_ratio` / `iris_v_ratio` from iris centers, used for gaze detection in `app.py`.

### Important landmark indices

- Left eye: `[33, 160, 158, 133, 153, 144]`
- Right eye: `[362, 385, 387, 263, 373, 380]`
- Mouth: `[13, 14, 78, 308]`
- Iris centers: `468` (left), `473` (right)

### Thresholds

`save_dataset.py` and `live_ml_app.py` use: EAR ≤ 0.40 (closed), mouth > 0.10 (open).
`drowsiness_warning.py` uses stricter thresholds: EAR < 0.20, mouth > 0.07, plus higher frame counts (10/25 vs 1/8).
`app.py` starts with EAR 0.40 / mouth 0.10, adjustable per-driver via eye calibration (`/api/calibrate`), persisted through `profiles.py`.

### Models

- `models/face_landmarker.task` — MediaPipe face landmarker model (binary, tracked via Git LFS / .gitattributes)
- `models_ml/drowsiness_model.pkl` — trained Random Forest classifier

## Web Dashboard API (`app.py`, port 5001)

| Route | Method | Description |
|---|---|---|
| `/` | GET | Dashboard page |
| `/video_feed` | GET | MJPEG annotated camera stream |
| `/events` | GET | SSE stream of live state (biometrics, status, fatigue score, events) |
| `/api/toggle_alert` | POST | Enable/disable the audible alert |
| `/api/set_threshold` | POST | Manually set EAR/mouth thresholds |
| `/api/calibrate` | POST | Run eye calibration, persist threshold to active driver's profile |
| `/api/config` | GET | Current alert/EAR/mouth thresholds |
| `/api/events_log` | GET | Persisted event history from `data/events.db` (`?limit=`, max 200) |
| `/api/session_summary` | GET | Event counts grouped by type for the last hour |
| `/api/set_driver` | POST | Switch active driver, restoring saved EAR threshold (`{"driver": "Davis"}`) |
| `/api/drivers` | GET | List driver names with saved calibration profiles |

## Additional Scripts

- `src/webcam_test.py`, `src/landmarks_test.py`, `src/feature_test.py` — standalone test/debug scripts for webcam, landmarks, and feature visualization
- `src/clean_warning.py`, `src/clean_warning2.py` — earlier iterations of the rule-based warning system
