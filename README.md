<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/drivesense-logo-dark.svg">
    <img src="assets/branding/drivesense-logo.svg" alt="DriveSense" width="420">
  </picture>
</p>

# DriveSense — Real-Time Driver Drowsiness Detection

DriveSense is a real-time driver drowsiness detection system that uses a standard webcam, MediaPipe facial landmarks, and a scikit-learn Random Forest classifier to continuously monitor driver alertness and trigger audible alerts before fatigue becomes dangerous.

The system classifies three driver states — **AWAKE**, **DROWSY**, and **YAWNING** — and additionally detects **head-pose distraction**, **gaze deviation**, **blink rate anomalies**, and **micro-sleep events**. A web-based dashboard provides a live camera feed, real-time biometric charts, and adjustable detection controls.

---

## Key Features

- **ML-based classification** — Random Forest trained on Eye Aspect Ratio (EAR) and Mouth Open Ratio features; 99.6% hold-out accuracy, 97.9% ± 2.2% cross-validated
- **Head-pose distraction detection** — flags excessive pitch/yaw via facial transformation matrix
- **Gaze tracking** — iris position monitoring detects looking left, right, or down (e.g. at phone)
- **Blink rate monitoring** — sliding 60-second window; abnormal rates flagged
- **Micro-sleep detection** — tracks rapid repeated eye closures (3–15 frames) distinct from blinks
- **Cross-platform audible alert** — synthesised 880 Hz tone via `sounddevice` (works on macOS, Windows, Linux)
- **Web dashboard** — live MJPEG video stream, SSE-driven biometric metrics, Chart.js EAR/mouth timeline, adjustable thresholds, eye calibration
- **Eye calibration** — 3-second baseline measurement auto-sets the EAR threshold for the current driver
- **Driver profiles** — calibrated EAR thresholds persist per-driver as JSON under `data/profiles/`, reloaded on driver switch
- **Fatigue score** — rolling 5-minute weighted score (0–100) computed from DROWSY/YAWNING/MICRO-SLEEP/DISTRACTED event frequency
- **Event persistence** — non-AWAKE events are logged to `data/events.db` (SQLite) so history survives app restarts

---

## Architecture

The system has two phases:

### Phase 1 — Data Collection & Training
`save_dataset.py` runs the webcam, computes per-frame features using rule-based thresholds, labels each frame, and appends rows to `data/drowsiness_dataset.csv`. `train_model.py` then trains a Random Forest on that CSV using a **temporal split** (first 80% of rows for training, last 20% for testing) to prevent data leakage from the sequential frame counters.

### Phase 2 — Live Inference
`live_ml_app.py` (terminal/OpenCV window) and `app.py` (web dashboard) compute the same features in real-time and pass them to the trained model for prediction, augmented by rule-based checks for head pose, gaze, and micro-sleep.

`app.py` additionally persists state across restarts via two helper modules:
- **`db.py`** — SQLite event log (`data/events.db`); `log_event()` records each non-AWAKE transition, `get_recent_events()` / `get_event_counts()` back the `/api/events_log` and `/api/session_summary` endpoints.
- **`profiles.py`** — per-driver calibration storage (`data/profiles/{name}.json`); `save_threshold()` / `load_threshold()` persist the EAR threshold set during calibration so it's restored when a driver is selected again via `/api/set_driver`.

### Feature Extraction (`src/features.py`)
All scripts share a single feature module:

| Feature | Description |
|---|---|
| `left_ear`, `right_ear`, `avg_ear` | Eye Aspect Ratio — ratio of vertical to horizontal eye distances. Low = closed. |
| `mouth_ratio` | Vertical/horizontal mouth distance ratio. High = open/yawning. |
| `closed_eye_frames` | Consecutive frames where EAR ≤ threshold |
| `open_mouth_frames` | Consecutive frames where mouth ratio > threshold |

**Landmark indices:**
- Left eye: `[33, 160, 158, 133, 153, 144]`
- Right eye: `[362, 385, 387, 263, 373, 380]`
- Mouth: `[13, 14, 78, 308]`
- Iris centres: `468` (left), `473` (right)

---

## Project Structure

```
DriveSense/
├── data/
│   └── drowsiness_dataset.csv       # Collected training data (4,742 rows)
├── models/
│   └── face_landmarker.task         # MediaPipe face landmark model
├── models_ml/
│   └── drowsiness_model.pkl         # Trained Random Forest classifier
├── results/
│   ├── evaluation_report.txt        # Accuracy, classification report, CV scores
│   └── confusion_matrix.png         # Confusion matrix plot
├── src/
│   ├── features.py                  # Shared feature extraction functions
│   ├── save_dataset.py              # Webcam data collection
│   ├── check_dataset.py             # Dataset inspection
│   ├── train_model.py               # Model training (temporal split)
│   ├── evaluate_model.py            # Hold-out + cross-validation evaluation
│   ├── compare_models.py            # Rule-based vs ML comparison
│   ├── live_ml_app.py               # Live detection (OpenCV window)
│   ├── app.py                       # Live detection (web dashboard)
│   ├── db.py                        # SQLite event log (data/events.db)
│   ├── profiles.py                  # Per-driver calibration profiles (data/profiles/)
│   ├── drowsiness_warning.py        # Rule-based detection (face mesh overlay)
│   └── templates/
│       └── index.html               # Web dashboard frontend
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Setup

### Automated (recommended)
```bash
./setup.sh
```

### Manual
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

All scripts must be run from the **project root**.

```bash
# 1. Collect training data via webcam → data/drowsiness_dataset.csv
python src/save_dataset.py

# 2. Inspect the collected dataset
python src/check_dataset.py

# 3. Train the Random Forest model → models_ml/drowsiness_model.pkl
python src/train_model.py

# 4. Evaluate the trained model → results/
python src/evaluate_model.py

# 5a. Run live detection — OpenCV window
python src/live_ml_app.py

# 5b. Run live detection — web dashboard (open http://127.0.0.1:5001)
python src/app.py

# 6. Compare rule-based vs ML detection
python src/compare_models.py

# 7. Rule-based detection only (no ML, face mesh overlay)
python src/drowsiness_warning.py
```

Press **q** to quit any webcam window.

---

## Dashboard API (`app.py`)

| Route | Method | Description |
|---|---|---|
| `/` | GET | Dashboard page |
| `/video_feed` | GET | MJPEG annotated camera stream |
| `/events` | GET | SSE stream of live state (biometrics, status, fatigue score, events) |
| `/api/toggle_alert` | POST | Enable/disable the audible alert |
| `/api/set_threshold` | POST | Manually set the EAR/mouth thresholds |
| `/api/calibrate` | POST | Run eye calibration, persist threshold to active driver's profile |
| `/api/config` | GET | Current alert/EAR/mouth thresholds |
| `/api/events_log` | GET | Persisted event history from `data/events.db` (`?limit=`, max 200) |
| `/api/session_summary` | GET | Event counts grouped by type for the last hour |
| `/api/set_driver` | POST | Switch active driver, restoring their saved EAR threshold (`{"driver": "Davis"}`) |
| `/api/drivers` | GET | List driver names with saved calibration profiles |

---

## Results

| Metric | Value |
|---|---|
| Temporal hold-out accuracy | **99.6%** |
| 5-fold cross-validation accuracy | **97.9% ± 2.2%** |
| Classes | AWAKE, DROWSY, YAWNING |
| Training samples | 3,034 |
| Test samples | 759 |

**Per-class performance (test set):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AWAKE | 0.98 | 1.00 | 0.99 |
| DROWSY | 1.00 | 1.00 | 1.00 |
| YAWNING | 1.00 | 0.99 | 0.99 |

---

## Rule-Based vs ML Comparison

Both approaches were evaluated on the same temporal test set (last 20% of collected data, 759 rows).

| Metric | Rule-Based | ML (Random Forest) |
|---|---|---|
| **Overall Accuracy** | 50.1% | **99.6%** |
| AWAKE — F1 | 0.44 | **0.99** |
| DROWSY — F1 | 0.00 | **1.00** |
| YAWNING — F1 | 0.86 | **1.00** |

**Key finding:** The rule-based system completely fails to detect DROWSY (0% recall). Its strict EAR threshold of 0.20 only catches severe eye closure, missing the early-stage drowsiness that the ML model learns to detect from the continuous EAR signal. YAWNING performance is reasonable (86% F1) in the rule-based system because mouth opening is a more binary, visually obvious event.

The ML model's advantage is that it learns the relationship between gradual EAR decline and drowsiness onset, rather than relying on a hard threshold that only triggers at full eye closure.

Full comparison saved to `results/comparison_report.txt`.

---

## Tech Stack

| Component | Library |
|---|---|
| Face landmarks | MediaPipe Face Landmarker |
| ML classifier | scikit-learn RandomForestClassifier |
| Computer vision | OpenCV |
| Web dashboard | Flask (MJPEG + SSE) |
| Frontend chart | Chart.js |
| Audio alert | sounddevice + numpy |
| Data handling | pandas |

---

## Future Work

The current system runs as a desktop app (laptop + webcam), which is fine for development and demoing but not how a driver-monitoring system would actually be used in a vehicle. Planned path to real in-car deployment:

- **Edge hardware** — port the inference pipeline to a Raspberry Pi 5 or NVIDIA Jetson Nano as a self-contained, dashboard- or steering-column-mounted unit with a USB or CSI camera, replacing the laptop-and-webcam setup.
- **Headless operation** — run as a `systemd` service that auto-starts on boot; no monitor or browser needed while driving. The audible alert becomes the primary driver-facing output, with the web dashboard kept around for post-drive review or fleet monitoring rather than in-cabin use.
- **Configurable camera source** — `cv2.VideoCapture(0)` is currently hardcoded across all scripts; a USB camera on embedded hardware won't reliably enumerate as index 0, so this needs to become configurable.
- **Vehicle power** — powered from the car's 12V accessory socket via a buck converter or USB port instead of a laptop battery.
- **Real vehicle validation** — all current testing is stationary; validation with vibration, variable lighting, and genuine fatigue in a moving vehicle is required before any safety-critical use.
