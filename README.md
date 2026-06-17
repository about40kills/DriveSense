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

---

## Architecture

The system has two phases:

### Phase 1 — Data Collection & Training
`save_dataset.py` runs the webcam, computes per-frame features using rule-based thresholds, labels each frame, and appends rows to `data/drowsiness_dataset.csv`. `train_model.py` then trains a Random Forest on that CSV using a **temporal split** (first 80% of rows for training, last 20% for testing) to prevent data leakage from the sequential frame counters.

### Phase 2 — Live Inference
`live_ml_app.py` (terminal/OpenCV window) and `app.py` (web dashboard) compute the same features in real-time and pass them to the trained model for prediction, augmented by rule-based checks for head pose, gaze, and micro-sleep.

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

# 5b. Run live detection — web dashboard (open http://127.0.0.1:5000)
python src/app.py

# 6. Compare rule-based vs ML detection
python src/compare_models.py

# 7. Rule-based detection only (no ML, face mesh overlay)
python src/drowsiness_warning.py
```

Press **q** to quit any webcam window.

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
