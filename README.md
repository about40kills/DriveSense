# Driver Drowsiness Detection System

## Project Overview
This project detects driver drowsiness using a webcam, MediaPipe facial landmarks, and machine learning. The system tracks eye openness and mouth openness to identify three states: AWAKE, DROWSY, and YAWNING.

## Objective
The goal is to build a real-time prototype that can warn when a driver shows signs of fatigue, such as prolonged eye closure or yawning.

## Tools Used
- Python
- OpenCV
- MediaPipe
- Pandas
- Scikit-learn
- Matplotlib

## Main Features
- Real-time webcam detection
- Eye Aspect Ratio calculation
- Mouth opening ratio calculation
- Drowsiness and yawning detection
- Dataset saving into CSV
- Random Forest machine learning model
- Evaluation using accuracy, precision, recall, and confusion matrix

## Project Structure
```text
drowsiness_project/
├── data/
│   └── drowsiness_dataset.csv
├── models/
│   └── face_landmarker.task
├── models_ml/
│   └── drowsiness_model.pkl
├── results/
│   ├── evaluation_report.txt
│   └── confusion_matrix.png
├── src/
│   ├── save_dataset.py
│   ├── check_dataset.py
│   ├── train_model.py
│   ├── live_ml_app.py
│   └── evaluate_model.py
└── README.md