import cv2
import csv
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import LEFT_EYE, RIGHT_EYE, MOUTH, eye_aspect_ratio, mouth_open_ratio

MODEL_PATH = "models/face_landmarker.task"
CSV_PATH = "data/drowsiness_dataset.csv"

EYE_CLOSED_THRESHOLD = 0.40
MOUTH_OPEN_THRESHOLD = 0.10
CLOSED_FRAMES_THRESHOLD = 1
YAWN_FRAMES_THRESHOLD = 8

closed_eye_frames = 0
open_mouth_frames = 0

# ---------- Create data folder ----------
os.makedirs("data", exist_ok=True)

# ---------- Create CSV if not exists ----------
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "left_ear",
            "right_ear",
            "avg_ear",
            "mouth_ratio",
            "closed_eye_frames",
            "open_mouth_frames",
            "status"
        ])

# ---------- MediaPipe setup ----------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    status_text = "AWAKE"
    status_color = (0, 255, 0)

    left_ear = 0.0
    right_ear = 0.0
    avg_ear = 0.0
    mouth_ratio = 0.0

    if result.face_landmarks:
        h, w, _ = frame.shape
        landmarks = result.face_landmarks[0]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        mouth_ratio = mouth_open_ratio(landmarks, MOUTH, w, h)

        if avg_ear <= EYE_CLOSED_THRESHOLD:
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        if mouth_ratio > MOUTH_OPEN_THRESHOLD:
            open_mouth_frames += 1
        else:
            open_mouth_frames = 0

        # YAWNING gets priority so drowsy does not hide it
        if open_mouth_frames >= YAWN_FRAMES_THRESHOLD:
            status_text = "YAWNING"
            status_color = (0, 165, 255)
        elif closed_eye_frames >= CLOSED_FRAMES_THRESHOLD:
            status_text = "DROWSY"
            status_color = (0, 0, 255)

    else:
        status_text = "NO FACE"
        status_color = (0, 0, 255)
        closed_eye_frames = 0
        open_mouth_frames = 0

    # ---------- Save one row to CSV ----------
    with open(CSV_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            time.time(),
            left_ear,
            right_ear,
            avg_ear,
            mouth_ratio,
            closed_eye_frames,
            open_mouth_frames,
            status_text
        ])

    # ---------- Show values ----------
    cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Avg EAR: {avg_ear:.3f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Mouth: {mouth_ratio:.3f}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Closed Frames: {closed_eye_frames}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Open Mouth Frames: {open_mouth_frames}", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Status: {status_text}", (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    cv2.imshow("Save Drowsiness Dataset", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()