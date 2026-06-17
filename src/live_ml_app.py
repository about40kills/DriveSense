import cv2
import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sounddevice as sd
import time
from collections import deque

from features import (
    LEFT_EYE, RIGHT_EYE, MOUTH, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    eye_aspect_ratio, mouth_open_ratio, iris_h_ratio, iris_v_ratio,
)

MODEL_PATH = "models/face_landmarker.task"
ML_MODEL_PATH = "models_ml/drowsiness_model.pkl"

# ── Alert ─────────────────────────────────────────────────────────────────────
_ALERT_SAMPLE_RATE = 44100
_t = np.linspace(0, 0.5, int(_ALERT_SAMPLE_RATE * 0.5), endpoint=False)
_ALERT_TONE = (0.5 * np.sin(2 * np.pi * 880 * _t)).astype(np.float32)


def play_alert():
    sd.play(_ALERT_TONE, _ALERT_SAMPLE_RATE, blocking=False)


# ── State variables ───────────────────────────────────────────────────────────
closed_eye_frames = 0
open_mouth_frames = 0
distracted_frames = 0
last_beep_time = 0.0

blink_timestamps: deque = deque()
micro_sleep_timestamps: deque = deque()
was_eye_closed = False
closure_frame_count = 0

BLINK_EAR = 0.25
BLINK_WINDOW = 60
MICRO_MIN_FRAMES = 3
MICRO_MAX_FRAMES = 15
MICRO_WINDOW = 60
MICRO_ALERT_COUNT = 3

gaze_off_frames = 0
GAZE_H_THRESHOLD = 0.28
GAZE_V_DOWN_THRESHOLD = 0.70

# ── Load models ───────────────────────────────────────────────────────────────
model = joblib.load(ML_MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

prev_frame_time = time.time()

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    status_text = "NO FACE"
    status_color = (0, 0, 255)
    left_ear = right_ear = avg_ear = mouth_ratio = 0.0
    pitch = yaw = roll = 0.0
    is_distracted = is_gaze_distracted = False
    left_h = left_v = right_h = right_v = 0.5
    iris_available = False

    now = time.time()
    fps = 1.0 / (now - prev_frame_time) if (now - prev_frame_time) > 0 else 0.0
    prev_frame_time = now

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,   w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE,  w, h)
        avg_ear   = (left_ear + right_ear) / 2.0
        mouth_ratio = mouth_open_ratio(landmarks, MOUTH, w, h)

        # ML features
        if avg_ear <= 0.40:
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        if mouth_ratio > 0.10:
            open_mouth_frames += 1
        else:
            open_mouth_frames = 0

        features_df = pd.DataFrame([{
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "mouth_ratio": mouth_ratio,
            "closed_eye_frames": closed_eye_frames,
            "open_mouth_frames": open_mouth_frames,
        }])
        prediction = model.predict(features_df)[0]

        # Head-pose distraction
        if result.facial_transformation_matrixes:
            pose_matrix = result.facial_transformation_matrixes[0]
            rmat = pose_matrix[:3, :3]
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = angles[0], angles[1], angles[2]

        if abs(pitch) > 20 or abs(yaw) > 20:
            distracted_frames += 1
        else:
            distracted_frames = 0
        is_distracted = distracted_frames > 15

        # Iris gaze tracking
        if len(landmarks) > RIGHT_IRIS_CENTER and avg_ear > BLINK_EAR:
            left_h  = iris_h_ratio(landmarks, LEFT_EYE,  LEFT_IRIS_CENTER,  w, h)
            left_v  = iris_v_ratio(landmarks, LEFT_EYE,  LEFT_IRIS_CENTER,  w, h)
            right_h = iris_h_ratio(landmarks, RIGHT_EYE, RIGHT_IRIS_CENTER, w, h)
            right_v = iris_v_ratio(landmarks, RIGHT_EYE, RIGHT_IRIS_CENTER, w, h)
            iris_available = True

            avg_h = (left_h + right_h) / 2.0
            avg_v = (left_v + right_v) / 2.0

            gaze_off = (
                abs(avg_h - 0.5) > GAZE_H_THRESHOLD or
                avg_v > GAZE_V_DOWN_THRESHOLD
            )
            gaze_off_frames = gaze_off_frames + 1 if gaze_off else 0
            is_gaze_distracted = gaze_off_frames > 20
        else:
            gaze_off_frames = 0

        # Blink & micro-sleep state machine
        eye_now_closed = avg_ear < BLINK_EAR
        if eye_now_closed:
            closure_frame_count += 1
        else:
            if was_eye_closed and closure_frame_count >= 1:
                blink_timestamps.append(now)
                if MICRO_MIN_FRAMES <= closure_frame_count <= MICRO_MAX_FRAMES:
                    micro_sleep_timestamps.append(now)
            closure_frame_count = 0
        was_eye_closed = eye_now_closed

        while blink_timestamps and now - blink_timestamps[0] > BLINK_WINDOW:
            blink_timestamps.popleft()
        while micro_sleep_timestamps and now - micro_sleep_timestamps[0] > MICRO_WINDOW:
            micro_sleep_timestamps.popleft()

        micro_sleep_count = len(micro_sleep_timestamps)
        blink_rate = len(blink_timestamps)

        if is_distracted or is_gaze_distracted:
            status_text = "DISTRACTED"
            status_color = (0, 0, 255)
        elif micro_sleep_count >= MICRO_ALERT_COUNT:
            status_text = "MICRO-SLEEP!"
            status_color = (0, 0, 255)
        else:
            status_text = prediction
            if prediction == "AWAKE":
                status_color = (0, 255, 0)
            elif prediction == "DROWSY":
                status_color = (0, 0, 255)
            elif prediction == "YAWNING":
                status_color = (0, 165, 255)
            else:
                status_color = (255, 255, 255)

    else:
        closed_eye_frames = open_mouth_frames = distracted_frames = gaze_off_frames = 0
        closure_frame_count = 0
        was_eye_closed = False
        micro_sleep_count = len(micro_sleep_timestamps)
        blink_rate = len(blink_timestamps)

    # ── Audible alert (cross-platform) ────────────────────────────────────────
    alert_needed = status_text in ["DROWSY", "YAWNING", "MICRO-SLEEP!", "DISTRACTED", "NO FACE"]
    if alert_needed and (now - last_beep_time > 1.0):
        play_alert()
        last_beep_time = now

    # ── HUD ───────────────────────────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f"Left EAR:  {left_ear:.3f}",   (20, 40),  font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Right EAR: {right_ear:.3f}",  (20, 63),  font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Avg EAR:   {avg_ear:.3f}",    (20, 86),  font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Mouth:     {mouth_ratio:.3f}", (20, 109), font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch:.1f}",           (20, 132), font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Yaw:   {yaw:.1f}",             (20, 155), font, 0.55, (0, 255, 255), 2)

    blink_color = (0, 255, 0) if 8 <= blink_rate <= 30 else (0, 0, 255)
    micro_color = (0, 0, 255) if micro_sleep_count >= MICRO_ALERT_COUNT else (0, 255, 255)
    gaze_color  = (0, 0, 255) if is_gaze_distracted else (0, 255, 255)

    if iris_available:
        avg_h_disp = (left_h + right_h) / 2.0
        avg_v_disp = (left_v + right_v) / 2.0
        h_label = "L" if avg_h_disp < 0.4 else ("R" if avg_h_disp > 0.6 else "C")
        v_label = "UP" if avg_v_disp < 0.35 else ("DN" if avg_v_disp > 0.65 else "C")
        gaze_str = f"Gaze: {h_label}/{v_label}"
    else:
        gaze_str = "Gaze: --"

    right_x = w - 220
    cv2.putText(frame, gaze_str,                           (right_x, 40),  font, 0.55, gaze_color,  2)
    cv2.putText(frame, f"Blink rate: {blink_rate}/min",    (right_x, 63),  font, 0.55, blink_color, 2)
    cv2.putText(frame, f"Micro-slp:  {micro_sleep_count}", (right_x, 86),  font, 0.55, micro_color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 60), font, 0.55, (200, 200, 200), 2)
    cv2.putText(frame, f"Status: {status_text}", (20, h - 30), font, 1.1, status_color, 3)

    cv2.imshow("DriveSense — Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
