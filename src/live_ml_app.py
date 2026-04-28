import cv2
import math
import joblib
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import subprocess
from collections import deque

MODEL_PATH = "models/face_landmarker.task"
ML_MODEL_PATH = "models_ml/drowsiness_model.pkl"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# Iris landmark centers (indices 468-477 in the 478-landmark model)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# ── State variables ─────────────────────────────────────────────────────────
closed_eye_frames = 0
open_mouth_frames = 0
distracted_frames = 0
last_beep_time = 0.0

# Blink & micro-sleep tracking
blink_timestamps: deque = deque()
micro_sleep_timestamps: deque = deque()
was_eye_closed = False
closure_frame_count = 0

# EAR threshold for blink/micro-sleep detection (separate from ML feature threshold)
BLINK_EAR = 0.25
BLINK_WINDOW = 60        # seconds for blink rate window
MICRO_MIN_FRAMES = 3     # shortest closure counted as micro-sleep (not a normal blink)
MICRO_MAX_FRAMES = 15    # longest micro-sleep before the ML model handles it as DROWSY
MICRO_WINDOW = 60        # seconds
MICRO_ALERT_COUNT = 3    # micro-sleeps in window before alert

# Gaze tracking
gaze_off_frames = 0
GAZE_H_THRESHOLD = 0.28  # iris deviation from center (horizontal)
GAZE_V_DOWN_THRESHOLD = 0.70  # iris below eye centre (looking down, e.g. at phone)


def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    vertical1 = euclidean_distance(p[1], p[4])
    vertical2 = euclidean_distance(p[2], p[5])
    horizontal = euclidean_distance(p[0], p[3])
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_open_ratio(landmarks, mouth_indices, w, h):
    top    = (landmarks[mouth_indices[0]].x * w, landmarks[mouth_indices[0]].y * h)
    bottom = (landmarks[mouth_indices[1]].x * w, landmarks[mouth_indices[1]].y * h)
    left   = (landmarks[mouth_indices[2]].x * w, landmarks[mouth_indices[2]].y * h)
    right  = (landmarks[mouth_indices[3]].x * w, landmarks[mouth_indices[3]].y * h)
    horizontal = euclidean_distance(left, right)
    if horizontal == 0:
        return 0.0
    return euclidean_distance(top, bottom) / horizontal


def iris_h_ratio(landmarks, eye_indices, iris_idx, w, h):
    """Horizontal iris position: 0.0 = full left, 0.5 = centre, 1.0 = full right."""
    iris_x = landmarks[iris_idx].x * w
    left_x = landmarks[eye_indices[0]].x * w
    right_x = landmarks[eye_indices[3]].x * w
    span = right_x - left_x
    if abs(span) < 1:
        return 0.5
    return (iris_x - left_x) / span


def iris_v_ratio(landmarks, eye_indices, iris_idx, w, h):
    """Vertical iris position: 0.0 = full up, 0.5 = centre, 1.0 = full down."""
    iris_y = landmarks[iris_idx].y * h
    top_y = (landmarks[eye_indices[1]].y + landmarks[eye_indices[2]].y) / 2 * h
    bot_y = (landmarks[eye_indices[4]].y + landmarks[eye_indices[5]].y) / 2 * h
    span = bot_y - top_y
    if abs(span) < 1:
        return 0.5
    return (iris_y - top_y) / span


# ── Load models ──────────────────────────────────────────────────────────────
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

# ── Main loop ────────────────────────────────────────────────────────────────
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

    # ── Per-frame values ─────────────────────────────────────────────────────
    status_text = "NO FACE"
    status_color = (0, 0, 255)
    left_ear = right_ear = avg_ear = mouth_ratio = 0.0
    pitch = yaw = roll = 0.0
    is_distracted = is_gaze_distracted = False
    left_h = left_v = right_h = right_v = 0.5
    iris_available = False

    now = time.time()

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        mouth_ratio = mouth_open_ratio(landmarks, MOUTH, w, h)

        # ── ML features (same order as training) ──────────────────────────
        if avg_ear <= 0.40:
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        if mouth_ratio > 0.10:
            open_mouth_frames += 1
        else:
            open_mouth_frames = 0

        features = pd.DataFrame([{
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "mouth_ratio": mouth_ratio,
            "closed_eye_frames": closed_eye_frames,
            "open_mouth_frames": open_mouth_frames,
        }])
        prediction = model.predict(features)[0]

        # ── Head-pose distraction ─────────────────────────────────────────
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

        # ── Iris gaze tracking ────────────────────────────────────────────
        if len(landmarks) > RIGHT_IRIS_CENTER and avg_ear > BLINK_EAR:
            left_h  = iris_h_ratio(landmarks, LEFT_EYE,  LEFT_IRIS_CENTER,  w, h)
            left_v  = iris_v_ratio(landmarks, LEFT_EYE,  LEFT_IRIS_CENTER,  w, h)
            right_h = iris_h_ratio(landmarks, RIGHT_EYE, RIGHT_IRIS_CENTER, w, h)
            right_v = iris_v_ratio(landmarks, RIGHT_EYE, RIGHT_IRIS_CENTER, w, h)
            iris_available = True

            avg_h = (left_h + right_h) / 2.0
            avg_v = (left_v + right_v) / 2.0

            # Flag gaze off if iris is displaced horizontally OR looking down
            gaze_off = (
                abs(avg_h - 0.5) > GAZE_H_THRESHOLD or
                avg_v > GAZE_V_DOWN_THRESHOLD
            )
            gaze_off_frames = gaze_off_frames + 1 if gaze_off else 0
            is_gaze_distracted = gaze_off_frames > 20
        else:
            gaze_off_frames = 0

        # ── Blink & micro-sleep state machine ─────────────────────────────
        eye_now_closed = avg_ear < BLINK_EAR

        if eye_now_closed:
            closure_frame_count += 1
        else:
            if was_eye_closed and closure_frame_count >= 1:
                # Eye just re-opened — evaluate the completed closure
                blink_timestamps.append(now)

                if MICRO_MIN_FRAMES <= closure_frame_count <= MICRO_MAX_FRAMES:
                    micro_sleep_timestamps.append(now)

            closure_frame_count = 0

        was_eye_closed = eye_now_closed

        # Expire old entries outside the sliding windows
        while blink_timestamps and now - blink_timestamps[0] > BLINK_WINDOW:
            blink_timestamps.popleft()
        while micro_sleep_timestamps and now - micro_sleep_timestamps[0] > MICRO_WINDOW:
            micro_sleep_timestamps.popleft()

        # ── Compose status ────────────────────────────────────────────────
        micro_sleep_count = len(micro_sleep_timestamps)
        blink_rate = len(blink_timestamps)  # blinks per last 60 s

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

    # ── Audible alert ─────────────────────────────────────────────────────────
    alert_needed = (
        is_distracted or
        is_gaze_distracted or
        status_text in ["DROWSY", "YAWNING", "MICRO-SLEEP!", "NO FACE"]
    )
    if alert_needed and (now - last_beep_time > 1.0):
        subprocess.Popen(["afplay", "/System/Library/Sounds/Hero.aiff"])
        last_beep_time = now

    # ── HUD ───────────────────────────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Left column — biometric readings
    cv2.putText(frame, f"Left EAR:  {left_ear:.3f}", (20, 40),  font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (20, 63),  font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Avg EAR:   {avg_ear:.3f}", (20, 86),  font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Mouth:     {mouth_ratio:.3f}", (20, 109), font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 132), font, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Yaw:   {yaw:.1f}",   (20, 155), font, 0.55, (0, 255, 255), 2)

    # Right column — gaze, blink rate, micro-sleep
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
    cv2.putText(frame, gaze_str,                          (right_x, 40),  font, 0.55, gaze_color,  2)
    cv2.putText(frame, f"Blink rate: {blink_rate}/min",   (right_x, 63),  font, 0.55, blink_color, 2)
    cv2.putText(frame, f"Micro-slp:  {micro_sleep_count}", (right_x, 86), font, 0.55, micro_color, 2)

    # Main status banner
    cv2.putText(frame, f"Status: {status_text}", (20, h - 30), font, 1.1, status_color, 3)

    cv2.imshow("DriveSense — Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
