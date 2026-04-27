import cv2
import math
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"
ML_MODEL_PATH = "models_ml/drowsiness_model.pkl"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

closed_eye_frames = 0
open_mouth_frames = 0

def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    p1 = (landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h)
    p2 = (landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h)
    p3 = (landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h)
    p4 = (landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h)
    p5 = (landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h)
    p6 = (landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h)

    vertical1 = euclidean_distance(p2, p5)
    vertical2 = euclidean_distance(p3, p6)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)

def mouth_open_ratio(landmarks, mouth_indices, w, h):
    top = (landmarks[mouth_indices[0]].x * w, landmarks[mouth_indices[0]].y * h)
    bottom = (landmarks[mouth_indices[1]].x * w, landmarks[mouth_indices[1]].y * h)
    left = (landmarks[mouth_indices[2]].x * w, landmarks[mouth_indices[2]].y * h)
    right = (landmarks[mouth_indices[3]].x * w, landmarks[mouth_indices[3]].y * h)

    vertical = euclidean_distance(top, bottom)
    horizontal = euclidean_distance(left, right)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal

# Load trained ML model
model = joblib.load(ML_MODEL_PATH)

# MediaPipe setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

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

    status_text = "NO FACE"
    status_color = (0, 0, 255)

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

        # Same counters as training data
        if avg_ear <= 0.40:
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        if mouth_ratio > 0.10:
            open_mouth_frames += 1
        else:
            open_mouth_frames = 0

        # Feature order must match training
        features = [[
            left_ear,
            right_ear,
            avg_ear,
            mouth_ratio,
            closed_eye_frames,
            open_mouth_frames
        ]]

        prediction = model.predict(features)[0]
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
        closed_eye_frames = 0
        open_mouth_frames = 0

    cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Avg EAR: {avg_ear:.3f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Mouth: {mouth_ratio:.3f}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"ML Status: {status_text}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    cv2.imshow("Live ML Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()