import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

# ---------- Landmark index groups ----------
# These are common face mesh landmark indices for eyes and mouth.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# ---------- Helper functions ----------
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

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

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

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

    # Mirror view
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        h, w, _ = frame.shape
        landmarks = result.face_landmarks[0]

        # Draw all landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Calculate features
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        mouth_ratio = mouth_open_ratio(landmarks, MOUTH, w, h)

        # Show values
        cv2.putText(frame, f"Left Eye: {left_ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Right Eye: {right_ear:.3f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Mouth: {mouth_ratio:.3f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Feature Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()