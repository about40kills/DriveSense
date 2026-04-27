import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

# ---------- Landmark index groups ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# ---------- Thresholds ----------
# Make drowsiness easier to trigger
EYE_CLOSED_THRESHOLD = 0.20
MOUTH_OPEN_THRESHOLD = 0.07
CLOSED_FRAMES_THRESHOLD = 10
YAWN_FRAMES_THRESHOLD = 25

# ---------- Counters ----------
closed_eye_frames = 0
open_mouth_frames = 0

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

def draw_face_mesh(frame, landmarks, w, h):
    """Draw face mesh with landmarks and connections"""
    # Define face outline connections
    face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 170, 171, 175, 10]
    
    # Draw face outline
    for i in range(len(face_outline) - 1):
        pt1 = landmarks[face_outline[i]]
        pt2 = landmarks[face_outline[i + 1]]
        p1 = (int(pt1.x * w), int(pt1.y * h))
        p2 = (int(pt2.x * w), int(pt2.y * h))
        cv2.line(frame, p1, p2, (0, 255, 0), 1)
    
    # Draw eyes
    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        for i in range(len(eye_indices)):
            pt1 = landmarks[eye_indices[i]]
            pt2 = landmarks[eye_indices[(i + 1) % len(eye_indices)]]
            p1 = (int(pt1.x * w), int(pt1.y * h))
            p2 = (int(pt2.x * w), int(pt2.y * h))
            cv2.line(frame, p1, p2, (0, 255, 255), 1)
    
    # Draw mouth
    for i in range(len(MOUTH)):
        pt1 = landmarks[MOUTH[i]]
        pt2 = landmarks[MOUTH[(i + 1) % len(MOUTH)]]
        p1 = (int(pt1.x * w), int(pt1.y * h))
        p2 = (int(pt2.x * w), int(pt2.y * h))
        cv2.line(frame, p1, p2, (255, 0, 255), 1)
    
    # Draw all landmarks as small circles
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

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

    if result.face_landmarks:
        h, w, _ = frame.shape
        landmarks = result.face_landmarks[0]

        avg_ear = (
            eye_aspect_ratio(landmarks, LEFT_EYE, w, h) +
            eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ) / 2.0

        mouth_ratio = mouth_open_ratio(landmarks, MOUTH, w, h)

        # Draw face mesh
        draw_face_mesh(frame, landmarks, w, h)

        # Eye closure logic
        if avg_ear < EYE_CLOSED_THRESHOLD:
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        # Mouth opening logic
        if mouth_ratio > MOUTH_OPEN_THRESHOLD:
            open_mouth_frames += 1
        else:
            open_mouth_frames = 0

        # Status decision
        if closed_eye_frames >= CLOSED_FRAMES_THRESHOLD:
            status_text = "DROWSY"
            status_color = (0, 0, 255)
        elif open_mouth_frames >= YAWN_FRAMES_THRESHOLD:
            status_text = "YAWNING"
            status_color = (0, 165, 255)

    else:
        status_text = "NO FACE"
        status_color = (0, 0, 255)

    # Show status in top-left corner
    cv2.putText(
        frame,
        f"Status: {status_text}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        3
    )

    cv2.imshow("Drowsiness Warning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()