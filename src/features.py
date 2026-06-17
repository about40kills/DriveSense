import math

# ── Landmark index groups ─────────────────────────────────────────────────────
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


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
