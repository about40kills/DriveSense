"""
DriveSense — Flask web dashboard.
Run from project root: python src/app.py
"""
import json
import os
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import sounddevice as sd
from flask import Flask, Response, render_template, request, jsonify

from features import (
    LEFT_EYE, RIGHT_EYE, MOUTH, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    eye_aspect_ratio, mouth_open_ratio, iris_h_ratio, iris_v_ratio,
)

# ── Paths (resolved relative to project root) ─────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(_ROOT, "models",    "face_landmarker.task")
ML_MODEL_PATH = os.path.join(_ROOT, "models_ml", "drowsiness_model.pkl")

# ── Alert tone (880 Hz, 0.5 s, cross-platform) ───────────────────────────────
_SR = 44100
_t  = np.linspace(0, 0.5, int(_SR * 0.5), endpoint=False)
_TONE = (0.5 * np.sin(2 * np.pi * 880 * _t)).astype(np.float32)

def _play_alert():
    sd.play(_TONE, _SR, blocking=False)

# ── Shared state ──────────────────────────────────────────────────────────────
_frame_lock = threading.Lock()
_state_lock  = threading.Lock()
_cfg_lock    = threading.Lock()

_output_frame = None   # latest annotated BGR frame for MJPEG

_state = {
    "status":         "LOADING",
    "status_color":   "grey",
    "left_ear":       0.0,
    "right_ear":      0.0,
    "avg_ear":        0.0,
    "mouth_ratio":    0.0,
    "pitch":          0.0,
    "yaw":            0.0,
    "fps":            0.0,
    "blink_rate":     0,
    "micro_sleep_count": 0,
    "gaze":           "--",
    "events":         [],
}

_cfg = {
    "alert_enabled":  True,
    "ear_threshold":  0.40,
    "mouth_threshold": 0.10,
    "calibrating":    False,
    "calib_samples":  [],
}

# ── Detection thread ──────────────────────────────────────────────────────────
def _detection_loop():
    global _output_frame

    ml_model = joblib.load(ML_MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with _state_lock:
            _state["status"] = "NO CAMERA"
        return

    closed_eye_frames = open_mouth_frames = distracted_frames = gaze_off_frames = 0
    last_beep_time = 0.0
    prev_status = ""
    prev_frame_time = time.time()
    was_eye_closed = False
    closure_frame_count = 0

    blink_ts:      deque = deque()
    micro_sleep_ts: deque = deque()

    BLINK_EAR          = 0.25
    BLINK_WINDOW       = 60
    MICRO_MIN_FRAMES   = 3
    MICRO_MAX_FRAMES   = 15
    MICRO_WINDOW       = 60
    MICRO_ALERT_COUNT  = 3
    GAZE_H_THRESH      = 0.28
    GAZE_V_DOWN_THRESH = 0.70

    _STATUS_COLOR = {
        "AWAKE":       "green",
        "DROWSY":      "red",
        "YAWNING":     "orange",
        "DISTRACTED":  "red",
        "MICRO-SLEEP!":"red",
        "NO FACE":     "yellow",
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        now = time.time()
        fps = 1.0 / max(now - prev_frame_time, 1e-6)
        prev_frame_time = now

        with _cfg_lock:
            ear_thresh   = _cfg["ear_threshold"]
            mouth_thresh = _cfg["mouth_threshold"]
            alert_on     = _cfg["alert_enabled"]
            calibrating  = _cfg["calibrating"]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        status_text  = "NO FACE"
        left_ear = right_ear = avg_ear = mouth_ratio = 0.0
        pitch = yaw = 0.0
        is_distracted = is_gaze_distracted = False
        gaze_str = "--"
        micro_sleep_count = len(micro_sleep_ts)
        blink_rate = len(blink_ts)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear    = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
            right_ear   = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear     = (left_ear + right_ear) / 2.0
            mouth_ratio = mouth_open_ratio(landmarks, MOUTH, w, h)

            if calibrating:
                with _cfg_lock:
                    _cfg["calib_samples"].append(avg_ear)

            closed_eye_frames = closed_eye_frames + 1 if avg_ear <= ear_thresh else 0
            open_mouth_frames = open_mouth_frames + 1 if mouth_ratio > mouth_thresh else 0

            features_df = pd.DataFrame([{
                "left_ear": left_ear, "right_ear": right_ear, "avg_ear": avg_ear,
                "mouth_ratio": mouth_ratio,
                "closed_eye_frames": closed_eye_frames,
                "open_mouth_frames": open_mouth_frames,
            }])
            prediction = ml_model.predict(features_df)[0]

            if result.facial_transformation_matrixes:
                rmat = result.facial_transformation_matrixes[0][:3, :3]
                angles, *_ = cv2.RQDecomp3x3(rmat)
                pitch, yaw = angles[0], angles[1]

            distracted_frames = distracted_frames + 1 if (abs(pitch) > 20 or abs(yaw) > 20) else 0
            is_distracted = distracted_frames > 15

            if len(landmarks) > RIGHT_IRIS_CENTER and avg_ear > BLINK_EAR:
                lh = iris_h_ratio(landmarks, LEFT_EYE,  LEFT_IRIS_CENTER,  w, h)
                lv = iris_v_ratio(landmarks, LEFT_EYE,  LEFT_IRIS_CENTER,  w, h)
                rh = iris_h_ratio(landmarks, RIGHT_EYE, RIGHT_IRIS_CENTER, w, h)
                rv = iris_v_ratio(landmarks, RIGHT_EYE, RIGHT_IRIS_CENTER, w, h)
                avg_h, avg_v = (lh + rh) / 2.0, (lv + rv) / 2.0
                gaze_off = abs(avg_h - 0.5) > GAZE_H_THRESH or avg_v > GAZE_V_DOWN_THRESH
                gaze_off_frames = gaze_off_frames + 1 if gaze_off else 0
                is_gaze_distracted = gaze_off_frames > 20
                h_lbl = "LEFT" if avg_h < 0.4 else ("RIGHT" if avg_h > 0.6 else "CENTER")
                v_lbl = "UP"   if avg_v < 0.35 else ("DOWN"  if avg_v > 0.65 else "CENTER")
                gaze_str = f"{h_lbl} / {v_lbl}"
            else:
                gaze_off_frames = 0

            eye_closed = avg_ear < BLINK_EAR
            if eye_closed:
                closure_frame_count += 1
            else:
                if was_eye_closed and closure_frame_count >= 1:
                    blink_ts.append(now)
                    if MICRO_MIN_FRAMES <= closure_frame_count <= MICRO_MAX_FRAMES:
                        micro_sleep_ts.append(now)
                closure_frame_count = 0
            was_eye_closed = eye_closed

            while blink_ts     and now - blink_ts[0]      > BLINK_WINDOW: blink_ts.popleft()
            while micro_sleep_ts and now - micro_sleep_ts[0] > MICRO_WINDOW: micro_sleep_ts.popleft()

            micro_sleep_count = len(micro_sleep_ts)
            blink_rate        = len(blink_ts)

            if is_distracted or is_gaze_distracted:
                status_text = "DISTRACTED"
            elif micro_sleep_count >= MICRO_ALERT_COUNT:
                status_text = "MICRO-SLEEP!"
            else:
                status_text = prediction

        else:
            closed_eye_frames = open_mouth_frames = distracted_frames = gaze_off_frames = 0
            closure_frame_count = 0
            was_eye_closed = False

        sc = _STATUS_COLOR.get(status_text, "grey")

        if alert_on and status_text in ("DROWSY","YAWNING","MICRO-SLEEP!","DISTRACTED","NO FACE"):
            if now - last_beep_time > 1.0:
                _play_alert()
                last_beep_time = now

        with _state_lock:
            events = _state["events"]
            if status_text != prev_status and status_text != "AWAKE":
                events.insert(0, {
                    "time":  datetime.now().strftime("%H:%M:%S"),
                    "event": status_text,
                    "color": sc,
                })
                del events[30:]

            _state.update({
                "status": status_text, "status_color": sc,
                "left_ear": round(left_ear, 3), "right_ear": round(right_ear, 3),
                "avg_ear":  round(avg_ear,  3), "mouth_ratio": round(mouth_ratio, 3),
                "pitch": round(pitch, 1), "yaw": round(yaw, 1),
                "fps": round(fps, 1),
                "blink_rate": blink_rate, "micro_sleep_count": micro_sleep_count,
                "gaze": gaze_str, "events": events,
            })

        prev_status = status_text

        # Minimal frame annotation (details shown in dashboard)
        ov_color = {"green":(0,255,136),"red":(68,68,255),"orange":(0,165,255),"yellow":(0,220,255)}.get(sc,(200,200,200))
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 2)
        cv2.putText(frame, status_text,     (20, h - 25),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, ov_color, 3)

        with _frame_lock:
            _output_frame = frame.copy()

    cap.release()


# ── MJPEG generator ───────────────────────────────────────────────────────────
def _gen_mjpeg():
    while True:
        with _frame_lock:
            frame = _output_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.033)


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(_gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/events")
def events():
    def _stream():
        while True:
            with _state_lock:
                data = json.dumps(_state)
            yield f"data: {data}\n\n"
            time.sleep(0.2)
    return Response(_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/toggle_alert", methods=["POST"])
def toggle_alert():
    with _cfg_lock:
        _cfg["alert_enabled"] = not _cfg["alert_enabled"]
        return jsonify({"alert_enabled": _cfg["alert_enabled"]})

@app.route("/api/set_threshold", methods=["POST"])
def set_threshold():
    body = request.get_json()
    with _cfg_lock:
        if "ear_threshold"   in body: _cfg["ear_threshold"]   = float(body["ear_threshold"])
        if "mouth_threshold" in body: _cfg["mouth_threshold"] = float(body["mouth_threshold"])
    return jsonify({"ok": True})

@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    with _cfg_lock:
        _cfg["calibrating"]   = True
        _cfg["calib_samples"] = []
    time.sleep(3)
    with _cfg_lock:
        _cfg["calibrating"] = False
        samples = _cfg["calib_samples"]
        if samples:
            _cfg["ear_threshold"] = round(sum(samples) / len(samples) * 0.6, 3)
        new_thresh = _cfg["ear_threshold"]
    return jsonify({"ear_threshold": new_thresh})

@app.route("/api/config")
def get_config():
    with _cfg_lock:
        return jsonify({k: _cfg[k] for k in ("alert_enabled","ear_threshold","mouth_threshold")})


if __name__ == "__main__":
    t = threading.Thread(target=_detection_loop, daemon=True)
    t.start()
    print("DriveSense dashboard → http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
