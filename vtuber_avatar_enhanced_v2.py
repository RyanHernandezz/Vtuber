
import cv2
import numpy as np
import mediapipe as mp
from math import hypot, exp
import threading
import time

# ---------- Config ----------
DRAW_LANDMARKS = True
CAM_INDEX = 0
SMOOTHING_DT = 1.0 / 30.0   # assumed camera fps for Kalman
EMOTION_FPS = 10            # target DeepFace emotion updates per second (10-15 Hz per plan)
USE_DEEPFACE = True         # set False to disable emotion thread if DeepFace unavailable
# ----------------------------

# Tuning parameters for mappings
# EAR normalization
EAR_MIN = 0.12   # lower bound where eye is essentially closed
EAR_MAX = 0.30   # upper bound where eye is fully open

# MAR -> mouth height via sigmoid cap (smaller cap as requested)
MAR_CENTER = 0.35     # midpoint of sigmoid
MAR_GAIN   = 9.0      # steepness
MOUTH_MAX_H = 18      # reduced from 28 to make mouth smaller

# Emotion smoothing
EMA_ALPHA = 0.25      # 0..1, higher tracks faster, lower is smoother
HYSTERESIS_MARGIN = 0.08  # require new emotion prob to exceed current by this margin before switching
MIN_HOLD_SEC = 0.8    # minimum time to hold an emotion before allowing switch

# Eye and mouth landmark indices for Mediapipe FaceMesh (normalized coords)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh

# --------- Small 1D Kalman filter helper (position, velocity) ----------
class KF1D:
    def __init__(self, x0=0.0, v0=0.0, dt=SMOOTHING_DT, q=1e-2, r=1e-2):
        self.dt = dt
        self.x = np.array([[x0], [v0]], dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32) * 1.0
        self.F = np.array([[1, dt],[0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0]], dtype=np.float32)  # measure position only
        self.Q = np.array([[q, 0],[0, q]], dtype=np.float32)
        self.R = np.array([[r]], dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array([[z]], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        return float(self.x[0,0])

# instantiate filters
kf_yaw   = KF1D()
kf_pitch = KF1D()
kf_lear  = KF1D()
kf_rEAR  = KF1D()
kf_mar   = KF1D()

def euclid(p1, p2):
    return hypot(p1[0]-p2[0], p1[1]-p2[1])

def clamp01(x):
    return max(0.0, min(1.0, x))

# Separate EAR for left and right eyes
def ears_from_landmarks(pts):
    def pick(i): return pts[i]
    lv1 = euclid(pick(160), pick(144))
    lv2 = euclid(pick(158), pick(153))
    lh  = euclid(pick(33),  pick(133))
    left = (lv1 + lv2) / (2.0 * lh + 1e-8)
    rv1 = euclid(pick(387), pick(373))
    rv2 = euclid(pick(385), pick(380))
    rh  = euclid(pick(263), pick(362))
    right = (rv1 + rv2) / (2.0 * rh + 1e-8)
    return left, right

def mar_from_landmarks(pts):
    def pick(i): return pts[i]
    v1 = euclid(pick(81),  pick(178))
    v2 = euclid(pick(13),  pick(14))
    v3 = euclid(pick(311), pick(402))
    h  = euclid(pick(78),  pick(308))
    return (v1 + v2 + v3) / (2.0 * h + 1e-8)

def yaw_pitch_proxy(pts):
    NOSE = 1
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    nx, ny = pts[NOSE]
    w = (x_max - x_min) + 1e-8
    h = (y_max - y_min) + 1e-8
    yaw = (nx - cx) / w
    pitch = (cy - ny) / h
    return yaw, pitch

# -------- Emotion detection (DeepFace) thread with smoothing --------
class EmotionThread(threading.Thread):
    EMOS = ['angry','disgust','fear','happy','sad','surprise','neutral']

    def __init__(self, target_fps=EMOTION_FPS):
        super().__init__(daemon=True)
        self.target_dt = 1.0 / max(1, target_fps)
        self.running = True
        self.frame_bgr = None
        self.face_rect = None  # (x,y,w,h)
        self.ready = False
        self.deepface_ok = False

        # smoothed probabilities for hysteresis and stability
        self.smooth = {k: (1.0/len(self.EMOS)) for k in self.EMOS}
        self.curr_label = 'neutral'
        self._last_switch_t = time.time()

        try:
            from deepface import DeepFace  # noqa: F401
            self.deepface_ok = True
        except Exception:
            self.deepface_ok = False

    def set_frame(self, frame_bgr, face_rect):
        self.frame_bgr = frame_bgr
        self.face_rect = face_rect
        self.ready = True

    @staticmethod
    def _normalize_dict(d, keys):
        vals = np.array([max(0.0, float(d.get(k, 0.0))) for k in keys], dtype=np.float32)
        s = float(np.sum(vals))
        if s <= 1e-8:
            vals = np.ones_like(vals) / len(vals)
        else:
            vals /= s
        return {k: float(v) for k, v in zip(keys, vals)}

    def run(self):
        if not self.deepface_ok:
            while self.running:
                time.sleep(self.target_dt)
            return

        from deepface import DeepFace
        while self.running:
            t0 = time.time()
            if self.ready and self.frame_bgr is not None and self.face_rect is not None:
                x,y,w,h = self.face_rect
                x = max(0, x); y = max(0, y)
                crop = self.frame_bgr[y:y+h, x:x+w]
                try:
                    res = DeepFace.analyze(img_path=crop, actions=['emotion'], enforce_detection=False)
                    if isinstance(res, list) and len(res) > 0:
                        res = res[0]
                    emo_dict = res.get('emotion') or res.get('emotions') or {}
                    probs = self._normalize_dict(emo_dict, self.EMOS)

                    # EMA smoothing
                    for k in self.EMOS:
                        self.smooth[k] = EMA_ALPHA * probs[k] + (1.0 - EMA_ALPHA) * self.smooth[k]

                    # hysteresis and hold
                    top_new = max(self.smooth, key=self.smooth.get)
                    p_new = self.smooth[top_new]
                    p_curr = self.smooth.get(self.curr_label, 0.0)
                    now = time.time()
                    hold_ok = (now - self._last_switch_t) >= MIN_HOLD_SEC
                    stronger = (p_new >= p_curr + HYSTERESIS_MARGIN)

                    if top_new != self.curr_label and (stronger or hold_ok):
                        self.curr_label = top_new
                        self._last_switch_t = now

                except Exception:
                    pass

            dt = time.time() - t0
            time.sleep(max(0.0, self.target_dt - dt))

    def stop(self):
        self.running = False

def emotion_to_color(emotion):
    palette = {
        'happy':   (0, 220, 0),
        'sad':     (180, 120, 0),
        'surprise':(0, 180, 220),
        'angry':   (0, 0, 255),
        'fear':    (80, 0, 160),
        'disgust': (0, 120, 120),
        'neutral': (255,255,255),
    }
    return palette.get(emotion, (255,255,255))

def normalize_ear(ear):
    t = (ear - EAR_MIN) / max(1e-6, (EAR_MAX - EAR_MIN))
    return clamp01(t)

def mar_to_mouth_height(mar):
    s = 1.0 / (1.0 + exp(-MAR_GAIN * (mar - MAR_CENTER)))
    return int(MOUTH_MAX_H * s) + 2  # small offset for visibility

def draw_avatar(frame, yaw, pitch, eyeL_open, eyeR_open, mar, emotion='neutral'):
    H, W = frame.shape[:2]
    base_x, base_y = 120, H - 140
    head_r = 50

    offset_x = int(40 * yaw)
    offset_y = int(40 * (-pitch))

    center = (base_x + offset_x, base_y + offset_y)
    color = emotion_to_color(emotion)
    cv2.circle(frame, center, head_r, color, 2)

    eye_h_L = int(8 * eyeL_open) + 1
    eye_h_R = int(8 * eyeR_open) + 1
    eye_w = 10
    eye_dx = 18; eye_dy = -10
    cv2.ellipse(frame, (center[0] - eye_dx, center[1] + eye_dy), (eye_w, eye_h_L), 0, 0, 360, color, 2)
    cv2.ellipse(frame, (center[0] + eye_dx, center[1] + eye_dy), (eye_w, eye_h_R), 0, 0, 360, color, 2)

    mouth_w = 24
    mouth_h = mar_to_mouth_height(mar)
    cv2.ellipse(frame, (center[0], center[1] + 15), (mouth_w, mouth_h), 0, 0, 360, color, 2)

def bbox_from_points(pts_norm, frame_w, frame_h, margin=0.08):
    xs = [p[0] for p in pts_norm]; ys = [p[1] for p in pts_norm]
    x_min = max(0.0, min(xs)); x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys)); y_max = min(1.0, max(ys))
    dx = (x_max - x_min); dy = (y_max - y_min)
    x_min = max(0.0, x_min - margin * dx); x_max = min(1.0, x_max + margin * dx)
    y_min = max(0.0, y_min - margin * dy); y_max = min(1.0, y_max + margin * dy)
    x = int(x_min * frame_w); y = int(y_min * frame_h)
    w = int((x_max - x_min) * frame_w); h = int((y_max - y_min) * frame_h)
    return (x, y, max(1, w), max(1, h))

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    emo_thread = EmotionThread()
    if USE_DEEPFACE:
        emo_thread.start()

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                pts = [(p.x, p.y) for p in lm.landmark]

                raw_lear, raw_rear = ears_from_landmarks(pts)
                raw_mar = mar_from_landmarks(pts)
                raw_yaw, raw_pitch = yaw_pitch_proxy(pts)

                lear  = kf_lear.step(raw_lear)
                rear  = kf_rEAR.step(raw_rear)
                mar   = kf_mar.step(raw_mar)
                yaw   = kf_yaw.step(raw_yaw)
                pitch = kf_pitch.step(raw_pitch)

                eyeL_open = normalize_ear(lear)
                eyeR_open = normalize_ear(rear)

                face_bbox = bbox_from_points(pts, w, h, margin=0.12)
                if USE_DEEPFACE and emo_thread.deepface_ok:
                    emo_thread.set_frame(frame, face_bbox)
                    curr_emotion = emo_thread.curr_label
                else:
                    curr_emotion = "neutral"

                cv2.putText(frame, f"LEAR: {lear:.3f}  REAR: {rear:.3f}  MAR: {mar:.3f}", (10, 30), 0, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.3f}  Pitch: {pitch:.3f}  Emote: {curr_emotion}", (10, 55), 0, 0.6, (0,255,0), 2)

                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        frame,
                        lm,
                        mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )

                draw_avatar(frame, yaw, pitch, eyeL_open, eyeR_open, mar, emotion=curr_emotion)

            cv2.imshow("VTuber POC - Smoothed Emotion", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    if USE_DEEPFACE:
        emo_thread.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
