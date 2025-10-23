import cv2
import numpy as np
import mediapipe as mp
from math import hypot

# ---------- Config ----------
DRAW_LANDMARKS = True
CAM_INDEX = 0
SMOOTHING_DT = 1.0 / 30.0   # assumed camera fps for Kalman
# ----------------------------

# Eye and mouth landmark indices per Mediapipe FaceMesh.
# Using the exact points referenced in the VTuber paper for EAR/MAR
# EAR: left eye (160,144),(158,153), corners (33,133); right eye (387,373),(385,380), corners (263,362)
# MAR: vertical (81,178),(13,14),(311,402), corners (78,308)
L_EYE_VERT = [(160,144), (158,153)]
L_EYE_HORZ = (33,133)
R_EYE_VERT = [(387,373), (385,380)]
R_EYE_HORZ = (263,362)
MOUTH_VERT = [(81,178), (13,14), (311,402)]
MOUTH_HORZ = (78,308)

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh

# --------- Small 1D Kalman filter helper (position, velocity) ----------
class KF1D:
    def __init__(self, x0=0.0, v0=0.0, dt=SMOOTHING_DT, q=1e-2, r=1e-2):
        # state: [x, v]^T
        self.dt = dt
        self.x = np.array([[x0], [v0]], dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32) * 1.0
        self.F = np.array([[1, dt],
                           [0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0]], dtype=np.float32)  # measure position only
        self.Q = np.array([[q, 0],
                           [0, q]], dtype=np.float32)
        self.R = np.array([[r]], dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array([[z]], dtype=np.float32)
        y = z - (self.H @ self.x)                  # innovation
        S = self.H @ self.P @ self.H.T + self.R    # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(2, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        return float(self.x[0,0])  # filtered position

# instantiate filters
kf_yaw   = KF1D()
kf_pitch = KF1D()
kf_ear   = KF1D()
kf_mar   = KF1D()

def euclid(p1, p2):
    return hypot(p1[0]-p2[0], p1[1]-p2[1])

def ear_from_landmarks(pts):
    # pts is list of (x,y) normalized
    def pick(i): return pts[i]
    # left eye
    v1 = euclid(pick(160), pick(144))
    v2 = euclid(pick(158), pick(153))
    h  = euclid(pick(33),  pick(133))
    left = (v1 + v2) / (2.0 * h + 1e-8)
    # right eye
    v1 = euclid(pick(387), pick(373))
    v2 = euclid(pick(385), pick(380))
    h  = euclid(pick(263), pick(362))
    right = (v1 + v2) / (2.0 * h + 1e-8)
    return (left + right) * 0.5

def mar_from_landmarks(pts):
    def pick(i): return pts[i]
    v1 = euclid(pick(81),  pick(178))
    v2 = euclid(pick(13),  pick(14))
    v3 = euclid(pick(311), pick(402))
    h  = euclid(pick(78),  pick(308))
    return (v1 + v2 + v3) / (2.0 * h + 1e-8)

def yaw_pitch_proxy(pts):
    # simple, robust proxies, no solvePnP, keeps POC simple
    # yaw  ≈ horizontal offset of nose relative to face bbox center
    # pitch ≈ vertical offset of nose relative to face bbox center
    NOSE = 1  # Mediapipe "nose tip" proxy works decently
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    nx, ny = pts[NOSE]
    # normalize by face size so values are roughly in [-1,1]
    w = (x_max - x_min) + 1e-8
    h = (y_max - y_min) + 1e-8
    yaw = (nx - cx) / w      # left positive, right negative
    pitch = (cy - ny) / h    # up positive, down negative
    return yaw, pitch

def draw_avatar(frame, yaw, pitch, ear, mar):
    # minimal 2D avatar at bottom-left
    H, W = frame.shape[:2]
    base_x, base_y = 120, H - 140
    head_r = 50

    # head position tilt
    offset_x = int(40 * yaw)
    offset_y = int(40 * (-pitch))

    center = (base_x + offset_x, base_y + offset_y)
    cv2.circle(frame, center, head_r, (255, 255, 255), 2)

    # eyes depend on EAR (smaller = blink)
    eye_open = max(0.0, min(1.0, (ear - 0.15) / 0.15))  # crude scaling
    eye_h = int(8 * eye_open) + 1
    eye_w = 10
    eye_dx = 18; eye_dy = -10
    # left eye
    cv2.ellipse(frame, (center[0] - eye_dx, center[1] + eye_dy), (eye_w, eye_h), 0, 0, 360, (255,255,255), 2)
    # right eye
    cv2.ellipse(frame, (center[0] + eye_dx, center[1] + eye_dy), (eye_w, eye_h), 0, 0, 360, (255,255,255), 2)

    # mouth depends on MAR
    mouth_w = 24
    mouth_h = int(60 * max(0.0, min(1.0, (mar - 0.2) / 0.4))) + 2
    cv2.ellipse(frame, (center[0], center[1] + 15), (mouth_w, mouth_h), 0, 0, 360, (255,255,255), 2)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

with mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # adds iris points
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
            pts = [(p.x, p.y) for p in lm.landmark]  # normalized

            # EAR, MAR
            raw_ear = ear_from_landmarks(pts)
            raw_mar = mar_from_landmarks(pts)

            # yaw, pitch proxies
            raw_yaw, raw_pitch = yaw_pitch_proxy(pts)

            # filter
            ear   = kf_ear.step(raw_ear)
            mar   = kf_mar.step(raw_mar)
            yaw   = kf_yaw.step(raw_yaw)
            pitch = kf_pitch.step(raw_pitch)

            # overlay debug text
            cv2.putText(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}", (10, 30), 0, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.3f}  Pitch: {pitch:.3f}", (10, 55), 0, 0.7, (0,255,0), 2)

            # draw mesh if desired
            if DRAW_LANDMARKS:
                mp_drawing.draw_landmarks(
                    frame,
                    lm,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

            # simple avatar
            draw_avatar(frame, yaw, pitch, ear, mar)

        cv2.imshow("VTuber POC - MediaPipe + Kalman", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

cap.release()
cv2.destroyAllWindows()
