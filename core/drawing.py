import cv2
from math import exp
from config import EAR_MIN, EAR_MAX, MAR_CENTER, MAR_GAIN, MOUTH_MAX_H, TONGUE_MIN, TONGUE_MAX
from utils.math_utils import clamp01
from core.emotion import emotion_to_color

def normalize_ear(ear):
    t = (ear - EAR_MIN) / max(1e-6, (EAR_MAX - EAR_MIN))
    return clamp01(t)

def mar_to_mouth_height(mar):
    s = 1.0 / (1.0 + exp(-MAR_GAIN * (mar - MAR_CENTER)))
    return int(MOUTH_MAX_H * s) + 2  # small offset for visibility

def normalize_mar(mar):
    """Normalize MAR to 0-1 using the same sigmoid as mar_to_mouth_height"""
    s = 1.0 / (1.0 + exp(-MAR_GAIN * (mar - MAR_CENTER)))
    return clamp01(s)

def normalize_yaw(yaw):
    """Normalize yaw from [-0.5, +0.5] to [0, 1] where 0.5 = center"""
    # yaw range is approximately -0.5 (left) to +0.5 (right), 0 = center
    # Map to 0-1: center (0) -> 0.5, left (-0.5) -> 0, right (+0.5) -> 1
    normalized = yaw + 0.5
    return clamp01(normalized)

def normalize_pitch(pitch):
    """Normalize pitch from [-0.5, +0.5] to [0, 1] where 0.5 = center"""
    # pitch range is approximately -0.5 (down) to +0.5 (up), 0 = center
    # Map to 0-1: center (0) -> 0.5, down (-0.5) -> 0, up (+0.5) -> 1
    normalized = pitch + 0.5
    return clamp01(normalized)

def normalize_tongue(tongue):
    """Normalize tongue value to 0-1 range"""
    t = (tongue - TONGUE_MIN) / max(1e-6, (TONGUE_MAX - TONGUE_MIN))
    return clamp01(t)

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

