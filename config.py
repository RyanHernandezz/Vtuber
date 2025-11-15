DRAW_LANDMARKS = True
CAM_INDEX = 0
SMOOTHING_DT = 1.0 / 30.0
EMOTION_FPS = 10
USE_DEEPFACE = True
USE_VTUBE_STUDIO = True
VTUBE_STUDIO_URL = "ws://localhost:8001"

EAR_MIN = 0.08
EAR_MAX = 0.35

MAR_CENTER = 0.35
MAR_GAIN = 9.0
MOUTH_MAX_H = 18

TONGUE_MIN = 0.0
TONGUE_MAX = 0.8

# Emotion detection settings
EMOTION_SMOOTH_ALPHA = 0.15  # Smoothing factor: smooth[k] = smooth[k]*(1-alpha) + probs[k]*alpha
EMOTION_HYSTERESIS_MARGIN = 0.05  # Switch when new prob > old prob + margin
EMOTION_MIN_HOLD_SEC = 0.3  # Minimum time between emotion switches (prevents micro flickers)
EMOTION_DEBUG_INTERVAL = 3  # Print debug info every N seconds
EMOTION_DEEPFACE_DEBUG_INTERVAL = 5  # Print DeepFace raw output every N seconds

# Neutral calibration settings
EMOTION_CALIBRATION_DURATION = 2.0  # Calibration window in seconds
EMOTION_CALIBRATION_ENABLED = True  # Enable neutral baseline calibration

PARAM_THRESHOLDS = {
    'yaw': 0.02,
    'pitch': 0.02,
    'eye_open': 0.08,
    'mar': 0.01,
    'tongue': 0.02,
}

VTUBE_STUDIO_PARAMS = {
    'EyeOpenLeft': 'EyeOpenLeft',
    'EyeOpenRight': 'EyeOpenRight',
    'MouthOpen': 'MouthOpen',
    'AngleX': 'AngleX',
    'AngleY': 'AngleY',
    'AngleZ': 'AngleZ',
    'TongueOut': 'TongueOut',
}

EMOTION_EXPRESSIONS = {
    'happy': 'ohh.exp3.json',
    'sad': 'crying.exp3.json',
    'angry': 'angry.exp3.json',
    'surprise': 'Q.exp3.json',
    'fear': None,
    'disgust': 'black.exp3.json',
    'neutral': None
}