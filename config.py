DRAW_LANDMARKS = True
CAM_INDEX = 0  # Try 0, 1, or 2 if your camera doesn't work
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

# ============================================================================
# DYNAMIC PARAM ROUTING SYSTEM
# ============================================================================

# Mediapipe Capability Whitelist
MEDIAPIPE_CAPABILITIES = {
    "EyeOpenLeft",
    "EyeOpenRight",
    "MouthOpen",
    "FaceAngleX",
    "FaceAngleY",
    "FaceAngleZ",
    "EyeRightX",
    "EyeRightY"
}

# Map Live2D param → (mp_feature_key, scale, invert)
# scale = multiply feature value
# invert = reverse sign (optional)

VTS_PARAM_MAP = {
    "EyeOpenLeft":  ("ear_left", 1.0, False),
    "EyeOpenRight": ("ear_right", 1.0, False),
    "MouthOpen":    ("mar", 1.0, False),
    "FaceAngleX":   ("yaw", 100.0, False),
    "FaceAngleY":   ("pitch", 100.0, False),
    "FaceAngleZ":   ("roll", 1.0, False),
    "EyeRightX":    ("pupil_x", 1.0, False),
    "EyeRightY":    ("pupil_y", 1.0, False),
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

# ============================================================================
# ADVANCED TRACKING SETTINGS (High-Quality Stability)
# ============================================================================

# Master toggle for advanced tracking features
USE_ADVANCED_TRACKING = True  # Enable advanced landmark tracking

# Individual feature toggles (only apply if USE_ADVANCED_TRACKING = True)
CONFIDENCE_WEIGHTING = True   # Use velocity-based confidence scoring
OUTLIER_REJECTION = True      # Reject outlier landmark measurements
VELOCITY_CLAMPING = True      # Clamp excessive velocities to prevent jitter
REGION_SMOOTHING = True       # Apply region-based smoothing weights

# Confidence weighting parameters
MIN_CONFIDENCE = 0.1          # Minimum confidence value (0.0 to 1.0)
VELOCITY_CONFIDENCE_SCALE = 0.03  # Lower = more sensitive to velocity changes

# Outlier detection
OUTLIER_THRESHOLD = 0.1      # Distance threshold for outlier detection (normalized coords)
                              # Increase to reject fewer outliers, decrease to be more strict

# Velocity clamping
MAX_LANDMARK_VELOCITY = 0.1   # Maximum allowed velocity per frame (normalized coords)
                              # Lower = smoother but less responsive, higher = more responsive

# Region-based smoothing weights (0.0 = no smoothing, 1.0 = maximum smoothing)
STABLE_REGION_WEIGHT = 0.8    # Strong smoothing for stable anchors (nose, inner eyes)
MEDIUM_REGION_WEIGHT = 0.5    # Medium smoothing for cheeks and jaw root
EXPRESSIVE_REGION_WEIGHT = 0.2  # Light smoothing for expressive features (lips, eyelids)

# Kalman filter parameters for advanced tracking
ADVANCED_KALMAN_Q = 1e-3      # Process noise (lower = trust model more)
ADVANCED_KALMAN_R = 1e-2      # Measurement noise (lower = trust measurements more)

# Debug output
ADVANCED_TRACKING_DEBUG = False  # Print confidence scores and velocities
ADVANCED_DEBUG_INTERVAL = 5.0    # Print debug info every N seconds