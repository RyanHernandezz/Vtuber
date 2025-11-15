from utils.math_utils import euclid

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

def tongue_from_landmarks(pts):
    """Calculate tongue out value from MediaPipe landmarks.
    Returns a value indicating how far the tongue is out (0 = in, higher = more out).
    Uses tongue tip (landmark 10) and mouth opening landmarks."""
    def pick(i): return pts[i]
    
    # Tongue tip (landmark 10) - only available with refine_landmarks=True
    tongue_tip = pick(10)
    
    # Mouth opening landmarks (top and bottom of inner mouth)
    # Top inner lip: 13, Bottom inner lip: 14
    mouth_top = pick(13)
    mouth_bottom = pick(14)
    
    # Calculate mouth opening center (vertical midpoint)
    mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2.0
    
    # Distance from tongue tip to mouth center line
    # In normalized coords, y increases downward
    # Positive = tongue is out (below mouth center), negative = tongue is in
    tongue_distance = tongue_tip[1] - mouth_center_y
    
    # Normalize by mouth opening height for scale invariance
    mouth_height = euclid(mouth_top, mouth_bottom) + 1e-8
    tongue_ratio = tongue_distance / mouth_height
    
    return max(0.0, tongue_ratio)  # Clamp to 0 (tongue in) or positive (tongue out)

def compute_expression_features(pts_norm):
    """Compute MediaPipe-based expression features from landmarks.
    
    Extracts a compact feature vector for expression analysis:
    - Eye aspect ratios (left and right)
    - Mouth aspect ratio
    - Head pose (yaw and pitch)
    - Tongue position
    
    Args:
        pts_norm: Normalized landmark points from MediaPipe
        
    Returns:
        dict[str, float] with keys: 'ear_left', 'ear_right', 'mar', 'yaw', 'pitch', 'tongue'
    """
    ear_left, ear_right = ears_from_landmarks(pts_norm)
    mar = mar_from_landmarks(pts_norm)
    yaw, pitch = yaw_pitch_proxy(pts_norm)
    tongue = tongue_from_landmarks(pts_norm)
    
    return {
        'ear_left': ear_left,
        'ear_right': ear_right,
        'mar': mar,
        'yaw': yaw,
        'pitch': pitch,
        'tongue': tongue
    }

def mp_features_relative(mp_feats, mp_baseline):
    """Compute relative MediaPipe features by subtracting baseline.
    
    Args:
        mp_feats: Current MediaPipe feature dict
        mp_baseline: Baseline MediaPipe feature dict
        
    Returns:
        dict[str, float] with relative feature values
    """
    return {k: mp_feats.get(k, 0.0) - mp_baseline.get(k, 0.0) for k in mp_feats}

def bbox_from_points(pts_norm, frame_w, frame_h, margin=0.35):
    """Generate bounding box from face landmarks with aggressive expansion.
    
    MediaPipe face mesh only covers inner facial features (eyes, nose, mouth, chin),
    not the full head. This function expands the bounding box to capture the full
    face including forehead, temples, and jawline for better emotion detection.
    
    Args:
        pts_norm: Normalized landmark points from MediaPipe
        frame_w: Frame width in pixels
        frame_h: Frame height in pixels
        margin: Unused (kept for compatibility), expansion is fixed
    
    Returns:
        (x, y, w, h) bounding box tuple
    """
    xs = [p[0] for p in pts_norm]
    ys = [p[1] for p in pts_norm]
    
    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))
    
    dx = (x_max - x_min)
    dy = (y_max - y_min)

    # Expand aggressively to capture full face
    # Left/right: 40% expansion (captures temples and cheeks)
    # Top: 60% expansion (captures full forehead)
    # Bottom: 30% expansion (captures jawline)
    x_min = max(0.0, x_min - dx * 0.4)
    x_max = min(1.0, x_max + dx * 0.4)
    y_min = max(0.0, y_min - dy * 0.6)
    y_max = min(1.0, y_max + dy * 0.3)

    x = int(x_min * frame_w)
    y = int(y_min * frame_h)
    w = int((x_max - x_min) * frame_w)
    h = int((y_max - y_min) * frame_h)

    return (x, y, max(1, w), max(1, h))

