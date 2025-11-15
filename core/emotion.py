import threading
import time
import numpy as np
import cv2
from config import (
    EMOTION_FPS,
    EMOTION_SMOOTH_ALPHA,
    EMOTION_HYSTERESIS_MARGIN,
    EMOTION_MIN_HOLD_SEC,
    EMOTION_DEBUG_INTERVAL,
    EMOTION_DEEPFACE_DEBUG_INTERVAL,
    EMOTION_CALIBRATION_DURATION,
    EMOTION_CALIBRATION_ENABLED
)

# -------- Emotion detection (DeepFace) thread with smoothing --------
class EmotionThread(threading.Thread):
    EMOS = ['angry','disgust','fear','happy','sad','surprise','neutral']

    def __init__(self, target_fps=EMOTION_FPS):
        super().__init__(daemon=True)
        self.target_dt = 1.0 / max(1, target_fps)
        self.running = True
        self.frame_bgr = None
        self.ready = False
        self.deepface_ok = False
        self._frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()

        # smoothed probabilities for hysteresis and stability
        self.smooth = {k: (1.0/len(self.EMOS)) for k in self.EMOS}
        self.curr_label = 'neutral'
        self._last_switch_t = time.time()
        
        # Neutral calibration state
        self.calibrating = False
        self.calibration_start_time = None
        self.df_baseline = None
        self.mp_baseline = None
        self._calib_sum_df = {k: 0.0 for k in self.EMOS}
        self._calib_sum_mp = {}
        self._calib_count = 0
        self._calib_lock = threading.Lock()

        try:
            from deepface import DeepFace  # noqa: F401
            self.deepface_ok = True
        except Exception as exc:
            self.deepface_ok = False
            error_msg = str(exc)
            if "__version__" in error_msg:
                print("DeepFace unavailable: TensorFlow installation issue detected.")
                print("Try reinstalling TensorFlow:")
                print("  pip uninstall tensorflow tf-keras")
                print("  pip install tensorflow>=2.15.0,<2.18.0 tf-keras>=2.15.0,<2.18.0")
            else:
                print(f"DeepFace unavailable. Emotion tracking disabled. Details: {exc}")

    def set_frame(self, frame_bgr, face_rect):
        if frame_bgr is None or face_rect is None:
            return

        H, W = frame_bgr.shape[:2]
        x, y, w, h = face_rect
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        w = max(1, min(W - x, w))
        h = max(1, min(H - y, h))
        roi = frame_bgr[y:y + h, x:x + w]
        if roi.size == 0:
            return

        with self._frame_lock:
            # copy to decouple from main thread buffer
            self.frame_bgr = roi.copy()
            self.ready = True
        self._new_frame_event.set()

    @staticmethod
    def canonicalize_keys(emo):
        """Uniform key normalization layer"""
        out = {k.lower(): float(v) for k, v in emo.items()}
        mapping = {
            'happiness': 'happy',
            'joy': 'happy',
            'sadness': 'sad',
            'anger': 'angry',
            'surprised': 'surprise',
            'fearful': 'fear',
            'disgusted': 'disgust'
        }
        fixed = {}
        for k, v in out.items():
            canonical_key = mapping.get(k, k)
            fixed[canonical_key] = fixed.get(canonical_key, 0) + v
        return fixed

    @staticmethod
    def _normalize_dict(d, keys):
        """Single normalization function"""
        vals = np.array([max(0.0, float(d.get(k, 0.0))) for k in keys], dtype=np.float32)
        s = float(np.sum(vals))
        if s <= 1e-8:
            vals = np.ones_like(vals) / len(vals)
        else:
            vals /= s
        return {k: float(v) for k, v in zip(keys, vals)}
    
    def start_calibration(self, duration_sec=EMOTION_CALIBRATION_DURATION):
        """Start neutral baseline calibration phase.
        
        Args:
            duration_sec: Duration of calibration window in seconds
        """
        if not EMOTION_CALIBRATION_ENABLED:
            return
        
        with self._calib_lock:
            self.calibrating = True
            self.calibration_start_time = time.time()
            self.df_baseline = None
            self.mp_baseline = None
            self._calib_sum_df = {k: 0.0 for k in self.EMOS}
            self._calib_sum_mp = {}
            self._calib_count = 0
        print(f"[Calibration] Starting neutral baseline calibration ({duration_sec}s)")
    
    def feed_calibration(self, emo_probs, mp_features=None):
        """Feed calibration data for baseline computation.
        
        Args:
            emo_probs: dict[str, float] of normalized emotion probabilities
            mp_features: optional dict[str, float] of MediaPipe features
        """
        if not self.calibrating:
            return
        
        with self._calib_lock:
            for k in self.EMOS:
                self._calib_sum_df[k] += emo_probs.get(k, 0.0)
            
            if mp_features:
                for k, v in mp_features.items():
                    if k not in self._calib_sum_mp:
                        self._calib_sum_mp[k] = 0.0
                    self._calib_sum_mp[k] += v
            
            self._calib_count += 1
    
    def finalize_calibration(self):
        """Finalize calibration and compute baselines."""
        if not self.calibrating or self._calib_count == 0:
            return
        
        with self._calib_lock:
            # Compute DeepFace baseline
            self.df_baseline = {
                k: self._calib_sum_df[k] / self._calib_count
                for k in self.EMOS
            }
            
            # Compute MediaPipe baseline
            if self._calib_sum_mp:
                self.mp_baseline = {
                    k: self._calib_sum_mp[k] / self._calib_count
                    for k in self._calib_sum_mp
                }
            
            self.calibrating = False
            
            print(f"[Calibration] Completed: {self._calib_count} samples")
            print(f"[Calibration] DeepFace baseline: {[(k, f'{v:.3f}') for k, v in sorted(self.df_baseline.items(), key=lambda x: x[1], reverse=True)[:3]]}")
    
    def _apply_baseline_compensation(self, probs_raw):
        """Apply baseline compensation to DeepFace probabilities.
        
        Args:
            probs_raw: dict[str, float] of raw normalized probabilities
            
        Returns:
            dict[str, float] of baseline-compensated probabilities
        """
        if self.df_baseline is None:
            return probs_raw
        
        # Subtract baseline and clamp to non-negative
        adjusted = {
            k: max(0.0, probs_raw.get(k, 0.0) - self.df_baseline.get(k, 0.0))
            for k in self.EMOS
        }
        
        # Renormalize
        total = sum(adjusted.values())
        if total > 1e-8:
            adjusted = {k: v / total for k, v in adjusted.items()}
        else:
            # Fallback to uniform if all values are zero
            adjusted = {k: 1.0 / len(self.EMOS) for k in self.EMOS}
        
        return adjusted

    def run(self):
        if not self.deepface_ok:
            while self.running:
                time.sleep(self.target_dt)
            return

        from deepface import DeepFace
        while self.running:
            t0 = time.time()
            if not self._new_frame_event.wait(timeout=self.target_dt):
                continue
            self._new_frame_event.clear()

            with self._frame_lock:
                frame = None if self.frame_bgr is None else self.frame_bgr.copy()
                self.ready = False

            if frame is None or frame.size == 0:
                continue

            try:
                # DeepFace expects RGB, convert from BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to 224x224 for stable DeepFace detection
                # DeepFace models are trained on 224x224 images
                frame_rgb = cv2.resize(frame_rgb, (224, 224))
                
                res = DeepFace.analyze(
                    img_path=frame_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(res, list) and len(res) > 0:
                    res = res[0]
                emo_dict = res.get('emotion') or res.get('emotions') or {}
                
                # Canonicalize keys
                canonical = self.canonicalize_keys(emo_dict)
                
                # Debug: print raw DeepFace output occasionally
                if int(time.time()) % EMOTION_DEEPFACE_DEBUG_INTERVAL == 0:
                    print(f"[DeepFace Raw] Original: {emo_dict}")
                    print(f"[DeepFace Raw] Canonicalized: {canonical}")
                
                # Single normalization
                probs_raw = self._normalize_dict(canonical, self.EMOS)
                
                # Check if calibration period has elapsed
                now = time.time()
                if self.calibrating and self.calibration_start_time is not None:
                    elapsed = now - self.calibration_start_time
                    if elapsed >= EMOTION_CALIBRATION_DURATION:
                        self.finalize_calibration()
                    else:
                        # Still calibrating - collect data but don't update emotion
                        self.feed_calibration(probs_raw)
                        continue
                
                # Apply baseline compensation if available
                probs = self._apply_baseline_compensation(probs_raw)
                
                # Stable smoothing: smooth[k] = smooth[k]*(1-alpha) + probs[k]*alpha
                for k in self.EMOS:
                    self.smooth[k] = self.smooth[k] * (1.0 - EMOTION_SMOOTH_ALPHA) + probs[k] * EMOTION_SMOOTH_ALPHA

                # Simple hysteresis: switch when new prob > old prob + margin
                top_new = max(self.smooth, key=self.smooth.get)
                p_new = self.smooth[top_new]
                p_curr = self.smooth.get(self.curr_label, 0.0)
                
                # Minimum hold time to prevent micro flickers
                if now - self._last_switch_t < EMOTION_MIN_HOLD_SEC:
                    continue
                
                # Reduced hysteresis: switch when new prob > old prob + margin
                if top_new != self.curr_label and p_new > p_curr + EMOTION_HYSTERESIS_MARGIN:
                    print(f"[Emotion] Switching: {self.curr_label} -> {top_new} (p_new={p_new:.2f}, p_curr={p_curr:.2f})")
                    self.curr_label = top_new
                    self._last_switch_t = now

                # Debug: print top emotions occasionally
                if int(now) % EMOTION_DEBUG_INTERVAL == 0:
                    top_3 = sorted(self.smooth.items(), key=lambda x: x[1], reverse=True)[:3]
                    time_held = now - self._last_switch_t
                    raw_top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"[Emotion] Raw Top 3: {[(e, f'{p:.2f}') for e, p in raw_top_3]}")
                    print(f"[Emotion] Smoothed Top 3: {[(e, f'{p:.2f}') for e, p in top_3]}, Current: {self.curr_label} ({p_curr:.2f}), Held for: {time_held:.1f}s")

            except Exception as e:
                # DeepFace occasionally throws when face ROI is too small /
                # blurred. Log occasionally to help debug.
                import time as time_module
                now_sec = int(time_module.time())
                if now_sec % 5 == 0:  # Log every 5 seconds max
                    print(f"[Emotion] DeepFace error: {type(e).__name__}: {str(e)[:100]}")
                pass

            dt = time.time() - t0
            time.sleep(max(0.0, self.target_dt - dt))

    def stop(self):
        self.running = False
        self._new_frame_event.set()

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

