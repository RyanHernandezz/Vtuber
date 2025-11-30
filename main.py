import cv2
import time
import mediapipe as mp
from config import (
    DRAW_LANDMARKS, CAM_INDEX, SMOOTHING_DT, USE_DEEPFACE, PARAM_THRESHOLDS,
    USE_VTUBE_STUDIO, VTUBE_STUDIO_URL, EMOTION_EXPRESSIONS, VTUBE_STUDIO_PARAMS,
    EMOTION_CALIBRATION_ENABLED, EMOTION_CALIBRATION_DURATION,
    # Advanced tracking settings
    USE_ADVANCED_TRACKING, CONFIDENCE_WEIGHTING, OUTLIER_REJECTION, VELOCITY_CLAMPING,
    REGION_SMOOTHING, MIN_CONFIDENCE, VELOCITY_CONFIDENCE_SCALE, OUTLIER_THRESHOLD,
    MAX_LANDMARK_VELOCITY, STABLE_REGION_WEIGHT, MEDIUM_REGION_WEIGHT,
    EXPRESSIVE_REGION_WEIGHT, ADVANCED_KALMAN_Q, ADVANCED_KALMAN_R,
    ADVANCED_TRACKING_DEBUG, ADVANCED_DEBUG_INTERVAL
)
from core.kalman import KF1D
from core.advanced_kalman import AdvancedKalmanFilter, LandmarkRegions
from core.landmarks import (
    ears_from_landmarks, mar_from_landmarks, yaw_pitch_proxy, bbox_from_points,
    tongue_from_landmarks, compute_expression_features
)
from core.drawing import normalize_ear, normalize_mar, normalize_yaw, normalize_pitch, normalize_tongue
from core.emotion import EmotionThread
from gui_panel import VTuberGUI

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_mesh

# Basic Kalman filters for simple features (kept for backward compatibility)
kf_yaw = KF1D(dt=SMOOTHING_DT)
kf_pitch = KF1D(dt=SMOOTHING_DT)
kf_lear = KF1D(dt=SMOOTHING_DT)
kf_rEAR = KF1D(dt=SMOOTHING_DT)
kf_mar = KF1D(dt=SMOOTHING_DT)
kf_tongue = KF1D(dt=SMOOTHING_DT)

# Advanced Kalman filters for all 468 MediaPipe landmarks (initialized on first use)
advanced_landmark_filters = None
previous_landmarks = None

def apply_advanced_tracking(pts, previous_pts=None):
    """
    Apply commercial-grade tracking to all landmarks.
    
    Args:
        pts: Current landmark points [(x, y), ...]
        previous_pts: Previous frame landmarks (optional)
        
    Returns:
        Filtered landmark points [(x, y), ...]
    """
    global advanced_landmark_filters, previous_landmarks
    
    if not USE_ADVANCED_TRACKING:
        return pts
    
    # Initialize filters on first use
    if advanced_landmark_filters is None:
        print("Initializing advanced tracking for 468 landmarks...")
        advanced_landmark_filters = []
        for i in range(len(pts)):
            x, y = pts[i]
            filter_x = AdvancedKalmanFilter(
                x0=x, dt=SMOOTHING_DT,
                q=ADVANCED_KALMAN_Q, r=ADVANCED_KALMAN_R,
                min_confidence=MIN_CONFIDENCE,
                velocity_scale=VELOCITY_CONFIDENCE_SCALE,
                outlier_threshold=OUTLIER_THRESHOLD,
                max_velocity=MAX_LANDMARK_VELOCITY
            )
            filter_y = AdvancedKalmanFilter(
                x0=y, dt=SMOOTHING_DT,
                q=ADVANCED_KALMAN_Q, r=ADVANCED_KALMAN_R,
                min_confidence=MIN_CONFIDENCE,
                velocity_scale=VELOCITY_CONFIDENCE_SCALE,
                outlier_threshold=OUTLIER_THRESHOLD,
                max_velocity=MAX_LANDMARK_VELOCITY
            )
            advanced_landmark_filters.append((filter_x, filter_y))
        previous_landmarks = pts
        print("Advanced tracking initialized!")
    
    # Apply filtering to each landmark
    filtered_pts = []
    for i, (x, y) in enumerate(pts):
        filter_x, filter_y = advanced_landmark_filters[i]
        
        # Get region weight for this landmark
        region_weight = 0.5  # Default
        if REGION_SMOOTHING:
            region_weight = LandmarkRegions.get_region_weight(
                i,
                STABLE_REGION_WEIGHT,
                MEDIUM_REGION_WEIGHT,
                EXPRESSIVE_REGION_WEIGHT
            )
        
        # Apply advanced Kalman filtering
        filtered_x = filter_x.step(
            x,
            region_weight=region_weight,
            use_confidence=CONFIDENCE_WEIGHTING,
            reject_outliers=OUTLIER_REJECTION,
            clamp_velocity=VELOCITY_CLAMPING
        )
        filtered_y = filter_y.step(
            y,
            region_weight=region_weight,
            use_confidence=CONFIDENCE_WEIGHTING,
            reject_outliers=OUTLIER_REJECTION,
            clamp_velocity=VELOCITY_CLAMPING
        )
        
        filtered_pts.append((filtered_x, filtered_y))
    
    previous_landmarks = filtered_pts
    return filtered_pts

def main():
    try:
        # Try to open camera with fallback to other indices
        cap = None
        camera_indices_to_try = [CAM_INDEX, 0, 1, 2]
        
        for idx in camera_indices_to_try:
            print(f"Trying to open camera index {idx}...")
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                # Test if we can actually read a frame
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    print(f"✓ Successfully opened camera at index {idx}")
                    cap = test_cap
                    break
                else:
                    print(f"✗ Camera {idx} opened but cannot read frames")
                    test_cap.release()
            else:
                print(f"✗ Cannot open camera at index {idx}")
        
        if cap is None or not cap.isOpened():
            print("\n" + "="*60)
            print("ERROR: Cannot open webcam!")
            print("="*60)
            print("Possible solutions:")
            print("1. Close VTube Studio (or disable its camera auto-start)")
            print("2. Close other applications using the camera (Zoom, Teams, OBS, etc.)")
            print("3. Make sure your webcam is connected and not in use")
            print("4. Try changing CAM_INDEX in config.py to 1 or 2")
            print("5. Check Windows Camera app to verify your camera works")
            print("="*60)
            raise SystemExit("Cannot open webcam")
        
        # Initialize GUI
        gui = VTuberGUI()
        gui.initialize()
        print("GUI initialized successfully")
    except Exception as e:
        print(f"ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    emo_thread = EmotionThread()
    calibration_start_time = None
    
    # Connect calibration button signal
    def on_calibration_requested():
        nonlocal calibration_start_time
        if USE_DEEPFACE and emo_thread.deepface_ok:
            emo_thread.start_calibration()
            calibration_start_time = time.time()
            print("Calibration: Please relax your face for 2 seconds...")
    
    gui.window.calibration_requested.connect(on_calibration_requested)
    
    if USE_DEEPFACE:
        emo_thread.start()
        time.sleep(1)
        if not emo_thread.deepface_ok:
            print("WARNING: DeepFace is not working. Emotions will be stuck at 'neutral'.")
            print("Check if DeepFace dependencies are installed correctly.")
            # Disable calibration button if DeepFace is not available
            gui.window.calibration_button.setEnabled(False)
            gui.window.calibration_button.setToolTip("DeepFace is not available. Calibration disabled.")
        else:
            print("DeepFace emotion detection initialized successfully.")
            # Don't auto-start calibration - user will click button

    vtube_ws = None
    if USE_VTUBE_STUDIO:
        try:
            from vtube_studio.client import VTubeStudioClient
            vtube_ws = VTubeStudioClient(url=VTUBE_STUDIO_URL)
            if vtube_ws.connect():
                print("VTube Studio connection initiated")
                time.sleep(2)
                if vtube_ws.authenticated:
                    params = vtube_ws.get_input_parameters()
                    if params:
                        print(f"Found {len(params)} available parameters in VTube Studio")
                        print("Make sure your parameter names in config.py match your model's parameters!")
            else:
                print("Warning: Could not connect to VTube Studio")
        except ImportError:
            print("Warning: websocket-client not installed. Install with: pip install websocket-client")
            vtube_ws = None
        except Exception as e:
            print(f"Warning: VTube Studio initialization failed: {e}")
            vtube_ws = None

    last_sent = {
        'yaw': None,
        'pitch': None,
        'eyeL_open': None,
        'eyeR_open': None,
        'mar': None,
        'tongue': None,
        'emotion': None,
    }
    last_active_expression = None  # Track the currently active expression file

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        try:
            print("Starting main loop...")
            
            # FPS tracking
            fps_history = []
            last_fps_update = time.time()
            frame_times = []
            
            # Debug: track expression updates
            last_expression_debug = time.time()
            
            # Debug: track advanced tracking stats
            last_advanced_debug = time.time()
            
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from camera")
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)

                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]
                    pts = [(p.x, p.y) for p in lm.landmark]
                    
                    # Apply advanced tracking to all landmarks
                    if USE_ADVANCED_TRACKING:
                        pts = apply_advanced_tracking(pts, previous_landmarks)
                    
                    # Extract facial features from (possibly filtered) landmarks
                    raw_lear, raw_rear = ears_from_landmarks(pts)
                    raw_mar = mar_from_landmarks(pts)
                    raw_yaw, raw_pitch = yaw_pitch_proxy(pts)
                    raw_tongue = tongue_from_landmarks(pts)

                    lear  = kf_lear.step(raw_lear)
                    rear  = kf_rEAR.step(raw_rear)
                    mar   = kf_mar.step(raw_mar)
                    yaw   = kf_yaw.step(raw_yaw)
                    pitch = kf_pitch.step(raw_pitch)
                    tongue = kf_tongue.step(raw_tongue)

                    eyeL_open = normalize_ear(lear)
                    eyeR_open = normalize_ear(rear)
                    tongue_out = normalize_tongue(tongue)

                    face_bbox = bbox_from_points(pts, w, h, margin=0.12)
                    if USE_DEEPFACE and emo_thread.deepface_ok:
                        emo_thread.set_frame(frame, face_bbox)
                        
                        # Only show emotions if calibration is complete
                        if emo_thread.calibrating or emo_thread.df_baseline is None:
                            # Still calibrating or not calibrated yet - show neutral
                            curr_emotion = "neutral"
                            emotion_probs = None  # This will make GUI show blank
                        else:
                            # Calibration complete - show real emotions
                            curr_emotion = emo_thread.curr_label
                            emotion_probs = emo_thread.smooth.copy()
                    else:
                        curr_emotion = "neutral"
                        emotion_probs = None  # No DeepFace - show blank
                    
                    # Get MediaPipe features for GUI
                    mp_features = compute_expression_features(pts)

                    params_changed = False
                    
                    if last_sent['yaw'] is None or abs(yaw - last_sent['yaw']) >= PARAM_THRESHOLDS['yaw']:
                        params_changed = True
                        last_sent['yaw'] = yaw
                    
                    if last_sent['pitch'] is None or abs(pitch - last_sent['pitch']) >= PARAM_THRESHOLDS['pitch']:
                        params_changed = True
                        last_sent['pitch'] = pitch
                    
                    if last_sent['eyeL_open'] is None or abs(eyeL_open - last_sent['eyeL_open']) >= PARAM_THRESHOLDS['eye_open']:
                        params_changed = True
                        last_sent['eyeL_open'] = eyeL_open
                    
                    if last_sent['eyeR_open'] is None or abs(eyeR_open - last_sent['eyeR_open']) >= PARAM_THRESHOLDS['eye_open']:
                        params_changed = True
                        last_sent['eyeR_open'] = eyeR_open
                    
                    if last_sent['mar'] is None or abs(mar - last_sent['mar']) >= PARAM_THRESHOLDS['mar']:
                        params_changed = True
                        last_sent['mar'] = mar
                    
                    if last_sent['tongue'] is None or abs(tongue_out - last_sent['tongue']) >= PARAM_THRESHOLDS['tongue']:
                        params_changed = True
                        last_sent['tongue'] = tongue_out
                    
                    if last_sent['emotion'] != curr_emotion:
                        params_changed = True
                        prev_emotion = last_sent['emotion']
                        last_sent['emotion'] = curr_emotion
                        
                        if vtube_ws and vtube_ws.connected and vtube_ws.authenticated:
                            # Deactivate previous expression if it exists
                            if last_active_expression:
                                vtube_ws.deactivate_expression(last_active_expression)
                                print(f"Deactivated expression: {last_active_expression} (switching from {prev_emotion} to {curr_emotion})")
                                last_active_expression = None
                            
                            # Activate new expression if it exists
                            expr_file = EMOTION_EXPRESSIONS.get(curr_emotion)
                            if expr_file:
                                vtube_ws.trigger_expression(expr_file)
                                last_active_expression = expr_file
                                print(f"Triggered expression: {expr_file} for emotion: {curr_emotion}")
                            else:
                                # Neutral or emotion without expression - ensure previous is deactivated
                                if curr_emotion == 'neutral' and last_active_expression:
                                    # Already deactivated above, but ensure it's cleared
                                    last_active_expression = None

                    # Update GUI
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gui.update_frame(frame_rgb)
                    gui.update_emotion(curr_emotion, emotion_probs)
                    gui.update_expression(mp_features)
                    
                    # Debug: print expression values occasionally
                    if time.time() - last_expression_debug >= 2.0:
                        print(f"Expression values: MAR={mp_features.get('mar', 0):.3f}, "
                              f"EAR_L={mp_features.get('ear_left', 0):.3f}, "
                              f"EAR_R={mp_features.get('ear_right', 0):.3f}, "
                              f"Yaw={mp_features.get('yaw', 0):.3f}, "
                              f"Pitch={mp_features.get('pitch', 0):.3f}")
                        last_expression_debug = time.time()
                    
                    # Debug: print advanced tracking stats
                    if ADVANCED_TRACKING_DEBUG and USE_ADVANCED_TRACKING and time.time() - last_advanced_debug >= ADVANCED_DEBUG_INTERVAL:
                        if advanced_landmark_filters:
                            # Sample confidence from a few stable and expressive landmarks
                            stable_confidences = []
                            expressive_confidences = []
                            
                            for idx in LandmarkRegions.STABLE[:5]:  # Sample 5 stable landmarks
                                if idx < len(advanced_landmark_filters):
                                    filter_x, _ = advanced_landmark_filters[idx]
                                    conf = filter_x.compute_confidence(pts[idx][0])
                                    stable_confidences.append(conf)
                            
                            for idx in LandmarkRegions.EXPRESSIVE[:5]:  # Sample 5 expressive landmarks
                                if idx < len(advanced_landmark_filters):
                                    filter_x, _ = advanced_landmark_filters[idx]
                                    conf = filter_x.compute_confidence(pts[idx][0])
                                    expressive_confidences.append(conf)
                            
                            avg_stable = sum(stable_confidences) / len(stable_confidences) if stable_confidences else 0
                            avg_expressive = sum(expressive_confidences) / len(expressive_confidences) if expressive_confidences else 0
                            
                            print(f"Advanced Tracking: Stable confidence={avg_stable:.3f}, Expressive confidence={avg_expressive:.3f}")
                        last_advanced_debug = time.time()
                    
                    # Update calibration progress
                    if calibration_start_time is not None and emo_thread.calibrating:
                        elapsed = time.time() - calibration_start_time
                        gui.update_calibration(elapsed, EMOTION_CALIBRATION_DURATION)
                    elif calibration_start_time is not None and not emo_thread.calibrating:
                        calibration_start_time = None
                else:
                    # No face detected - still update frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gui.update_frame(frame_rgb)

                # Process GUI events
                gui.process_events()
                
                # Calculate and update FPS
                current_time = time.time()
                if len(frame_times) > 0:
                    frame_delta = current_time - frame_times[-1]
                    if frame_delta > 0:
                        instant_fps = 1.0 / frame_delta
                        fps_history.append(instant_fps)
                        # Keep only last 30 frames for averaging
                        if len(fps_history) > 30:
                            fps_history.pop(0)
                
                frame_times.append(current_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                # Update FPS display every 0.5 seconds
                if current_time - last_fps_update >= 0.5 and len(fps_history) > 0:
                    avg_fps = sum(fps_history) / len(fps_history)
                    gui.update_fps(avg_fps)
                    last_fps_update = current_time
                
                # Check for window close
                if not gui.window or not gui.window.isVisible():
                    print("Window closed by user")
                    break

                # VTube Studio parameter updates (only when face detected and params changed)
                if res.multi_face_landmarks and params_changed and vtube_ws and vtube_ws.connected and vtube_ws.authenticated:
                        vtube_yaw = last_sent['yaw'] if last_sent['yaw'] is not None else yaw
                        vtube_pitch = last_sent['pitch'] if last_sent['pitch'] is not None else pitch
                        vtube_mar = last_sent['mar'] if last_sent['mar'] is not None else mar
                        
                        norm_yaw = normalize_yaw(vtube_yaw)
                        norm_pitch = normalize_pitch(vtube_pitch)
                        norm_mar = normalize_mar(vtube_mar)
                        
                        angle_x = (norm_yaw - 0.5) * 60.0
                        angle_y = (norm_pitch - 0.5) * 60.0
                        
                        vtube_params = {}
                        
                        if VTUBE_STUDIO_PARAMS.get('EyeOpenLeft'):
                            vtube_params[VTUBE_STUDIO_PARAMS['EyeOpenLeft']] = last_sent['eyeL_open'] if last_sent['eyeL_open'] is not None else eyeL_open
                      
                        if VTUBE_STUDIO_PARAMS.get('EyeOpenRight'):
                            vtube_params[VTUBE_STUDIO_PARAMS['EyeOpenRight']] = last_sent['eyeR_open'] if last_sent['eyeR_open'] is not None else eyeR_open
                      
                        if VTUBE_STUDIO_PARAMS.get('MouthOpen'):
                            vtube_params[VTUBE_STUDIO_PARAMS['MouthOpen']] = norm_mar
                      
                        if VTUBE_STUDIO_PARAMS.get('AngleX'):
                            vtube_params[VTUBE_STUDIO_PARAMS['AngleX']] = angle_x
                      
                        if VTUBE_STUDIO_PARAMS.get('AngleY'):
                            vtube_params[VTUBE_STUDIO_PARAMS['AngleY']] = angle_y
                      
                        if VTUBE_STUDIO_PARAMS.get('AngleZ'):
                            vtube_params[VTUBE_STUDIO_PARAMS['AngleZ']] = 0.0
                        
                        if VTUBE_STUDIO_PARAMS.get('TongueOut'):
                            vtube_params[VTUBE_STUDIO_PARAMS['TongueOut']] = last_sent['tongue'] if last_sent['tongue'] is not None else tongue_out
                        
                        if vtube_params:
                            vtube_ws.send_parameters(vtube_params)
        
        except Exception as e:
            print(f"ERROR in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            if USE_DEEPFACE:
                emo_thread.stop()
            if vtube_ws:
                vtube_ws.disconnect()
            cap.release()
            gui.close()
            print("Cleanup complete")

if __name__ == "__main__":
    main()
