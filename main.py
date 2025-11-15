import cv2
import time
import mediapipe as mp
from config import (
    DRAW_LANDMARKS, CAM_INDEX, SMOOTHING_DT, USE_DEEPFACE, PARAM_THRESHOLDS,
    USE_VTUBE_STUDIO, VTUBE_STUDIO_URL, EMOTION_EXPRESSIONS, VTUBE_STUDIO_PARAMS,
    EMOTION_CALIBRATION_ENABLED, EMOTION_CALIBRATION_DURATION
)
from core.kalman import KF1D
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

kf_yaw = KF1D(dt=SMOOTHING_DT)
kf_pitch = KF1D(dt=SMOOTHING_DT)
kf_lear = KF1D(dt=SMOOTHING_DT)
kf_rEAR = KF1D(dt=SMOOTHING_DT)
kf_mar = KF1D(dt=SMOOTHING_DT)
kf_tongue = KF1D(dt=SMOOTHING_DT)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    
    # Initialize GUI
    gui = VTuberGUI()
    gui.initialize()

    emo_thread = EmotionThread()
    calibration_start_time = None
    if USE_DEEPFACE:
        emo_thread.start()
        time.sleep(1)
        if not emo_thread.deepface_ok:
            print("WARNING: DeepFace is not working. Emotions will be stuck at 'neutral'.")
            print("Check if DeepFace dependencies are installed correctly.")
        else:
            print("DeepFace emotion detection initialized successfully.")
            if EMOTION_CALIBRATION_ENABLED:
                emo_thread.start_calibration()
                calibration_start_time = time.time()
                print("Calibration: Please relax your face for 2 seconds...")

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
                    curr_emotion = emo_thread.curr_label
                    emotion_probs = emo_thread.smooth.copy()
                else:
                    curr_emotion = "neutral"
                    emotion_probs = {'neutral': 1.0}
                
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
            
            # Check for window close
            if not gui.window or not gui.window.isVisible():
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

    if USE_DEEPFACE:
        emo_thread.stop()
    if vtube_ws:
        vtube_ws.disconnect()
    cap.release()
    gui.close()

if __name__ == "__main__":
    main()
