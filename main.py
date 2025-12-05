import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
from PySide6.QtCore import QTimer

import config
from core.emotion import EmotionThread
from core.landmarks import compute_expression_features, bbox_from_points
from gui_panel import VTuberGUI
from vtube_studio.client import VTubeStudioClient

def main():
    # 1. Initialize GUI
    gui = VTuberGUI()
    gui.initialize()
    
    # 2. Initialize Emotion Thread
    emotion_thread = EmotionThread(target_fps=config.EMOTION_FPS)
    emotion_thread.start()
    
    # 3. Initialize VTube Studio Client
    vts_client = None
    if config.USE_VTUBE_STUDIO:
        vts_client = VTubeStudioClient(url=config.VTUBE_STUDIO_URL)
        # Connect in a separate thread to avoid blocking GUI startup
        threading.Thread(target=vts_client.connect, daemon=True).start()
    
    # 4. Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 5. Camera Architecture (Producer-Consumer)
    camera_cmd_queue = queue.Queue()
    frame_queue = queue.Queue(maxsize=1) # Only keep latest frame
    
    # Global state for camera thread management
    current_run_id = 0
    run_id_lock = threading.Lock()
    
    def camera_read_loop(run_id, cap):
        """Dedicated thread for reading frames from camera."""
        print(f"[Camera Loop {run_id}] Started")
        while True:
            # Check if this loop is still valid
            with run_id_lock:
                if run_id != current_run_id:
                    print(f"[Camera Loop {run_id}] Stale, stopping")
                    break
            
            if not cap.isOpened():
                break
                
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Try to put frame in queue, drop old if full (non-blocking)
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait() # Remove old
                    frame_queue.put_nowait(frame) # Put new
                except queue.Empty:
                    pass
            
            # Small sleep to prevent busy loop if camera is fast
            # time.sleep(0.001) 
            
        cap.release()
        print(f"[Camera Loop {run_id}] Stopped and released camera")

    def camera_manager():
        """Worker thread to process OPEN/CLOSE commands sequentially."""
        nonlocal current_run_id
        
        while True:
            cmd = camera_cmd_queue.get()
            if cmd == "EXIT":
                break
                
            if cmd == "OPEN":
                # Invalidate old loops
                with run_id_lock:
                    current_run_id += 1
                    my_run_id = current_run_id
                
                print(f"[Camera Manager] Processing OPEN (Run ID: {my_run_id})")
                
                # Open camera
                cap = cv2.VideoCapture(config.CAM_INDEX)
                if not cap.isOpened():
                    # Try other indices
                    for i in range(3):
                        if i == config.CAM_INDEX: continue
                        print(f"Trying camera index {i}...")
                        cap = cv2.VideoCapture(i)
                        if cap.isOpened():
                            print(f"Opened camera index {i}")
                            break
                    else:
                        print("Error: Could not open any webcam.")
                        cap = None
                
                if cap and cap.isOpened():
                    # Start reading loop
                    t = threading.Thread(target=camera_read_loop, args=(my_run_id, cap), daemon=True)
                    t.start()
                    
            elif cmd == "CLOSE":
                print("[Camera Manager] Processing CLOSE")
                # Invalidate current loop - this will cause the read loop to exit and release cap
                with run_id_lock:
                    current_run_id += 1
                
                # Clear frame queue so UI updates immediately
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Send empty frame to clear UI
                gui.update_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    # Start manager thread
    manager_thread = threading.Thread(target=camera_manager, daemon=True)
    manager_thread.start()
    
    # Initial camera open
    camera_cmd_queue.put("OPEN")
    
    # Connect GUI signals
    def on_camera_toggled(enabled):
        if enabled:
            camera_cmd_queue.put("OPEN")
        else:
            camera_cmd_queue.put("CLOSE")
            
    gui.window.camera_toggled.connect(on_camera_toggled)
    
    def on_calibration_requested():
        emotion_thread.start_calibration(duration_sec=config.EMOTION_CALIBRATION_DURATION)
        
    gui.window.calibration_requested.connect(on_calibration_requested)
    
    # Main Loop Variables
    last_time = time.time()
    frame_count = 0
    fps_update_time = time.time()
    
    current_emotion = 'neutral'
    
    print("Starting main loop...")
    
    while True:
        # Process GUI events
        gui.process_events()
        if not gui.window.isVisible():
            break
            
        current_time = time.time()
        
        # FPS Calculation
        if current_time - fps_update_time >= 1.0:
            fps = frame_count / (current_time - fps_update_time)
            gui.update_fps(fps)
            frame_count = 0
            fps_update_time = current_time
            
            # Update VTS status
            if vts_client:
                gui.window.set_vts_status(vts_client.connected and vts_client.authenticated)
            
        # Update calibration progress
        if emotion_thread.calibrating and emotion_thread.calibration_start_time:
            elapsed = current_time - emotion_thread.calibration_start_time
            gui.update_calibration(elapsed, config.EMOTION_CALIBRATION_DURATION)
        elif gui.window.calibration_widget.isVisible() and not emotion_thread.calibrating:
             # Just finished?
             gui.update_calibration(config.EMOTION_CALIBRATION_DURATION, config.EMOTION_CALIBRATION_DURATION)
            
        # Consume frame from queue
        try:
            frame = frame_queue.get_nowait()
            frame_count += 1
        except queue.Empty:
            time.sleep(0.001)
            continue
            
        # Process Frame
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = face_mesh.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array
            pts_norm = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
            
            # 1. Update Emotion Thread
            # Get bounding box for emotion detection
            bbox = bbox_from_points(pts_norm, W, H)
            emotion_thread.set_frame(frame, bbox)
            
            # 2. Get Emotion Data
            # Get smoothed probabilities
            emotion_probs = emotion_thread.smooth.copy()
            new_emotion = emotion_thread.curr_label
            
            # Handle VTube Studio Expressions based on emotion
            if vts_client and vts_client.connected and vts_client.authenticated:
                if new_emotion != current_emotion:
                    # Deactivate old expression
                    old_expr = config.EMOTION_EXPRESSIONS.get(current_emotion)
                    if old_expr:
                        vts_client.deactivate_expression(old_expr)
                    
                    # Activate new expression
                    new_expr = config.EMOTION_EXPRESSIONS.get(new_emotion)
                    if new_expr:
                        vts_client.trigger_expression(new_expr)
                        
                    current_emotion = new_emotion
            
            # Update GUI with emotion info
            gui.update_emotion(new_emotion, emotion_probs)
            
            # 3. Compute Expression Features (MediaPipe)
            mp_features = compute_expression_features(pts_norm)
            gui.update_expression(mp_features)
            
            # 4. Send to VTube Studio (Dynamic Mapping)
            if vts_client and vts_client.connected and vts_client.authenticated:
                params = {}

                for param_name, (feature_key, scale, invert) in config.VTS_PARAM_MAP.items():
                    if feature_key not in mp_features:
                        continue

                    value = mp_features[feature_key] * scale
                    if invert:
                        value = -value

                    params[param_name] = float(value)

                vts_client.send_parameters(params)
            
        else:
            # No face detected
            pass
        
        # Update GUI video feed
        gui.update_frame(frame_rgb)
            
    # Cleanup
    print("Exiting...")
    camera_cmd_queue.put("EXIT")
    emotion_thread.stop()
    if vts_client:
        vts_client.disconnect()
    gui.close()

if __name__ == "__main__":
    main()
