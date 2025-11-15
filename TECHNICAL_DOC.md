# Technical Documentation: AI-Powered Expression-to-Avatar System

## 1. Deep Learning Models

### 1.1 MediaPipe Face Mesh
- **Type**: CNN-based real-time face landmark detection
- **Output**: 468 3D facial landmarks (with `refine_landmarks=True`)
- **Performance**: ~30 FPS real-time inference
- **Features Detected**: Eye aspect ratio (EAR), mouth aspect ratio (MAR), head pose (yaw/pitch), tongue position
- **Landmark Extraction**: Extracts eyebrow positions, mouth movements, and eye states for real-time expression mapping

### 1.2 DeepFace Emotion Recognition
- **Type**: Pre-trained CNN ensemble (TensorFlow/Keras backend)
- **Input**: Face ROI extracted from video frame (resized to 224x224)
- **Output**: Probability distribution over 7 emotions: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
- **Processing**: Runs asynchronously at ~10 FPS in separate thread
- **Calibration**: Neutral baseline compensation system subtracts user-specific bias

### 1.3 Kalman Filtering
- **Algorithm**: 1D Kalman filter (position-velocity model)
- **Purpose**: Smooths noisy landmark measurements (EAR, MAR, yaw, pitch, tongue)
- **State Model**: `[position, velocity]` with constant velocity assumption

## 2. System Architecture

```
Webcam → MediaPipe (Landmarks) → Feature Extraction → Kalman Filtering
                                              ↓
                                    Emotion Thread (DeepFace)
                                              ↓
                                    Parameter Fusion & Normalization
                                              ↓
                          ┌───────────────────┴───────────────────┐
                          ↓                                       ↓
                    VTube Studio API                      GUI Display
                    (WebSocket)                           (PySide6)
                                              ↓
                                    OBS / Streaming Platforms
```

### 2.1 Component Breakdown

**Main Thread (main.py)**: 
- Video capture at ~30 FPS
- MediaPipe landmark extraction
- Geometric feature calculation (EAR, MAR, yaw, pitch, tongue)
- Kalman filter smoothing
- VTube Studio communication

**Emotion Thread (core/emotion.py)**:
- Asynchronous DeepFace processing at ~10 FPS
- EMA smoothing of emotion probabilities
- Hysteresis-based emotion switching logic
- Thread-safe communication with main thread
- Neutral baseline calibration and compensation

**GUI Layer (gui_panel.py)**:
- PySide6-based modern interface
- Real-time video display with emotion-colored borders
- Animated emotion probability bars
- Expression indicator widgets
- Calibration progress visualization

**VTube Studio Client (vtube_studio/client.py)**:
- WebSocket API communication
- Parameter injection and expression triggering
- Authentication token management

## 3. Code Structure

```
Vtuber/
├── main.py                 # Main application loop
├── config.py               # Configuration constants
├── gui_panel.py            # PySide6 GUI interface
├── core/
│   ├── landmarks.py        # EAR, MAR, yaw, pitch, tongue extraction
│   ├── emotion.py          # DeepFace emotion detection with calibration
│   ├── kalman.py           # Kalman filter implementation
│   └── drawing.py          # Normalization and visualization
├── utils/
│   └── math_utils.py       # Math helper functions
└── vtube_studio/
    └── client.py           # VTube Studio WebSocket client
```

### 3.1 Key Algorithms

**Eye Aspect Ratio (EAR)**:
- Calculates vertical eye opening normalized by horizontal width
- EAR decreases when eyes close (blink detection)
- Used to detect blinking and eye state

**Mouth Aspect Ratio (MAR)**:
- Calculates vertical mouth opening normalized by horizontal width
- Higher MAR = more open mouth
- Mapped to Live2D MouthOpen parameter

**Head Pose Estimation**:
- Uses nose position relative to face bounding box center
- Estimates yaw (left/right) and pitch (up/down) rotation
- Mapped to Live2D AngleX and AngleY parameters

**Tongue Detection**:
- Calculates tongue position relative to mouth opening
- Uses MediaPipe landmark 10 (tongue tip) with refine_landmarks=True
- Mapped to Live2D TongueOut parameter

**Emotion Processing Pipeline**:
1. DeepFace raw probabilities → Normalization
2. Baseline compensation (subtract neutral baseline if calibrated)
3. EMA smoothing (α = 0.15)
4. Hysteresis evaluation (switch when new prob > old prob + 0.05)
5. Minimum hold time enforcement (0.3s between switches)
6. Expression triggering via VTube Studio API

## 4. Technical Implementation

### 4.1 Performance Optimization
- Multi-threading: Emotion detection in separate thread to avoid blocking video pipeline
- Change thresholds: Parameters only sent when change exceeds threshold (reduces unnecessary updates)
- ROI extraction: Only face region sent to DeepFace (resized to 224x224 for optimal performance)
- Aggressive bounding box expansion: Captures full face including forehead, temples, and jawline for better emotion detection

### 4.2 Noise Reduction
- **Kalman Filtering**: Smooths continuous measurements (EAR, MAR, yaw, pitch, tongue)
- **EMA Smoothing**: Smooths discrete emotion probabilities (α = 0.15)
- **Hysteresis**: Prevents oscillation between similar emotions (0.05 margin)
- **Minimum Hold Time**: Time-based locks prevent rapid switching (0.3s)
- **Baseline Compensation**: Subtracts user-specific neutral bias from DeepFace outputs

### 4.3 Parameter Normalization
- **Linear**: `(value - min) / (max - min)` for EAR, tongue
- **Sigmoid**: `1 / (1 + exp(-gain * (value - center)))` for MAR
- **Offset**: `value + 0.5` for yaw/pitch (maps [-0.5, 0.5] → [0, 1])

### 4.4 VTube Studio Integration
- **Parameters Sent**: `EyeOpenLeft/Right`, `MouthOpen`, `AngleX/Y`, `TongueOut`
- **Expressions**: Triggered based on detected emotion (happy, sad, angry, etc.)
- **Protocol**: VTube Studio Public API v1.0 via WebSocket
- **OBS Compatibility**: Avatar can be captured in OBS via transparent window capture or VTube Studio virtual webcam
- **Streaming**: Enables real-time avatar streaming on platforms like Twitch, YouTube, or Zoom

### 4.5 User Interface
- **Framework**: PySide6 (Qt for Python)
- **Theme**: Monochrome dark theme (#0d0d0f background, #191a1d panels)
- **Emotion Colors**: Accent colors only for emotion bars and borders
- **Animations**: QPropertyAnimation with cubic easing (150-200ms duration)
- **Components**:
  - VideoDisplay: QLabel with vignette overlay and animated emotion border
  - EmotionBar: Animated probability bar with glow effect
  - ExpressionBar: Monochrome indicator bars for facial features
  - CalibrationWidget: Progress indicator for neutral calibration
- **Window**: Frameless, always-on-top, draggable, rounded corners

## 5. Neutral Baseline Calibration

### 5.1 Calibration Phase
- **Duration**: 2 seconds on application startup
- **Purpose**: Establish user-specific neutral expression baseline
- **Data Collection**: 
  - DeepFace emotion probabilities per frame
  - MediaPipe expression features (EAR, MAR, yaw, pitch, tongue)
- **Computation**: Running average of collected samples

### 5.2 Baseline Compensation
- **Method**: Subtract baseline probabilities from raw DeepFace output
- **Formula**: `adjusted[k] = max(0.0, probs_raw[k] - baseline[k])` then renormalize
- **Effect**: Only deviations from neutral are detected as emotions
- **Benefits**: 
  - Eliminates false positives (e.g., "sad lock" when user has naturally downturned mouth)
  - Accounts for individual facial structure and resting expression
  - Improves emotion detection accuracy for all users

### 5.3 Implementation
- **Location**: `core/emotion.py` - `EmotionThread` class
- **API**: `start_calibration()`, `feed_calibration()`, `finalize_calibration()`
- **Thread Safety**: Uses locks to protect calibration state
- **Backward Compatibility**: Falls back to raw probabilities if calibration not performed

### 5.4 MediaPipe Feature Support
- **Helper Functions**: `compute_expression_features()`, `mp_features_relative()` in `core/landmarks.py`
- **Features Extracted**: EAR (left/right), MAR, yaw, pitch, tongue
- **Future Use**: Prepared for MediaPipe-based emotion fusion (not yet implemented)

## 6. Limitations

- **Calibration Requirement**: The neutral calibration step assumes the user can remain still for 2 seconds at startup. Accuracy may be affected if calibration is interrupted.
- **Parameter Support**: The system outputs all landmark-derived facial features, but VTube Studio model compatibility varies. Some models may not support all parameters (e.g., TongueOut, Fear expression).
- **Model Dependencies**: Different VTube Studio models use different parameter names. Users must configure parameter mappings in `config.py` to match their specific model.

## 7. Summary

**Deep Learning Models**: MediaPipe Face Mesh (468 landmarks), DeepFace (emotion classification)
**Algorithms**: Kalman filtering, EMA smoothing, hysteresis logic, baseline compensation
**Performance**: ~30 FPS video processing, ~10 FPS emotion detection, <100ms latency
**Integration**: Real-time WebSocket communication with VTube Studio for live avatar animation
**Calibration**: Automatic neutral baseline calibration for improved emotion accuracy
**User Interface**: Modern PySide6 GUI with monochrome theme and emotion-based accent colors
**Streaming**: OBS-compatible via VTube Studio virtual webcam or window capture

