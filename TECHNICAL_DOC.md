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

### 1.3 Advanced Kalman Filtering
- **Algorithm**: Enhanced 1D Kalman filter with confidence weighting
- **Features**:
  - **Per-landmark confidence scoring**: Velocity-based confidence (fast movement = low confidence)
  - **Region-based smoothing**: Stable anchors (80%), medium regions (50%), expressive features (20%)
  - **Outlier rejection**: Detects and ignores sudden jumps from MediaPipe glitches
  - **Velocity clamping**: Limits maximum movement speed to prevent jitter
  - **Confidence-weighted Kalman gain**: Blends Kalman prediction with raw measurement based on confidence
- **Coverage**: All 468 MediaPipe landmarks (936 filters total: X and Y per landmark)
- **Configurable**: Can be disabled entirely or per-feature if performance is a concern
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
- MediaPipe landmark extraction (468 landmarks)
- **Advanced tracking**: Per-landmark Kalman filtering with confidence weighting
- Geometric feature calculation (EAR, MAR, yaw, pitch, tongue)
- Basic Kalman filter smoothing for derived features
- VTube Studio communication
- FPS calculation and display

**Emotion Thread (core/emotion.py)**:
- Asynchronous DeepFace processing at ~10 FPS
- EMA smoothing of emotion probabilities
- Hysteresis-based emotion switching logic
- Thread-safe communication with main thread
- Neutral baseline calibration and compensation

**GUI Layer (gui_panel.py)**:
- PySide6-based modern interface
- Real-time FPS counter with color coding
- Live video display with emotion-colored borders
- Top 3 emotion probabilities (blank before calibration)
- Expression indicators with percentage values
- Calibration progress visualization
- Clean, minimal design (removed progress bars)

**VTube Studio Client (vtube_studio/client.py)**:
- WebSocket API communication
- Parameter injection and expression triggering
- Authentication token management

## 3. Code Structure

```
Vtuber/
├── main.py                 # Main application loop with advanced tracking
├── config.py               # Configuration constants (including advanced tracking settings)
├── gui_panel.py            # PySide6 GUI interface
├── core/
│   ├── landmarks.py        # EAR, MAR, yaw, pitch, tongue extraction
│   ├── emotion.py          # DeepFace emotion detection with calibration
│   ├── kalman.py           # Basic Kalman filter implementation
│   ├── advanced_kalman.py  # Advanced Kalman with confidence weighting
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

**Advanced Landmark Tracking**:
- **Per-landmark confidence**: `confidence = 1.0 / (1.0 + velocity)`
  - Fast-moving landmarks (noise) → low confidence
  - Stable landmarks (anchors) → high confidence
- **Region-based smoothing**:
  - Stable anchors (inner eyes, nose): 80% smoothing weight
  - Medium regions (cheeks, jaw): 50% smoothing weight
  - Expressive features (lips, eyelids): 20% smoothing weight
- **Outlier rejection**: Rejects measurements that deviate too far from prediction
- **Velocity clamping**: Limits maximum movement speed per frame
- **Confidence-weighted Kalman gain**: `K_weighted = confidence * K`

**Feature-Level Smoothing**:
- **Kalman Filtering**: Smooths derived features (EAR, MAR, yaw, pitch, tongue)
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
- **Emotion Colors**: Accent colors only for emotion text and borders
- **Animations**: QPropertyAnimation with cubic easing (150-200ms duration)
- **Components**:
  - **FPS Counter**: Real-time frame rate display with color coding (green/yellow/red)
  - **VideoDisplay**: QLabel with vignette overlay and animated emotion border
  - **Top 3 Emotions**: Real-time probabilities, blank (`--`) before calibration
  - **ExpressionBar**: Clean white bars with percentage values (no grey background)
  - **CalibrationWidget**: Progress indicator for neutral calibration
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
- **GUI Integration**: Emotions show as blank (`--`) until calibration is complete
- **User Flow**:
  1. Click "Begin Calibration" button
  2. 3-second countdown
  3. 2-second calibration (emotions frozen)
  4. Baseline established
  5. Emotions appear with delta compensation

### 5.4 MediaPipe Feature Support
- **Helper Functions**: `compute_expression_features()`, `mp_features_relative()` in `core/landmarks.py`
- **Features Extracted**: EAR (left/right), MAR, yaw, pitch, tongue
- **Future Use**: Prepared for MediaPipe-based emotion fusion (not yet implemented)

## 6. Advanced Tracking Configuration

All advanced tracking features are configurable in `config.py`:

### Master Toggle
```python
USE_ADVANCED_TRACKING = True  # Enable/disable all advanced features
```

### Individual Features
```python
CONFIDENCE_WEIGHTING = True   # Velocity-based confidence scoring
OUTLIER_REJECTION = True      # Reject outlier measurements
VELOCITY_CLAMPING = True      # Clamp excessive velocities
REGION_SMOOTHING = True       # Apply region-based smoothing
```

### Tuning Parameters
```python
MIN_CONFIDENCE = 0.1                # Minimum confidence value
VELOCITY_CONFIDENCE_SCALE = 0.05    # Velocity → confidence sensitivity
OUTLIER_THRESHOLD = 0.05            # Distance threshold for outliers
MAX_LANDMARK_VELOCITY = 0.1         # Maximum allowed velocity

STABLE_REGION_WEIGHT = 0.8          # Smoothing for stable anchors
MEDIUM_REGION_WEIGHT = 0.5          # Smoothing for medium regions
EXPRESSIVE_REGION_WEIGHT = 0.2      # Smoothing for expressive features
```

## 7. Limitations

- **Calibration Requirement**: The neutral calibration step assumes the user can remain still for 2 seconds at startup. Accuracy may be affected if calibration is interrupted.
- **Parameter Support**: The system outputs all landmark-derived facial features, but VTube Studio model compatibility varies. Some models may not support all parameters (e.g., TongueOut, Fear expression).
- **Model Dependencies**: Different VTube Studio models use different parameter names. Users must configure parameter mappings in `config.py` to match their specific model.

## 8. Summary

**Deep Learning Models**: MediaPipe Face Mesh (468 landmarks), DeepFace (emotion classification)
**Advanced Tracking**: Enhanced Kalman filtering with confidence weighting, outlier rejection, velocity clamping
**Algorithms**: Kalman filtering, EMA smoothing, hysteresis logic, delta-based calibration
**Performance**: Real-time processing with FPS counter for monitoring, ~10 FPS emotion detection, <100ms latency
**Integration**: Real-time WebSocket communication with VTube Studio for live avatar animation
**Calibration**: Delta-based neutral baseline calibration for personalized emotion detection
**User Interface**: Modern PySide6 GUI with FPS counter, clean expression bars, and calibration flow
**Streaming**: OBS-compatible via VTube Studio virtual webcam or window capture

