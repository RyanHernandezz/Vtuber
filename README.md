# AI-Powered Expression-to-Avatar System

A real-time VTuber avatar system that converts facial expressions into animated character movements using MediaPipe face detection and DeepFace emotion recognition. The system processes real-time video to extract emotional cues and facial movements, then sends the data to VTube Studio via WebSocket API for Live2D avatar control.

## Requirements

- **Python 3.9 - 3.12** (tested on Python 3.12.7)
  
  **Note**: TensorFlow 2.15.0+ requires Python 3.9+. Python 3.8 is not supported by TensorFlow 2.15.0.
- Webcam
- Virtual environment (recommended)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main VTuber avatar script:
```bash
python main.py
```

## Project Structure

```
Vtuber/
├── main.py              # Main application entry point
├── config.py            # Configuration constants
├── gui_panel.py         # Modern PySide6 GUI interface
├── core/                # Core functionality modules
│   ├── emotion.py       # Emotion detection (DeepFace) with calibration
│   ├── landmarks.py     # Face landmark processing
│   ├── drawing.py       # Avatar drawing functions
│   └── kalman.py        # Kalman filter for smoothing
├── utils/               # Utility functions
│   └── math_utils.py    # Math helper functions
└── vtube_studio/        # VTube Studio integration
```

## Features

### Core Tracking
- **Real-time face tracking** using MediaPipe (468 landmarks)
- **Advanced stability** with enhanced Kalman filtering:
  - Per-landmark confidence weighting based on velocity
  - Region-based smoothing (stable anchors vs expressive features)
  - Outlier detection and rejection
  - Velocity clamping for jitter prevention
- Eye aspect ratio (EAR) detection for blinking
- Mouth aspect ratio (MAR) detection for mouth movement
- Head pose estimation (yaw, pitch)
- Tongue detection for expressive animations

### Emotion Detection
- **DeepFace emotion recognition** with 7 emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- **Delta-based calibration** - your neutral face becomes the baseline
- Emotions only appear after calibration is complete
- Top 3 emotions displayed with real-time probabilities

### User Interface
- **Modern monochrome GUI** with emotion-based accent colors
- **FPS counter** showing real-time performance
- **Live video feed** with emotion-colored border
- **Expression indicators** with percentage values (mouth, eyes, head pose)
- **Calibration progress** with visual feedback
- **Camera Toggle** with responsive control and "Webcam OFF" visual
- **VTube Studio Status** indicator (connected/disconnected)
- **Frameless, always-on-top window** with drag support

## Controls

- Close the window to exit
- Drag the window by clicking and holding anywhere on it

## Documentation

For detailed technical documentation including deep learning models, architecture, and code explanations, see [TECHNICAL_DOC.md](TECHNICAL_DOC.md).

## VTube Studio Setup

To use VTube Studio integration:

1. **Install VTube Studio** (if not already installed):
   - Download from [VTube Studio website](https://denchisoft.com/)
   - Install and launch VTube Studio

2. **Install OBS Spout2 Plugin** (recommended for OBS streaming):
   - Download OBS Spout2 Plugin from the [OBS Spout2 Plugin GitHub releases](https://github.com/Off-World-Live/obs-spout2-plugin/releases)
   - Install Spout2 on your system
   - Spout2 enables low-latency texture sharing between VTube Studio and OBS for better streaming performance

3. **Enable API in VTube Studio**:
   - Open VTube Studio settings
   - Go to "API" or "Plugin Settings"
   - Enable "Allow API connections"
   - Note the WebSocket port (default is `8001`)

4. **Configure the application**:
   - Edit `config.py` and set `USE_VTUBE_STUDIO = True`
   - If using a different port, update `VTUBE_STUDIO_URL` (default: `ws://localhost:8001`)

5. **First-time authentication**:
   - When you run the application, VTube Studio will show an authentication prompt
   - Click "Allow" to authorize the plugin
   - The authentication token will be saved for future sessions

6. **Parameter mapping**:
   The application sends the following parameters to VTube Studio:
   - `EyeOpenLeft` / `EyeOpenRight`: Eye blink detection (0.0 = closed, 1.0 = open)
   - `MouthOpen`: Mouth opening (0.0 = closed, 1.0 = open)
   - `AngleX`: Head yaw/rotation (-30° to +30°)
   - `AngleY`: Head pitch/tilt (-30° to +30°)
   
   **Important**: Different VTube Studio models use different parameter names. If you get parameter errors:
   - The application will automatically query and display available parameters when it connects
   - Edit `config.py` and update the `VTUBE_STUDIO_PARAMS` dictionary with your model's actual parameter names
   - Set a parameter to `None` in the config to disable sending that parameter
   - You can find your model's parameter names in VTube Studio: Settings → Model Settings → Input Parameters

7. **Expression mapping** (optional):
   - Edit `config.py` and update the `EMOTION_EXPRESSIONS` dictionary with your actual VTube Studio expression file names
   - Expression files should match the emotion names: happy, sad, angry, surprise, fear, disgust, neutral
   - Set to `None` for emotions that don't have corresponding expressions

**Note**: Make sure VTube Studio is running before starting the application, or the connection will fail.

8. **OBS Integration**:
   - The finished avatar can be viewed in OBS through VTube Studio's built-in output features
   - **Recommended**: Use Spout2 source in OBS for low-latency streaming (requires Spout2 installed)
   - **Alternative**: Use transparent window capture or VTube Studio's virtual webcam
   - This enables streaming the avatar directly on platforms like Twitch, YouTube, or Zoom
   - The avatar responds in real time to your facial expressions and emotions

## Neutral Calibration

The system uses **delta-based calibration** to personalize emotion detection:

### How It Works

1. **Click "Begin Calibration"** button in the GUI
2. **3-second countdown** - get ready
3. **2-second calibration phase** - relax your face and look at the camera
   - Emotions **do not update** during this time
   - Top 3 emotions show `--` (blank)
4. **Baseline established** - your neutral face becomes the zero point
5. **Emotions appear** - only deviations from YOUR neutral face are detected

### Benefits
- **Eliminates false positives** - no more "sad lock" if you have a naturally downturned mouth
- **Personalized to you** - accounts for individual facial structure
- **Delta-based detection** - only changes from your baseline are detected as emotions
- **Formula**: `emotion = raw_detection - your_baseline`

### Best Practices
- Relax your face during calibration
- Look directly at the camera
- Don't smile or frown - just be natural
- Recalibrate if lighting changes significantly

**Configuration**: Can be disabled by setting `EMOTION_CALIBRATION_ENABLED = False` in `config.py`

## User Interface

The application features a modern, minimal GUI built with PySide6:

### Main Display
- **FPS Counter**: Real-time frame rate display in title bar (color-coded: green/yellow/red)
- **Video Display**: Live camera feed with soft vignette and emotion-colored border
- **Emotion Label**: Large, clear current emotion display
- **Top 3 Emotions**: Real-time probabilities with color-coded text
  - Shows `--` before calibration is complete
  - Updates only after neutral baseline is established

### Expression Indicators
- **Percentage values** next to each bar (e.g., "mouth 45%")
- Clean white progress bars (no grey background)
- Real-time updates for:
  - Mouth opening
  - Left/right eye opening
  - Head yaw (left/right)
  - Head pitch (up/down)

### Calibration
- **"Begin Calibration" button** - click to start
- **Progress indicator** - shows calibration status
- **Visual feedback** - countdown and progress bar

### Theme
- **Dark monochrome design** with emotion-specific accent colors
- **Frameless window** - always on top, draggable
- **Rounded corners** - modern, polished look

### Camera Controls
- **Camera Toggle Button**: Turns the camera on/off.
  - **OFF State**: Shows a black screen with "Webcam OFF" text.
  - **ON State**: Shows the live video feed.
- **VTube Studio Status**:
  - **Green Dot**: Connected to VTube Studio.
  - **Grey Dot**: Disconnected.

The GUI replaces all console output with visual feedback. Debug information still prints to the console for troubleshooting.

## Troubleshooting

### Camera Not Opening / "Cannot open webcam" Error

If you get a "Cannot open webcam" error or the GUI closes immediately, try these solutions in order:

1. **Close VTube Studio** - VTube Studio often auto-starts with the camera enabled, which locks the webcam
   - Either close VTube Studio completely before running this tracker
   - Or disable "Start camera on launch" in VTube Studio settings
   
2. **Close other camera applications** - Check for:
   - Zoom, Microsoft Teams, Skype
   - OBS Studio (if using webcam source)
   - Windows Camera app
   - Any other video conferencing or streaming software

3. **Try a different camera index** - If you have multiple cameras:
   - Edit `config.py` and change `CAM_INDEX` from `0` to `1` or `2`
   - The program will automatically try multiple indices, but setting the correct one helps

4. **Check camera permissions**:
   - Open Windows Settings → Privacy → Camera
   - Make sure "Allow apps to access your camera" is enabled
   - Ensure Python is allowed to use the camera

5. **Test your camera** - Open the Windows Camera app to verify your webcam works

### DeepFace Not Working

If you see "WARNING: DeepFace is not working":
- This is usually fine - the tracker will still work without emotion detection
- DeepFace may download models on first run (requires internet connection)
- Check that TensorFlow installed correctly: `pip install tensorflow`

### VTube Studio Not Connecting

If you see "Warning: Could not connect to VTube Studio":
- Make sure VTube Studio is running
- Enable API in VTube Studio: Settings → API → "Allow API connections"
- Check that the port matches (default: 8001)
- The tracker will still work without VTube Studio connection

## Notes

- The enhanced version uses DeepFace for emotion detection, which may download models on first run
- Make sure your webcam is connected and accessible
- Camera index can be changed in the `CAM_INDEX` configuration variable
- During the 2-second calibration phase, keep your face relaxed and neutral for best results
- The GUI requires PySide6 (included in requirements.txt)

