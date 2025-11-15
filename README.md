# AI-Powered Expression-to-Avatar System

A real-time VTuber avatar system that converts facial expressions into animated character movements using MediaPipe face detection and DeepFace emotion recognition. The system processes real-time video to extract emotional cues and facial movements, then sends the data to VTube Studio via WebSocket API for Live2D avatar control.

## Requirements

- **Python 3.8 - 3.11** (Python 3.12+ not tested)
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

- Real-time face tracking using MediaPipe
- Eye aspect ratio (EAR) detection for blinking
- Mouth aspect ratio (MAR) detection for mouth movement
- Head pose estimation (yaw, pitch)
- Emotion detection using DeepFace with neutral baseline calibration
- Kalman filtering for smooth animations
- Modern monochrome GUI with emotion-based accent colors
- Automatic neutral face calibration for improved emotion accuracy
- Animated emotion bars and expression indicators
- Frameless, always-on-top window with drag support

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

The system includes an automatic neutral baseline calibration system that improves emotion detection accuracy:

- **On startup**: The system performs a 2-second calibration phase where you should relax your face
- **Baseline compensation**: DeepFace probabilities are adjusted by subtracting your personal neutral baseline
- **Benefits**: Reduces false positives (e.g., "sad lock") by accounting for individual facial structure
- **Configuration**: Can be disabled by setting `EMOTION_CALIBRATION_ENABLED = False` in `config.py`

During calibration, the system collects DeepFace emotion probabilities and computes a baseline distribution. After calibration, all emotion detections are adjusted relative to this baseline, so only deviations from your neutral expression are detected as emotions.

## User Interface

The application features a modern, minimal GUI built with PySide6:

- **Video Display**: Live camera feed with soft vignette and emotion-colored border
- **Emotion Bar**: Animated probability bar showing current dominant emotion
- **Expression Indicators**: Monochrome bars for mouth, eyes, yaw, and pitch
- **Calibration Widget**: Progress indicator during neutral calibration phase
- **Theme**: Dark monochrome design with emotion-specific accent colors

The GUI replaces all console output with visual feedback. Debug information still prints to the console for troubleshooting.

## Notes

- The enhanced version uses DeepFace for emotion detection, which may download models on first run
- Make sure your webcam is connected and accessible
- Camera index can be changed in the `CAM_INDEX` configuration variable
- During the 2-second calibration phase, keep your face relaxed and neutral for best results
- The GUI requires PySide6 (included in requirements.txt)

