import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from core.emotion import emotion_to_color

class EmotionBar(QWidget):
    """Animated emotion probability bar with glow effect."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        self._value = 0.0
        self._target_value = 0.0
        self._color = QColor(215, 215, 224)  # neutral
        self._target_color = QColor(215, 215, 224)
        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._color_animation = QPropertyAnimation(self, b"color")
        self._color_animation.setDuration(200)
        self._color_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
    
    def setValue(self, value):
        self._value = max(0.0, min(1.0, float(value)))
        self.update()
    
    def getValue(self):
        return self._value
    
    value = property(getValue, setValue)
    
    def setColor(self, color):
        if isinstance(color, QColor):
            self._color = color
        else:
            self._color = QColor(color)
        self.update()
    
    def getColor(self):
        return self._color
    
    color = property(getColor, setColor)
    
    def animateTo(self, value, color):
        """Animate bar to new value and color."""
        self._target_value = max(0.0, min(1.0, float(value)))
        if isinstance(color, QColor):
            self._target_color = color
        else:
            self._target_color = QColor(color)
        
        self._animation.stop()
        self._animation.setStartValue(self._value)
        self._animation.setEndValue(self._target_value)
        self._animation.start()
        
        self._color_animation.stop()
        self._color_animation.setStartValue(self._color)
        self._color_animation.setEndValue(self._target_color)
        self._color_animation.start()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        bar_width = int(rect.width() * self._value)
        
        # Background
        bg_color = QColor(25, 26, 29)
        painter.fillRect(rect, bg_color)
        
        if bar_width > 0:
            bar_rect = rect.adjusted(0, 0, -(rect.width() - bar_width), 0)
            
            # Glow effect (subtle)
            glow_pen = QPen(self._color, 2)
            glow_pen.setCosmetic(True)
            painter.setPen(glow_pen)
            painter.setBrush(QBrush(self._color))
            painter.drawRoundedRect(bar_rect, 4, 4)
            
            # Main bar
            bar_color = QColor(self._color)
            bar_color.setAlpha(180)
            painter.fillRect(bar_rect, bar_color)

class CalibrationWidget(QWidget):
    """Calibration progress indicator."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        self.status_label = QLabel("calibrating")
        self.status_label.setStyleSheet("color: #d7d7e0; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #191a1d;
                border-radius: 4px;
                background: #0d0d0f;
                height: 8px;
            }
            QProgressBar::chunk {
                background: #7fffe1;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress)
        
        self.hide()
    
    def updateProgress(self, elapsed, duration):
        """Update calibration progress (0.0 to 1.0)."""
        progress = min(100, int((elapsed / duration) * 100))
        self.progress.setValue(progress)
    
    def setReady(self):
        """Mark calibration as complete."""
        self.status_label.setText("ready")
        self.progress.setValue(100)
        QTimer.singleShot(1000, self.hide)

class ExpressionBar(QWidget):
    """Monochrome expression indicator bar."""
    
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(20)
        self.setMaximumHeight(20)
        self._value = 0.0
        self._label = label
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet("color: #7a7a85; font-size: 10px;")
        self.label_widget.setMinimumWidth(50)
        layout.addWidget(self.label_widget)
    
    def setValue(self, value):
        self._value = max(0.0, min(1.0, float(value)))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        bar_width = int((rect.width() - 60) * self._value)
        
        # Background
        bg_color = QColor(25, 26, 29)
        painter.fillRect(rect, bg_color)
        
        if bar_width > 0:
            bar_rect = rect.adjusted(55, 2, -(rect.width() - 55 - bar_width), -2)
            bar_color = QColor(122, 122, 133)
            painter.fillRect(bar_rect, bar_color)

class VideoDisplay(QLabel):
    """Video feed display with vignette and emotion border."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #0d0d0f;")
        self._border_color = QColor(215, 215, 224)
        self._target_border_color = QColor(215, 215, 224)
        self._border_red = 215
        self._border_green = 215
        self._border_blue = 224
        self._border_animation = QPropertyAnimation(self, b"borderRed")
        self._border_animation.setDuration(150)
        self._border_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._border_animation_g = QPropertyAnimation(self, b"borderGreen")
        self._border_animation_g.setDuration(150)
        self._border_animation_g.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._border_animation_b = QPropertyAnimation(self, b"borderBlue")
        self._border_animation_b.setDuration(150)
        self._border_animation_b.setEasingCurve(QEasingCurve.Type.InOutCubic)
    
    def setBorderRed(self, value):
        self._border_red = int(value)
        self._border_color = QColor(self._border_red, self._border_green, self._border_blue)
        self.update()
    
    def getBorderRed(self):
        return self._border_red
    
    borderRed = property(getBorderRed, setBorderRed)
    
    def setBorderGreen(self, value):
        self._border_green = int(value)
        self._border_color = QColor(self._border_red, self._border_green, self._border_blue)
        self.update()
    
    def getBorderGreen(self):
        return self._border_green
    
    borderGreen = property(getBorderGreen, setBorderGreen)
    
    def setBorderBlue(self, value):
        self._border_blue = int(value)
        self._border_color = QColor(self._border_red, self._border_green, self._border_blue)
        self.update()
    
    def getBorderBlue(self):
        return self._border_blue
    
    borderBlue = property(getBorderBlue, setBorderBlue)
    
    def animateBorderColor(self, color):
        """Animate border color change."""
        if isinstance(color, QColor):
            target = color
        else:
            target = QColor(color)
        
        self._border_animation.stop()
        self._border_animation.setStartValue(self._border_red)
        self._border_animation.setEndValue(target.red())
        self._border_animation.start()
        
        self._border_animation_g.stop()
        self._border_animation_g.setStartValue(self._border_green)
        self._border_animation_g.setEndValue(target.green())
        self._border_animation_g.start()
        
        self._border_animation_b.stop()
        self._border_animation_b.setStartValue(self._border_blue)
        self._border_animation_b.setEndValue(target.blue())
        self._border_animation_b.start()
    
    def setFrame(self, frame_rgb):
        """Update video frame from numpy array."""
        if frame_rgb is None or frame_rgb.size == 0:
            return
        
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        # Ensure contiguous array for QImage
        if not frame_rgb.flags['C_CONTIGUOUS']:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Apply vignette effect
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        vignette_rect = pixmap.rect()
        center = vignette_rect.center()
        radius = max(vignette_rect.width(), vignette_rect.height()) * 0.7
        
        # Soft vignette using radial gradient
        for i in range(5):
            alpha = int(15 * (1 - i / 5))
            pen = QPen(QColor(13, 13, 15, alpha), 20 - i * 4)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, int(radius + i * 10), int(radius + i * 10))
        
        painter.end()
        
        # Scale to fit display while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Draw emotion border
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        pen = QPen(self._border_color, 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)

class MainWindow(QMainWindow):
    """Main VTuber tracking UI window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTuber Tracker")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Central widget with rounded corners
        central = QWidget()
        central.setStyleSheet("""
            QWidget {
                background: #0d0d0f;
                border-radius: 12px;
            }
        """)
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Video display (left)
        self.video_display = VideoDisplay()
        main_layout.addWidget(self.video_display, 3)
        
        # Side panel (right)
        panel = QWidget()
        panel.setStyleSheet("background: #191a1d; border-radius: 0 12px 12px 0;")
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(280)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(15)
        
        # Calibration widget
        self.calibration_widget = CalibrationWidget()
        panel_layout.addWidget(self.calibration_widget)
        
        # Emotion tag
        self.emotion_label = QLabel("neutral")
        self.emotion_label.setStyleSheet("color: #d7d7e0; font-size: 24px; font-weight: 300;")
        panel_layout.addWidget(self.emotion_label)
        
        # Emotion bar
        self.emotion_bar = EmotionBar()
        panel_layout.addWidget(self.emotion_bar)
        
        # Expression indicators
        panel_layout.addWidget(QLabel("expression"))  # Section header
        header_style = "color: #7a7a85; font-size: 11px; font-weight: 500;"
        for label in panel_layout.findChildren(QLabel):
            if label.text() == "expression":
                label.setStyleSheet(header_style)
        
        self.expression_bars = {}
        for name in ["mouth", "eye_l", "eye_r", "yaw", "pitch"]:
            bar = ExpressionBar(name)
            self.expression_bars[name] = bar
            panel_layout.addWidget(bar)
        
        panel_layout.addStretch()
        main_layout.addWidget(panel)
        
        # Emotion color mapping
        self.emotion_colors = {
            'angry': QColor(255, 107, 107),
            'sad': QColor(125, 178, 255),
            'happy': QColor(127, 255, 150),
            'surprise': QColor(138, 255, 255),
            'fear': QColor(199, 146, 255),
            'disgust': QColor(184, 255, 221),
            'neutral': QColor(215, 215, 224)
        }
        
        # Current state
        self._current_emotion = 'neutral'
        self._fade_animation = QPropertyAnimation(self.emotion_label, b"styleSheet")
        self._fade_animation.setDuration(200)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
    
    def update_frame(self, frame_rgb):
        """Update video frame display."""
        if frame_rgb is not None:
            self.video_display.setFrame(frame_rgb)
    
    def update_emotion(self, emotion_label, emotion_probs):
        """Update emotion display and bar."""
        if emotion_label != self._current_emotion:
            self._current_emotion = emotion_label
            
            # Update label with fade
            color = self.emotion_colors.get(emotion_label, self.emotion_colors['neutral'])
            if emotion_label == 'neutral':
                style = "color: #d7d7e0; font-size: 24px; font-weight: 300;"
            else:
                style = f"color: rgb({color.red()}, {color.green()}, {color.blue()}); font-size: 24px; font-weight: 300;"
            
            self._fade_animation.stop()
            self._fade_animation.setStartValue(self.emotion_label.styleSheet())
            self._fade_animation.setEndValue(style)
            self._fade_animation.start()
            
            self.emotion_label.setText(emotion_label.lower())
            
            # Update border color
            self.video_display.animateBorderColor(color)
        
        # Update bar
        prob = emotion_probs.get(emotion_label, 0.0) if emotion_probs else 0.0
        color = self.emotion_colors.get(emotion_label, self.emotion_colors['neutral'])
        self.emotion_bar.animateTo(prob, color)
    
    def update_expression(self, mp_features):
        """Update expression indicator bars."""
        if not mp_features:
            return
        
        # Map MediaPipe features to bars
        mapping = {
            'mouth': mp_features.get('mar', 0.0),
            'eye_l': mp_features.get('ear_left', 0.0),
            'eye_r': mp_features.get('ear_right', 0.0),
            'yaw': (mp_features.get('yaw', 0.0) + 0.5),  # Normalize -0.5 to 0.5 -> 0 to 1
            'pitch': (mp_features.get('pitch', 0.0) + 0.5)
        }
        
        for name, value in mapping.items():
            if name in self.expression_bars:
                self.expression_bars[name].setValue(value)
    
    def update_calibration(self, elapsed, duration):
        """Update calibration progress."""
        if elapsed < duration:
            self.calibration_widget.show()
            self.calibration_widget.updateProgress(elapsed, duration)
        else:
            self.calibration_widget.setReady()
    
    def mousePressEvent(self, event):
        """Enable window dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle window dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, '_drag_position'):
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

class VTuberGUI:
    """GUI manager for VTuber tracking application."""
    
    def __init__(self):
        self.app = None
        self._window = None
        self._initialized = False
    
    def initialize(self):
        """Initialize Qt application and main window."""
        if self._initialized:
            return
        
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        self._window = MainWindow()
        self._window.resize(1280, 720)
        self._window.show()
        self._initialized = True
    
    @property
    def window(self):
        """Get main window instance."""
        return self._window
    
    def update_frame(self, frame_rgb):
        """Update video frame."""
        if self.window:
            self.window.update_frame(frame_rgb)
    
    def update_emotion(self, emotion_label, emotion_probs):
        """Update emotion display."""
        if self.window and emotion_label and emotion_probs:
            self.window.update_emotion(emotion_label, emotion_probs)
    
    def update_expression(self, mp_features):
        """Update expression indicators."""
        if self.window and mp_features:
            self.window.update_expression(mp_features)
    
    def update_calibration(self, elapsed, duration):
        """Update calibration progress."""
        if self.window:
            self.window.update_calibration(elapsed, duration)
    
    def process_events(self):
        """Process Qt events (call in main loop)."""
        if self.app:
            self.app.processEvents()
    
    def close(self):
        """Close GUI."""
        if self.window:
            self.window.close()
        if self.app:
            self.app.quit()

