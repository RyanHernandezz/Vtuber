import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame, QPushButton
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
        if elapsed >= duration:
            progress = 100
        else:
            progress = round((elapsed / duration) * 100)
        self.progress.setValue(progress)
    
    def setReady(self):
        """Mark calibration as complete."""
        self.status_label.setText("ready")
        self.progress.setValue(100)
        QTimer.singleShot(1000, self.hide)
    
    def setCountdown(self, count):
        """Set countdown text."""
        if count > 0:
            self.status_label.setText(f"starting in {count}...")
            self.show()
        else:
            self.status_label.setText("calibrating")

class ExpressionBar(QWidget):
    """Monochrome expression indicator bar with value display."""
    
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(24)
        self.setMaximumHeight(24)
        self._value = 0.0
        self._label = label
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet("color: #7a7a85; font-size: 10px;")
        self.label_widget.setMinimumWidth(40)
        layout.addWidget(self.label_widget)
        
        # Value label
        self.value_label = QLabel("0%")
        self.value_label.setStyleSheet("color: #d7d7e0; font-size: 10px; font-family: monospace;")
        self.value_label.setMinimumWidth(35)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.value_label)
    
    def setValue(self, value):
        self._value = max(0.0, min(1.0, float(value)))
        self.value_label.setText(f"{int(self._value * 100)}%")
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw bar starting after label and value
        rect = self.rect()
        bar_start_x = 85  # After label (40) + value (35) + spacing (10)
        available_width = rect.width() - bar_start_x - 5
        bar_width = int(available_width * self._value)
        
        # No background - transparent
        
        if bar_width > 0:
            # Create bar rectangle: x, y, width, height
            bar_rect = rect.adjusted(0, 0, 0, 0)
            bar_rect.setLeft(bar_start_x)
            bar_rect.setTop(rect.top() + 4)
            bar_rect.setWidth(bar_width)
            bar_rect.setHeight(rect.height() - 8)
            
            bar_color = QColor(215, 215, 224)  # White/light grey
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
        
        self._camera_active = True
        
    def setCameraActive(self, active):
        self._camera_active = active
        if not active:
            self.clear()  # Clear current pixmap
        self.update()
    
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
        
        if not self._camera_active:
            # Draw black background
            painter.fillRect(rect, Qt.black)
            
            # Draw "Webcam OFF" text
            painter.setPen(QColor(122, 122, 133))
            font = painter.font()
            font.setPointSize(16)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Webcam OFF")
        
        pen = QPen(self._border_color, 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)

class TitleBar(QWidget):
    """Custom title bar with window controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QWidget {
                background: #191a1d;
                border-radius: 12px 12px 0 0;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 10, 0)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("VTuber Tracker")
        title.setStyleSheet("color: #d7d7e0; font-size: 14px; font-weight: 500;")
        layout.addWidget(title)
        
        # Spacer between title and FPS
        layout.addSpacing(15)
        
        # FPS counter
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #7a7a85; font-size: 11px; font-family: monospace;")
        layout.addWidget(self.fps_label)
        
        layout.addStretch()
        
        # Window controls
        self.minimize_btn = QPushButton("−")
        self.minimize_btn.setFixedSize(30, 30)
        self.minimize_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #d7d7e0;
                border: none;
                border-radius: 4px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2a2b2e;
            }
        """)
        self.minimize_btn.clicked.connect(self.window().showMinimized)
        
        self.maximize_btn = QPushButton("□")
        self.maximize_btn.setFixedSize(30, 30)
        self.maximize_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #d7d7e0;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2a2b2e;
            }
        """)
        self.maximize_btn.clicked.connect(self._toggle_maximize)
        
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #d7d7e0;
                border: none;
                border-radius: 4px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff4444;
                color: white;
            }
        """)
        self.close_btn.clicked.connect(self.window().close)
        
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)
    
    def _toggle_maximize(self):
        if self.window().isMaximized():
            self.window().showNormal()
        else:
            self.window().showMaximized()
    
    def mousePressEvent(self, event):
        """Enable window dragging from title bar."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle window dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, '_drag_position'):
            self.window().move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

class MainWindow(QMainWindow):
    """Main VTuber tracking UI window."""
    
    calibration_requested = Signal()
    camera_toggled = Signal(bool)  # New signal: True=On, False=Off
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTuber Tracker")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Main container
        main_container = QWidget()
        main_container.setStyleSheet("""
            QWidget {
                background: #0d0d0f;
                border-radius: 12px;
            }
        """)
        
        container_layout = QVBoxLayout(main_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # Title bar
        self.title_bar = TitleBar(self)
        container_layout.addWidget(self.title_bar)
        
        # Central widget
        central = QWidget()
        central.setStyleSheet("""
            QWidget {
                background: #0d0d0f;
                border-radius: 0 0 12px 12px;
            }
        """)
        container_layout.addWidget(central)
        
        self.setCentralWidget(main_container)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Video display (left)
        self.video_display = VideoDisplay()
        main_layout.addWidget(self.video_display, 3)
        
        # Side panel (right)
        panel = QWidget()
        panel.setStyleSheet("background: #191a1d; border-radius: 0 0 12px 0;")
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(280)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(15)
        
        # Calibration button
        self.calibration_button = QPushButton("Begin Calibration")
        self.calibration_button.setStyleSheet("""
            QPushButton {
                background: #2a2b2e;
                color: #d7d7e0;
                border: 1px solid #3a3b3e;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3a3b3e;
                border-color: #4a4b4e;
            }
            QPushButton:pressed {
                background: #1a1b1e;
            }
            QPushButton:disabled {
                background: #1a1b1e;
                color: #7a7a85;
                border-color: #2a2b2e;
            }
        """)
        self.calibration_button.clicked.connect(self._on_calibration_clicked)
        panel_layout.addWidget(self.calibration_button)
        
        # Camera Toggle button
        self.camera_button = QPushButton("Camera: ON")
        self.camera_button.setCheckable(True)
        self.camera_button.setChecked(True)
        self.camera_button.setStyleSheet("""
            QPushButton {
                background: #2a2b2e;
                color: #d7d7e0;
                border: 1px solid #3a3b3e;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3a3b3e;
                border-color: #4a4b4e;
            }
            QPushButton:checked {
                background: #2a2b2e;
                color: #7fffe1;
                border-color: #7fffe1;
            }
            QPushButton:!checked {
                background: #1a1b1e;
                color: #ff6b6b;
                border-color: #ff6b6b;
            }
        """)
        self.camera_button.clicked.connect(self._on_camera_toggled)
        panel_layout.addWidget(self.camera_button)
        
        # Calibration widget
        self.calibration_widget = CalibrationWidget()
        panel_layout.addWidget(self.calibration_widget)
        
        # Countdown timer
        self._countdown_timer = QTimer()
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_value = 0
        
        # Emotion tag
        self.emotion_label = QLabel("neutral")
        self.emotion_label.setStyleSheet("color: #d7d7e0; font-size: 24px; font-weight: 300;")
        panel_layout.addWidget(self.emotion_label)
        
        # Top 3 emotion probabilities
        prob_header = QLabel("top emotions")
        prob_header.setStyleSheet("color: #7a7a85; font-size: 11px; font-weight: 500; margin-top: 10px;")
        panel_layout.addWidget(prob_header)
        
        self.emotion_prob_labels = []
        for i in range(3):
            prob_label = QLabel("--")
            prob_label.setStyleSheet("color: #d7d7e0; font-size: 10px; font-family: monospace;")
            self.emotion_prob_labels.append(prob_label)
            panel_layout.addWidget(prob_label)
        
        # Expression indicators
        expr_header = QLabel("expression")
        expr_header.setStyleSheet("color: #7a7a85; font-size: 11px; font-weight: 500; margin-top: 10px;")
        panel_layout.addWidget(expr_header)
        
        self.expression_bars = {}
        for name in ["mouth", "eye_l", "eye_r", "yaw", "pitch"]:
            bar = ExpressionBar(name)
            self.expression_bars[name] = bar
            panel_layout.addWidget(bar)
        
        panel_layout.addStretch()
        
        # VTube Studio Status
        self.vts_status_container = QWidget()
        vts_layout = QHBoxLayout(self.vts_status_container)
        vts_layout.setContentsMargins(0, 0, 0, 0)
        vts_layout.setSpacing(8)
        
        self.vts_status_dot = QLabel()
        self.vts_status_dot.setFixedSize(8, 8)
        self.vts_status_dot.setStyleSheet("background: #3a3b3e; border-radius: 4px;")
        
        self.vts_status_label = QLabel("VTube Studio")
        self.vts_status_label.setStyleSheet("color: #7a7a85; font-size: 11px;")
        
        vts_layout.addWidget(self.vts_status_dot)
        vts_layout.addWidget(self.vts_status_label)
        vts_layout.addStretch()
        
        panel_layout.addWidget(self.vts_status_container)
        
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
        
        # FPS tracking
        self._fps_history = []
        self._last_fps_update = 0
        
        self.calibration_active = False
    
    def _on_calibration_clicked(self):
        """Handle calibration button click - start countdown."""
        self.calibration_button.setEnabled(False)
        self.camera_button.setEnabled(False)
        self._countdown_value = 3
        self.calibration_widget.setCountdown(self._countdown_value)
        self._countdown_timer.start(1000)  # Update every second
    
    def _on_camera_toggled(self, checked):
        """Handle camera toggle button click."""
        if self.calibration_active:
            # Prevent toggling during calibration
            self.camera_button.setChecked(not checked) # Revert state
            return

        if checked:
            self.camera_button.setText("Camera: ON")
            self.camera_button.setStyleSheet("""
                QPushButton {
                    background: #2a2b2e;
                    color: #d7d7e0;
                    border: 1px solid #3a3b3e;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #3a3b3e;
                    border-color: #4a4b4e;
                }
                QPushButton:checked {
                    background: #2a2b2e;
                    color: #7fffe1;
                    border-color: #7fffe1;
                }
            """)
        else:
            self.camera_button.setText("Camera: OFF")
            self.camera_button.setStyleSheet("""
                QPushButton {
                    background: #1a1b1e;
                    color: #ff6b6b;
                    border: 1px solid #ff6b6b;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #2a2b2e;
                    border-color: #ff8888;
                }
            """)
        self.camera_toggled.emit(checked)
        self.video_display.setCameraActive(checked)
        
    def set_vts_status(self, connected):
        """Update VTube Studio connection status indicator."""
        if connected:
            self.vts_status_dot.setStyleSheet("background: #7fffe1; border-radius: 4px;")  # Green
            self.vts_status_label.setText("VTube Studio: Connected")
            self.vts_status_label.setStyleSheet("color: #d7d7e0; font-size: 11px;")
        else:
            self.vts_status_dot.setStyleSheet("background: #3a3b3e; border-radius: 4px;")  # Grey
            self.vts_status_label.setText("VTube Studio: Disconnected")
            self.vts_status_label.setStyleSheet("color: #7a7a85; font-size: 11px;")
    
    def _countdown_tick(self):
        """Handle countdown timer tick."""
        self._countdown_value -= 1
        if self._countdown_value > 0:
            self.calibration_widget.setCountdown(self._countdown_value)
        else:
            self._countdown_timer.stop()
            self.calibration_widget.setCountdown(0)
            self.calibration_requested.emit()
            # Re-enable button is handled by update_calibration completion
            # QTimer.singleShot(3000, lambda: self.calibration_button.setEnabled(True))
            # QTimer.singleShot(3000, lambda: self.calibration_button.setEnabled(True))
    
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
        
        # Update top 3 emotion probabilities
        if emotion_probs:
            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (emo, prob) in enumerate(sorted_emotions):
                emo_color = self.emotion_colors.get(emo, self.emotion_colors['neutral'])
                self.emotion_prob_labels[i].setText(f"{emo:8s} {int(prob * 100):2d}%")
                self.emotion_prob_labels[i].setStyleSheet(
                    f"color: rgb({emo_color.red()}, {emo_color.green()}, {emo_color.blue()}); "
                    f"font-size: 10px; font-family: monospace;"
                )
        else:
            # No emotions yet (before calibration or no DeepFace)
            for i in range(3):
                self.emotion_prob_labels[i].setText("--")
                self.emotion_prob_labels[i].setStyleSheet(
                    "color: #7a7a85; font-size: 10px; font-family: monospace;"
                )
    
    def update_expression(self, mp_features):
        """Update expression indicator bars."""
        if not mp_features:
            return
        
        # Map MediaPipe features to bars with proper normalization
        # MAR: typically 0.2-0.6, normalize to 0-1 range
        mar_raw = mp_features.get('mar', 0.0)
        mar_normalized = max(0.0, min(1.0, (mar_raw - 0.2) / 0.4)) if mar_raw > 0 else 0.0
        
        # EAR: typically 0.08-0.35, normalize to 0-1 range
        ear_left_raw = mp_features.get('ear_left', 0.0)
        ear_right_raw = mp_features.get('ear_right', 0.0)
        ear_left_normalized = max(0.0, min(1.0, (ear_left_raw - 0.08) / 0.27)) if ear_left_raw > 0 else 0.0
        ear_right_normalized = max(0.0, min(1.0, (ear_right_raw - 0.08) / 0.27)) if ear_right_raw > 0 else 0.0
        
        # Yaw/Pitch: already in -0.5 to 0.5 range, normalize to 0-1
        yaw_raw = mp_features.get('yaw', 0.0)
        pitch_raw = mp_features.get('pitch', 0.0)
        yaw_normalized = max(0.0, min(1.0, yaw_raw + 0.5))
        pitch_normalized = max(0.0, min(1.0, pitch_raw + 0.5))
        
        mapping = {
            'mouth': mar_normalized,
            'eye_l': ear_left_normalized,
            'eye_r': ear_right_normalized,
            'yaw': yaw_normalized,
            'pitch': pitch_normalized
        }
        
        for name, value in mapping.items():
            if name in self.expression_bars:
                self.expression_bars[name].setValue(value)
    
    def update_fps(self, fps):
        """Update FPS display."""
        color = "#FFFFFF"
        
        self.title_bar.fps_label.setText(f"FPS: {fps:.1f}")
        self.title_bar.fps_label.setStyleSheet(f"color: {color}; font-size: 11px; font-family: monospace;")
    
    def update_calibration(self, elapsed, duration):
        """Update calibration progress."""
        if elapsed < duration:
            self.calibration_widget.show()
            self.calibration_widget.updateProgress(elapsed, duration)
        else:
            self.calibration_widget.setReady()
            self.calibration_button.setEnabled(True)
            self.camera_button.setEnabled(True)
            
    def set_camera_button_enabled(self, enabled):
        """Enable or disable the camera toggle button."""
        self.camera_button.setEnabled(enabled)
        self.calibration_active = not enabled
        

    

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
        if self.window and emotion_label:
            self.window.update_emotion(emotion_label, emotion_probs)
    
    def update_expression(self, mp_features):
        """Update expression indicators."""
        if self.window and mp_features:
            self.window.update_expression(mp_features)
    
    def update_fps(self, fps):
        """Update FPS display."""
        if self.window:
            self.window.update_fps(fps)
    
    def update_calibration(self, elapsed, duration):
        """Update calibration progress."""
        if self.window:
            self.window.update_calibration(elapsed, duration)
            
    def set_camera_button_enabled(self, enabled):
        """Enable or disable camera button."""
        if self.window:
            self.window.set_camera_button_enabled(enabled)
    
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

