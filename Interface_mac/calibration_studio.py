"""Guided optical calibration for the Mac 3D model output."""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QStandardPaths, QTimer, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
    QHBoxLayout, QLabel, QPushButton, QProgressBar, QSpinBox, QVBoxLayout,
    QWidget, QDoubleSpinBox, QTabWidget)

from cone_calibration import CalibrationSequence, build_profile, decode_sequence, verification_grid
from calibration_camera import CalibrationCamera


def default_profile_path():
    root = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericDataLocation)) / "PeppersCone"
    return root / "front_audience_calibration.npz"


class CameraRoiPreview(QLabel):
    """Camera preview with a draggable rectangle around the cone reflection."""

    def __init__(self):
        super().__init__("Start the camera, then drag a box around only the cone reflection.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640,360)
        self.setStyleSheet("background:#080c14;color:#9aa7bc;border-radius:11px;")
        self.frame = None
        self.roi = (.2,.05,.8,.78)
        self.drag_start = None

    def roi_pixels(self, shape=None):
        height,width = (shape or self.frame.shape)[:2]
        x1,y1,x2,y2 = self.roi
        return (max(0,min(width-1,round(x1*width))),
                max(0,min(height-1,round(y1*height))),
                max(1,min(width,round(x2*width))),
                max(1,min(height,round(y2*height))))

    def set_frame(self, frame):
        self.frame = frame
        shown = frame.copy()
        x1,y1,x2,y2 = self.roi_pixels(frame.shape)
        cv2.rectangle(shown,(x1,y1),(x2,y2),(40,220,255),max(2,round(frame.shape[1]/500)))
        cv2.putText(shown,"CONE REFLECTION ONLY",(x1+8,max(24,y1+28)),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(40,220,255),2,cv2.LINE_AA)
        rgb = cv2.cvtColor(shown,cv2.COLOR_BGR2RGB)
        height,width = rgb.shape[:2]
        pixmap = QPixmap.fromImage(QImage(rgb.data,width,height,rgb.strides[0],
                                         QImage.Format.Format_RGB888).copy())
        self.setPixmap(pixmap.scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))

    def _normalized_point(self, point):
        if self.frame is None:
            return None
        height,width = self.frame.shape[:2]
        scale = min(self.width()/width,self.height()/height)
        draw_width,draw_height = width*scale,height*scale
        left,top = (self.width()-draw_width)/2,(self.height()-draw_height)/2
        x = np.clip((point.x()-left)/max(1,draw_width),0,1)
        y = np.clip((point.y()-top)/max(1,draw_height),0,1)
        return float(x),float(y)

    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton:
            self.drag_start = self._normalized_point(event.position())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self,event):
        end = self._normalized_point(event.position())
        if self.drag_start is not None and end is not None:
            x1,x2 = sorted((self.drag_start[0],end[0]))
            y1,y2 = sorted((self.drag_start[1],end[1]))
            if x2-x1>.04 and y2-y1>.04:
                self.roi = (x1,y1,x2,y2)
            self.drag_start = None
            if self.frame is not None:
                self.set_frame(self.frame)
        super().mouseReleaseEvent(event)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.frame is not None:
            self.set_frame(self.frame)


class PatternWindow(QWidget):
    dismissed = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cone calibration pattern")
        self.setStyleSheet("background:black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setScaledContents(True)
        layout.addWidget(self.image)

    def set_frame(self,frame):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        height,width = rgb.shape[:2]
        pixmap = QPixmap.fromImage(QImage(rgb.data,width,height,rgb.strides[0],
                                         QImage.Format.Format_RGB888).copy())
        self.image.setPixmap(pixmap)

    def open_on(self,screen):
        self.hide()
        self.winId()
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(screen)
        self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.raise_()

    def keyPressEvent(self,event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Escape,Qt.Key.Key_Q):
            self.hide()
            self.dismissed.emit()
        else:
            super().keyPressEvent(event)

    def closeEvent(self,event):
        self.dismissed.emit()
        super().closeEvent(event)


class CalibrationDialog(QDialog):
    profile_ready = Signal(object)

    def __init__(self, screen=None, output_size=(1920,1080), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate cone for the front audience")
        self.resize(940,850)
        self.capture = None
        self.latest_frame = None
        self.pattern_window = PatternWindow()
        self.sequence = CalibrationSequence(*output_size)
        self.pattern_index = -1
        self.captured_frames = []
        self.profile = None
        self.last_frame_time = 0
        self.pattern_started = 0
        self.pattern_samples = []
        self.capture_roi = None
        self.capture_shape = None
        self.capture_screen = None
        self.pending_cameras = []
        self.pattern_window.dismissed.connect(self.cancel_output)

        layout = QVBoxLayout(self)
        title = QLabel("Front audience calibration")
        title.setObjectName("title")
        layout.addWidget(title)
        explanation = QLabel("1. Mount the iPhone at the audience's eye height and distance, facing the cone. Connect it to this Mac using Continuity Camera (USB recommended). Turn off Center Stage and camera effects.\n2. Start the camera and confirm you see the iPhone view. Drag a box around the reflection only; exclude the visible TV below it.")
        explanation.setWordWrap(True)
        explanation.setObjectName("subtitle")
        layout.addWidget(explanation)

        controls = QFormLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0,12)
        controls.addRow("Camera number",self.camera_index)
        self.camera_index.setToolTip("Try 0, then 1 or 2, stopping the camera between attempts. Numbers depend on connected cameras. The preview must show your iPhone's view.")
        self.screen = QComboBox()
        screens = QApplication.screens()
        for candidate in screens:
            label = candidate.name() or "Display"
            if candidate==QApplication.primaryScreen():
                label += " (primary)"
            self.screen.addItem(label,candidate)
        selected = screen or (screens[1] if len(screens)>1 else QApplication.primaryScreen())
        self.screen.setCurrentIndex(max(0,next((i for i,item in enumerate(screens) if item==selected),0)))
        controls.addRow("Pattern display",self.screen)
        refresh = QPushButton("Refresh displays")
        refresh.clicked.connect(self.refresh_screens)
        self.refresh_button = refresh
        controls.addRow(refresh)
        self.settle_seconds = QDoubleSpinBox()
        self.settle_seconds.setRange(.5,5)
        self.settle_seconds.setSingleStep(.5)
        self.settle_seconds.setValue(1.5)
        controls.addRow("Seconds per pattern",self.settle_seconds)
        self.pattern_size = QLabel(f"{output_size[0]} × {output_size[1]} • {len(self.sequence)} patterns")
        controls.addRow("Calibration sequence",self.pattern_size)
        layout.addLayout(controls)

        self.preview = CameraRoiPreview()
        self.preview.setMinimumSize(580,240)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.preview,"Camera / reflection")
        self.map_preview = QLabel("Capture patterns to see the measured distortion map.")
        self.map_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(self.map_preview,"Distortion map")
        layout.addWidget(self.tabs,1)
        self.map_hint = QLabel("Map: red = horizontal source position, green = vertical; black = unsupported. Coverage is not an accuracy score.")
        self.map_hint.setWordWrap(True)
        layout.addWidget(self.map_hint)
        self.progress = QProgressBar()
        self.progress.setRange(0,len(self.sequence))
        layout.addWidget(self.progress)
        self.status = QLabel("Start the camera to set the cone area.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        verification = QHBoxLayout()
        self.plain_button = QPushButton("Grid without correction")
        self.corrected_button = QPushButton("Grid with correction")
        self.save_button = QPushButton("Save and use this map")
        self.plain_button.clicked.connect(lambda: self.show_grid(False))
        self.corrected_button.clicked.connect(lambda: self.show_grid(True))
        self.save_button.clicked.connect(self.save_profile)
        for button in (self.plain_button,self.corrected_button,self.save_button):
            button.setEnabled(False)
            verification.addWidget(button)
        layout.addLayout(verification)

        buttons = QHBoxLayout()
        self.camera_button = QPushButton("Start camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.capture_button = QPushButton(f"Capture {len(self.sequence)} patterns")
        self.capture_button.setObjectName("primary")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.start_calibration)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        buttons.addWidget(self.camera_button)
        buttons.addWidget(self.capture_button)
        self.cancel_button = QPushButton("Cancel / hide TV")
        self.cancel_button.clicked.connect(self.cancel_output)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch()
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera)

    def toggle_camera(self):
        if self.capture is not None:
            self.stop_camera()
            return
        self.pending_cameras = [camera for camera in self.pending_cameras if camera.thread.is_alive()]
        if self.pending_cameras:
            self.status.setText("Waiting for the previous camera to close. Check any macOS camera permission prompt, then try again.")
            return
        capture = CalibrationCamera(self.camera_index.value())
        capture.thread.start()
        self.capture = capture
        self.profile = None
        self.map_preview.setText("Capture patterns to see the measured distortion map.")
        for button in (self.plain_button,self.corrected_button,self.save_button):
            button.setEnabled(False)
        self.camera_button.setText("Stop camera")
        self.camera_index.setEnabled(False)
        self.latest_frame = None
        self.last_frame_time = 0
        self.capture_button.setEnabled(False)
        self.camera_timer.start(40)
        self.status.setText("Opening camera… If unavailable, check Camera permission and try another number. Confirm this is the iPhone view before capturing.")

    def update_camera(self):
        if self.capture is None:
            return
        if self.capture.error:
            message = self.capture.error
            self.stop_camera()
            self.status.setText(message)
            return
        sample = self.capture.snapshot()
        now = time.monotonic()
        if self.pattern_index>=0 and self.capture_screen not in QApplication.screens():
            self.abort_calibration("The pattern display disconnected. Reconnect the TV, refresh displays and capture again.")
        if self.pattern_index>=0 and now-self.pattern_started>self.settle_seconds.value()+8:
            self.abort_calibration("No fresh camera frames arrived for this pattern. Reconnect the iPhone and try again.")
        if sample is None or sample[0]<=self.last_frame_time:
            return
        timestamp,frame = sample
        if self.last_frame_time==0:
            self.status.setText(f"Camera preview ready ({frame.shape[1]} × {frame.shape[0]}). Confirm it is the iPhone view, then drag a box around the cone reflection.")
        self.last_frame_time = timestamp
        self.latest_frame = frame
        self.preview.set_frame(frame)
        self.capture_button.setEnabled(self.pattern_index<0)
        if self.pattern_index>=0:
            if frame.shape != self.capture_shape:
                self.abort_calibration("Camera resolution changed. Keep the lens and orientation fixed, then capture again.")
                return
            if timestamp>=self.pattern_started+self.settle_seconds.value():
                x1,y1,x2,y2 = self.capture_roi
                self.pattern_samples.append(frame[y1:y2,x1:x2].copy())
                if len(self.pattern_samples)==3:
                    self.capture_pattern()

    def start_calibration(self):
        if self.capture is None or self.latest_frame is None:
            self.status.setText("Start the camera and wait for a preview first.")
            return
        if len(QApplication.screens())<2:
            self.status.setText("Connect the TV as a separate, extended display, then press Refresh displays. Calibration needs the TV while the camera preview stays on the Mac.")
            return
        if time.monotonic()-self.last_frame_time>1:
            self.status.setText("Wait for a fresh iPhone preview before starting.")
            return
        x1,y1,x2,y2 = self.preview.roi_pixels(self.latest_frame.shape)
        if x2-x1<40 or y2-y1<40:
            self.status.setText("Draw a larger box around the cone reflection.")
            return
        self.capture_button.setEnabled(False)
        self.camera_button.setEnabled(False)
        self.screen.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.settle_seconds.setEnabled(False)
        self.preview.setEnabled(False)
        self.capture_roi = (x1,y1,x2,y2)
        self.capture_shape = self.latest_frame.shape
        self.capture_screen = self.screen.currentData()
        self.profile = None
        self.map_preview.setText("Capturing a new map…")
        for button in (self.plain_button,self.corrected_button,self.save_button):
            button.setEnabled(False)
        self.pattern_index = 0
        self.captured_frames = []
        self.progress.setValue(0)
        self.pattern_window.open_on(self.capture_screen or QApplication.primaryScreen())
        self.show_pattern()

    def show_pattern(self):
        if self.pattern_index<0:
            return
        if self.pattern_index>=len(self.sequence):
            self.finish_calibration()
            return
        frame = self.sequence.frame(self.pattern_index)
        self.pattern_window.set_frame(frame)
        self.pattern_started = time.monotonic()
        self.pattern_samples = []
        self.status.setText(f"Showing {self.pattern_index+1}/{len(self.sequence)}: {self.sequence.label(self.pattern_index)}. Keep the camera and cone still.")

    def capture_pattern(self):
        if self.capture is None or self.pattern_index<0:
            return
        if len(self.pattern_samples)!=3:
            return
        self.captured_frames.append(np.median(self.pattern_samples,axis=0).astype(np.uint8))
        self.pattern_index += 1
        self.progress.setValue(self.pattern_index)
        self.show_pattern()

    def finish_calibration(self):
        self.pattern_window.hide()
        self.status.setText("Building and smoothing the deformation map…")
        try:
            screen_x,screen_y,valid = decode_sequence(
                self.captured_frames,(self.sequence.width,self.sequence.height))
            profile = build_profile(screen_x,screen_y,valid,
                                    (self.sequence.width,self.sequence.height))
            if profile.coverage<.01:
                raise ValueError("Too little of the TV reflection was decoded. Tighten the selection around the cone and reduce room light.")
        except Exception as exc:
            self.abort_calibration(str(exc))
            return
        self.profile = profile
        self.abort_calibration(f"Map built from {profile.sample_count:,} samples ({profile.coverage:.1%} TV coverage). Compare both grids in the camera view: look for straight lines, square cells and a round circle. Save only after checking. This is not yet saved.")
        rgb = cv2.cvtColor(profile.visualization(),cv2.COLOR_BGR2RGB)
        height,width = rgb.shape[:2]
        pixmap = QPixmap.fromImage(QImage(rgb.data,width,height,rgb.strides[0],QImage.Format.Format_RGB888).copy())
        self.map_preview.setPixmap(pixmap.scaled(740,280,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        for button in (self.plain_button,self.corrected_button,self.save_button):
            button.setEnabled(True)

    def abort_calibration(self,message):
        self.pattern_window.hide()
        self.pattern_index = -1
        self.captured_frames = []
        self.pattern_samples = []
        self.camera_button.setEnabled(True)
        self.screen.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.settle_seconds.setEnabled(True)
        self.preview.setEnabled(True)
        self.capture_button.setEnabled(self.capture is not None)
        self.status.setText(message)

    def cancel_output(self):
        self.abort_calibration("TV output hidden. Any unfinished capture was cancelled; your saved profile was not changed.")

    def refresh_screens(self):
        self.screen.clear()
        for screen in QApplication.screens():
            self.screen.addItem(screen.name() or "Display",screen)
        self.screen.setCurrentIndex(1 if self.screen.count()>1 else 0)

    def show_grid(self,corrected):
        if self.profile is None:
            return
        width,height = self.profile.screen_size
        # Use the capture's aspect ratio so square cells stay square in the
        # calibrated camera plane, even when the selected reflection is tall.
        x1,y1,x2,y2 = self.capture_roi
        target = verification_grid(x2-x1,y2-y1)
        output = self.profile.apply(target) if corrected else cv2.resize(target,(width,height))
        self.pattern_window.set_frame(output)
        if self.capture_screen not in QApplication.screens():
            self.status.setText("The calibrated display disconnected. Reconnect the TV and capture a new map.")
            return
        self.pattern_window.open_on(self.capture_screen)
        self.tabs.setCurrentIndex(0)
        self.status.setText(("Corrected" if corrected else "Uncorrected")+" grid on TV. Keep the phone fixed and compare the reflection. Escape/Q or Cancel hides the TV output.")

    def save_profile(self):
        if self.profile is None:
            return
        try:
            self.profile.save(default_profile_path())
        except (OSError,ValueError) as exc:
            self.status.setText(f"Could not save calibration: {exc}")
            return
        self.profile_ready.emit(self.profile)
        self.pattern_window.hide()
        self.status.setText("Map saved and selected in 3D Model. Close this window, test the cube, then choose Tiger demo. Moving the cone, TV or viewpoint requires recalibration.")

    def stop_camera(self):
        self.cancel_output()
        self.camera_timer.stop()
        if self.capture is not None:
            self.capture.stop()
            self.pending_cameras.append(self.capture)
            self.capture = None
        self.latest_frame = None
        self.camera_button.setText("Start camera")
        self.camera_button.setEnabled(True)
        self.camera_index.setEnabled(True)
        self.capture_button.setEnabled(False)

    def closeEvent(self,event):
        self.stop_camera()
        super().closeEvent(event)

    def done(self,result):
        # QDialog.reject() (Close/Escape) does not reliably call closeEvent.
        self.stop_camera()
        super().done(result)
