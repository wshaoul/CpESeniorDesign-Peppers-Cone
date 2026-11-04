"""Guided optical calibration for the Mac 3D model output."""
from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout, QFileDialog,
    QHBoxLayout, QLabel, QPushButton, QProgressBar, QSpinBox, QVBoxLayout,
    QWidget, QDoubleSpinBox, QTabWidget, QScrollArea)

from cone_calibration import (CalibrationSequence, build_profile, decode_sequence,
                              default_profile_path, verification_grid)
from calibration_camera import CalibrationCamera
from calibration_recording import RecordingAnalysis, TAIL_SECONDS


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
        self.recording = None
        self.analysis = None
        self.playback = None
        self.pattern_window.dismissed.connect(self.cancel_output)

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)
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
        self.output_width = QSpinBox()
        self.output_height = QSpinBox()
        for spin, value in ((self.output_width, output_size[0]), (self.output_height, output_size[1])):
            spin.setRange(1, 7680)
            spin.setValue(value)
            spin.valueChanged.connect(self.update_sequence)
        size_row = QHBoxLayout()
        size_row.addWidget(self.output_width)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.output_height)
        size_row.setStretch(0, 1)
        size_row.setStretch(2, 1)
        controls.addRow("Pattern / recording size", size_row)
        self.pattern_size = QLabel(f"{output_size[0]} × {output_size[1]} • {len(self.sequence)} patterns")
        controls.addRow("Calibration sequence",self.pattern_size)
        layout.addLayout(controls)

        recording_actions = QHBoxLayout()
        self.lead_seconds = QSpinBox()
        self.lead_seconds.setRange(3, 30)
        self.lead_seconds.setValue(10)
        self.lead_seconds.setMaximumWidth(80)
        recording_actions.addWidget(QLabel("Black lead-in (s)"))
        recording_actions.addWidget(self.lead_seconds)
        self.play_button = QPushButton("Play for recording")
        self.load_button = QPushButton("Load recording…")
        self.build_button = QPushButton("Build map")
        self.build_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_for_recording)
        self.load_button.clicked.connect(self.load_recording)
        self.build_button.clicked.connect(self.build_from_recording)
        for button in (self.play_button, self.load_button, self.build_button):
            recording_actions.addWidget(button)
        layout.addLayout(recording_actions)

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
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self.poll_recording)
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_tick)

    def update_sequence(self):
        self.sequence = CalibrationSequence(self.output_width.value(), self.output_height.value())
        self.pattern_size.setText(f"{self.sequence.width} × {self.sequence.height} • {len(self.sequence)} patterns")
        self.capture_button.setText(f"Capture {len(self.sequence)} patterns")
        self.progress.setRange(0, len(self.sequence))

    def set_setup_enabled(self, enabled):
        for widget in (self.camera_button, self.screen, self.refresh_button, self.settle_seconds,
                       self.output_width, self.output_height, self.lead_seconds, self.play_button, self.load_button):
            widget.setEnabled(enabled)
        self.build_button.setEnabled(enabled and self.recording is not None)
        self.capture_button.setEnabled(enabled and self.capture is not None and self.latest_frame is not None)
        self.preview.setEnabled(enabled)

    def play_for_recording(self):
        if len(QApplication.screens()) < 2:
            self.status.setText("Connect the TV as an extended display before playing recording patterns.")
            return
        self.stop_camera()
        self.capture_screen = self.screen.currentData()
        self.playback = {"start": time.monotonic(), "screen": self.capture_screen,
                         "lead": self.lead_seconds.value(), "hold": self.settle_seconds.value(), "index": None}
        self.set_setup_enabled(False)
        self.pattern_window.open_on(self.capture_screen)
        self.playback_timer.start(20)
        self.playback_tick()

    def playback_tick(self):
        playback = self.playback
        if playback is None:
            return
        if playback["screen"] not in QApplication.screens():
            self.abort_calibration("The pattern display disconnected. Reconnect the TV and record again.")
            return
        elapsed = time.monotonic() - playback["start"]
        duration = len(self.sequence) * playback["hold"]
        if elapsed >= playback["lead"] + duration + TAIL_SECONDS:
            self.abort_calibration("Finished. Stop the phone recording, then load it with the same pattern size selected.")
            return
        index = int((elapsed - playback["lead"]) // playback["hold"]) if elapsed >= playback["lead"] else -1
        if index >= len(self.sequence):
            index = -1
        if index == playback["index"]:
            return
        playback["index"] = index
        if index < 0:
            self.pattern_window.set_frame(np.zeros((self.sequence.height, self.sequence.width, 3), np.uint8))
            self.status.setText("TV is black. Keep the phone recording through the patterns and final black tail.")
        else:
            self.pattern_window.set_frame(self.sequence.frame(index))
            self.progress.setValue(index + 1)
            self.status.setText(f"Playing {index + 1}/{len(self.sequence)}: {self.sequence.label(index)}. Keep the phone fixed.")

    def load_recording(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load calibration recording", "", "Videos (*.mp4 *.mov *.m4v *.avi)")
        if not path:
            return
        self.stop_camera()
        size = (self.sequence.width, self.sequence.height)
        self.analysis = RecordingAnalysis(path, len(self.sequence), size)
        self.analysis.thread.start()
        self.profile = None
        self.map_preview.setText("Build a new map after selecting the reflection.")
        for button in (self.plain_button, self.corrected_button, self.save_button):
            button.setEnabled(False)
        self.set_setup_enabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Reading recording…")
        self.recording_timer.start(100)

    def poll_recording(self):
        analysis = self.analysis
        if analysis is None:
            return
        if analysis.thread.is_alive():
            self.status.setText(analysis.message)
            self.progress.setValue(round(analysis.fraction * 100))
            return
        self.recording_timer.stop()
        self.analysis = None
        self.progress.setRange(0, len(self.sequence))
        self.progress.setValue(0)
        if analysis.error:
            self.set_setup_enabled(True)
            self.status.setText(analysis.error)
            return
        frames, period = analysis.result
        self.recording = {"frames": frames, "size": analysis.screen_size}
        self.profile = None
        for button in (self.plain_button, self.corrected_button, self.save_button):
            button.setEnabled(False)
        self.latest_frame = frames[0]
        self.preview.set_frame(frames[0])
        self.tabs.setCurrentIndex(0)
        self.set_setup_enabled(True)
        mismatch = " Check the source display size: the period differs from Seconds per pattern." if abs(period - self.settle_seconds.value()) > .025 * self.settle_seconds.value() else ""
        self.status.setText(f"Found {len(frames)} patterns at {period:.3f} s each. Drag a box around only the cone reflection, then Build map.{mismatch}")

    def build_from_recording(self):
        if self.recording is None:
            return
        if self.recording["size"] != (self.sequence.width, self.sequence.height):
            self.status.setText("Restore the recording's source pattern size before building the map.")
            return
        frames = self.recording["frames"]
        x1, y1, x2, y2 = self.preview.roi_pixels(frames[0].shape)
        if x2 - x1 < 40 or y2 - y1 < 40:
            self.status.setText("Draw a larger box around the cone reflection.")
            return
        self.capture_roi = (x1, y1, x2, y2)
        self.capture_screen = self.screen.currentData()
        self.captured_frames = [frame[y1:y2, x1:x2] for frame in frames]
        self.finish_calibration()

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
        self.set_setup_enabled(False)
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
        self.profile = None
        for button in (self.plain_button,self.corrected_button,self.save_button):
            button.setEnabled(False)
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
        self.playback_timer.stop()
        self.playback = None
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
        self.set_setup_enabled(True)
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
        self.status.setText("Map saved and selected in the page that opened calibration. Close this window and test the live view or 3D model. Moving the cone, TV or viewpoint requires recalibration.")

    def stop_camera(self):
        self.recording_timer.stop()
        if self.analysis is not None:
            self.analysis.cancel()
            self.analysis = None
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
