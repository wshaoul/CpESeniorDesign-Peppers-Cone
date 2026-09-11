"""Standalone Qt application for Pepper's Cone on macOS."""

from __future__ import annotations

import os
import sys

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from processing import CANVAS_SIZE, ConeProcessor, open_mac_camera


STYLE = """
QMainWindow, QWidget { background: #f5f7fb; color: #172033; font-size: 14px; }
QFrame#card { background: white; border: 1px solid #dce3ef; border-radius: 12px; }
QPushButton { background: #e8edf5; border: 0; border-radius: 8px; padding: 10px 16px; }
QPushButton:hover { background: #dce5f2; }
QPushButton#primary { background: #ef4444; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #dc2626; }
QPushButton:disabled { color: #94a3b8; background: #e9edf3; }
QLineEdit, QComboBox, QSpinBox { background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px; }
QLabel#title { font-size: 24px; font-weight: 700; }
QLabel#section { font-size: 17px; font-weight: 650; }
QLabel#status { color: #475569; }
QTabBar::tab { padding: 10px 22px; }
"""


def frame_pixmap(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(
        rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
    ).copy()
    return QPixmap.fromImage(image)


class VideoLabel(QLabel):
    def __init__(self, text="Camera stopped"):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background:#0f172a;color:#94a3b8;border-radius:10px;")
        self._source = None

    def set_frame(self, frame):
        self._source = frame_pixmap(frame)
        self._draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._draw()

    def _draw(self):
        if self._source is not None:
            self.setPixmap(
                self._source.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class FullscreenDisplay(QWidget):
    def __init__(self, title="Pepper's Cone Display"):
        super().__init__()
        self.setWindowTitle(title)
        self.setStyleSheet("background:black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.video = VideoLabel("")
        self.video.setStyleSheet("background:black;")
        layout.addWidget(self.video)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
            return
        super().keyPressEvent(event)


class LivePage(QWidget):
    def __init__(self):
        super().__init__()
        self.capture = None
        self.processor = ConeProcessor()
        self.fullscreen = FullscreenDisplay()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        root = QVBoxLayout(self)
        title = QLabel("Live Display")
        title.setObjectName("title")
        root.addWidget(title)
        root.addWidget(QLabel("Mac camera → Pepper's Cone output"))

        controls = QHBoxLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 10)
        self.resolution = QComboBox()
        self.resolution.addItems(["1280x720", "1920x1080", "640x480"])
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(30)
        self.background = QCheckBox("Remove background")
        self.background.setChecked(self.processor.segmentation_available)
        self.background.setEnabled(self.processor.segmentation_available)
        controls.addWidget(QLabel("Camera index"))
        controls.addWidget(self.camera_index)
        controls.addWidget(QLabel("Resolution"))
        controls.addWidget(self.resolution)
        controls.addWidget(QLabel("FPS"))
        controls.addWidget(self.fps)
        controls.addWidget(self.background)
        controls.addStretch()
        root.addLayout(controls)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Start Preview")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("Stop")
        self.output_button = QPushButton("Open Cone Screen")
        self.stop_button.setEnabled(False)
        self.output_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.output_button.clicked.connect(self.open_output)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.output_button)
        actions.addStretch()
        root.addLayout(actions)

        self.preview = VideoLabel()
        root.addWidget(self.preview, 1)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        root.addWidget(self.status)

    def start(self):
        self.stop()
        capture = open_mac_camera(self.camera_index.value())
        if capture is None:
            QMessageBox.critical(
                self,
                "Camera",
                "Could not open that camera. Check macOS Camera permission or try another index.",
            )
            return
        width, height = map(int, self.resolution.currentText().split("x"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, self.fps.value())
        self.capture = capture
        self.timer.start(max(1, round(1000 / self.fps.value())))
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.output_button.setEnabled(True)
        self.status.setText(f"Live on camera {self.camera_index.value()}")

    def next_frame(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.status.setText("Camera stopped returning frames")
            return
        self.preview.set_frame(frame)
        if self.fullscreen.isVisible():
            warped = self.processor.process(frame, self.background.isChecked())
            self.fullscreen.video.set_frame(warped)

    def open_output(self):
        if self.capture is None:
            return
        self.fullscreen.showFullScreen()

    def stop(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.fullscreen.close()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.output_button.setEnabled(False)
        self.status.setText("Ready")

    def shutdown(self):
        self.stop()
        self.processor.close()


class ProcessedVideoWindow(FullscreenDisplay):
    def __init__(self, path, remove_background=True):
        super().__init__("Pepper's Cone Video")
        self.path = path
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open video:\n{path}")
        self.processor = ConeProcessor()
        self.remove_background = remove_background
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(max(1, round(1000 / min(60.0, fps))))

    def next_frame(self):
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if ok:
            self.video.set_frame(
                self.processor.process(frame, self.remove_background)
            )

    def closeEvent(self, event):
        self.timer.stop()
        self.capture.release()
        self.processor.close()
        super().closeEvent(event)


class UploadPage(QWidget):
    def __init__(self):
        super().__init__()
        self.player = None
        layout = QVBoxLayout(self)
        title = QLabel("Upload Video")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(QLabel("Choose a video and play it through the cone-warp pipeline."))
        row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("Video file")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        self.background = QCheckBox("Remove background")
        self.background.setChecked(True)
        layout.addWidget(self.background)
        play = QPushButton("Open Cone Screen")
        play.setObjectName("primary")
        play.clicked.connect(self.play)
        layout.addWidget(play, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose video", "", "Videos (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)"
        )
        if path:
            self.path.setText(path)

    def play(self):
        path = self.path.text().strip()
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Upload", "Choose an existing video first.")
            return
        try:
            self.player = ProcessedVideoWindow(path, self.background.isChecked())
            if not self.player.processor.segmentation_available:
                self.player.remove_background = False
            self.player.showFullScreen()
        except Exception as error:
            QMessageBox.critical(self, "Upload", str(error))

    def shutdown(self):
        if self.player is not None:
            self.player.close()


class RecordPage(QWidget):
    def __init__(self):
        super().__init__()
        self.capture = None
        self.writer = None
        self.recording = False
        self.player = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        layout = QVBoxLayout(self)
        title = QLabel("Record")
        title.setObjectName("title")
        layout.addWidget(title)
        settings = QFormLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 10)
        self.resolution = QComboBox()
        self.resolution.addItems(["1280x720", "1920x1080", "640x480"])
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(30)
        self.output = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_output)
        output_row.addWidget(browse)
        settings.addRow("Camera index", self.camera_index)
        settings.addRow("Resolution", self.resolution)
        settings.addRow("FPS", self.fps)
        settings.addRow("Output", output_row)
        layout.addLayout(settings)

        actions = QHBoxLayout()
        self.preview_button = QPushButton("Start Preview")
        self.record_button = QPushButton("Start Recording")
        self.record_button.setObjectName("primary")
        self.stop_button = QPushButton("Stop")
        self.play_button = QPushButton("Play Cone Recording")
        self.preview_button.clicked.connect(self.start_preview)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop)
        self.play_button.clicked.connect(self.play_recording)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.record_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.play_button)
        actions.addStretch()
        layout.addLayout(actions)
        self.preview = VideoLabel()
        layout.addWidget(self.preview, 1)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save recording", "recording.mp4", "MP4 video (*.mp4)"
        )
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output.setText(path)

    def start_preview(self):
        if self.capture is not None:
            return True
        capture = open_mac_camera(self.camera_index.value())
        if capture is None:
            QMessageBox.critical(self, "Camera", "Could not open that camera index.")
            return False
        width, height = map(int, self.resolution.currentText().split("x"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, self.fps.value())
        self.capture = capture
        self.timer.start(max(1, round(1000 / self.fps.value())))
        self.status.setText("Previewing")
        return True

    def start_recording(self):
        path = self.output.text().strip()
        if not path:
            QMessageBox.warning(self, "Record", "Choose an output file first.")
            return
        if not self.start_preview():
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps.value(), (width, height)
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            QMessageBox.critical(self, "Record", "Could not create the MP4 file.")
            return
        self.recording = True
        self.status.setText(f"Recording {os.path.basename(path)}")

    def next_frame(self):
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            return
        self.preview.set_frame(frame)
        if self.recording and self.writer is not None:
            self.writer.write(frame)

    def stop(self):
        self.timer.stop()
        self.recording = False
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.status.setText("Ready")

    def play_recording(self):
        path = self.output.text().strip()
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Record", "Record a video first.")
            return
        self.stop()
        try:
            self.player = ProcessedVideoWindow(path)
            self.player.showFullScreen()
        except Exception as error:
            QMessageBox.critical(self, "Record", str(error))

    def shutdown(self):
        self.stop()
        if self.player is not None:
            self.player.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pepper's Cone Studio — macOS")
        self.resize(1220, 780)
        self.tabs = QTabWidget()
        self.live = LivePage()
        self.record = RecordPage()
        self.upload = UploadPage()
        self.tabs.addTab(self.live, "Live")
        self.tabs.addTab(self.record, "Record")
        self.tabs.addTab(self.upload, "Upload")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event):
        self.live.shutdown()
        self.record.shutdown()
        self.upload.shutdown()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
