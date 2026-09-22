"""Standalone Qt application for Pepper's Cone on macOS."""

from __future__ import annotations

import os
import sys

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QKeyEvent, QPixmap
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
from live_studio import LivePage
from model_studio import ModelPage


STYLE = """
QWidget { background: transparent; color: #192336; font-size: 13px; }
QMainWindow, QWidget#shell { background: #f3f5f9; }
QDialog, QMessageBox { background: white; }
QLabel { border: none; }
QLabel#brandMark { background: #315ce8; color: white; border-radius: 12px; font-size: 16px; font-weight: 700; }
QLabel#brandName { font-size: 20px; font-weight: 700; color: #15213a; }
QLabel#eyebrow { font-size: 10px; font-weight: 700; color: #7b8799; }
QLabel#badge { background: #e6ecfc; color: #3155ac; border-radius: 10px; padding: 6px 12px; font-size: 11px; font-weight: 600; }
QFrame#card, QFrame#previewCard, QWidget#controlCard { background: white; border: 1px solid #e0e5ef; border-radius: 16px; }
QLabel#title { font-size: 28px; font-weight: 700; color: #17243e; }
QLabel#section { font-size: 15px; font-weight: 600; }
QLabel#subtitle, QLabel#status { color: #66748b; font-size: 12px; }
QLabel#tip { color: #697994; font-size: 11px; padding: 0; }
QPushButton { background: white; border: 1px solid #dce3ef; border-radius: 10px; padding: 11px 18px; font-weight: 600; }
QPushButton:hover { background: #f0f4fd; border-color: #b7c7eb; }
QPushButton:pressed { background: #e3ebfc; }
QPushButton:focus { border-color: #315ce8; }
QPushButton#primary { background: #315ce8; border-color: #315ce8; color: white; }
QPushButton#primary:hover { background: #244cce; border-color: #244cce; }
QPushButton#primary:pressed { background: #1c3dae; }
QPushButton#tvAction { background: #17243e; color: white; border-color: #17243e; }
QPushButton#tvAction:hover { background: #283d60; }
QPushButton#quiet { border: none; background: transparent; color: #63728b; text-align: left; padding: 8px 0; font-size: 12px; }
QPushButton#quiet:hover { color: #315ce8; }
QPushButton:disabled, QPushButton#primary:disabled, QPushButton#tvAction:disabled { background: #edf0f6; color: #a0aabd; border-color: #edf0f6; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #f8f9fc; border: 1px solid #e0e5ef; border-radius: 8px; padding: 9px 10px; selection-background-color: #315ce8; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #315ce8; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox::down-arrow { image: url("__CHEVRON_ICON__"); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: white; color: #192336; border: 1px solid #dce3ef; selection-background-color: #e8eefc; selection-color: #244cce; padding: 4px; }
QCheckBox { spacing: 9px; padding: 5px 0; font-size: 12px; }
QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px; border: 1px solid #cbd5e5; background: white; }
QCheckBox::indicator:checked { background: #315ce8; border-color: #315ce8; image: url("__CHECK_ICON__"); }
QCheckBox::indicator:hover { border-color: #315ce8; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 6px 0; }
QScrollBar::handle:vertical { background: #ccd5e4; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QTabWidget::pane { border: none; padding-top: 14px; }
QTabBar { background: transparent; }
QTabBar::tab { background: #e9edf5; color: #77849a; border: none; padding: 10px 26px; margin-right: 4px; border-radius: 9px; font-weight: 600; }
QTabBar::tab:selected { background: white; color: #315ce8; }
QTabBar::tab:hover:!selected { background: #e0e7f3; color: #405778; }
QToolTip { background: #17243e; color: white; border: none; padding: 6px; }
""".replace("__CHECK_ICON__", os.path.join(os.path.dirname(__file__), "assets", "check.svg")).replace("__CHEVRON_ICON__", os.path.join(os.path.dirname(__file__), "assets", "chevron.svg"))


def configure_appearance(app):
    app.setStyle("Fusion")
    app.setFont(QFont("Helvetica Neue", 11))
    app.setStyleSheet(STYLE)


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
        self.resize(1280, 850)
        self.setMinimumSize(1000, 720)
        shell = QWidget()
        shell.setObjectName("shell")
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(24,20,24,20)
        layout.setSpacing(20)
        header = QHBoxLayout()
        header.setSpacing(12)
        mark = QLabel("PC")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(44,44)
        header.addWidget(mark)
        brand = QVBoxLayout()
        brand.setSpacing(3)
        name = QLabel("Pepper's Cone")
        name.setObjectName("brandName")
        brand.addWidget(name)
        descriptor = QLabel("DISPLAY STUDIO")
        descriptor.setObjectName("eyebrow")
        brand.addWidget(descriptor)
        header.addLayout(brand)
        header.addStretch()
        badge = QLabel("MAC EDITION")
        badge.setObjectName("badge")
        header.addWidget(badge)
        layout.addLayout(header)
        self.tabs = QTabWidget()
        self.live = LivePage()
        self.record = RecordPage()
        self.upload = UploadPage()
        self.model = ModelPage()
        self.tabs.addTab(self.live, "Live")
        self.tabs.addTab(self.model, "3D Model")
        self.tabs.addTab(self.record, "Record")
        self.tabs.addTab(self.upload, "Upload")
        layout.addWidget(self.tabs,1)
        self.setCentralWidget(shell)

    def closeEvent(self, event):
        self.live.shutdown()
        self.model.shutdown()
        self.record.shutdown()
        self.upload.shutdown()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    configure_appearance(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
