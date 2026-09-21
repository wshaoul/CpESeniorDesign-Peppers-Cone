"""Mac live camera preview and fullscreen cone output."""
import threading
import time
from collections import deque

import cv2
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget)

from processing import open_mac_camera
from projection import ProjectionSettings, LiveProjector, test_card, alignment_pattern


class Preview(QLabel):
    def __init__(self, text="", full_resolution=False):
        super().__init__(text)
        self.setMinimumSize(240,180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#080c14;color:#8998b2;border:1px solid #121b2c;border-radius:11px;font-size:13px;")
        self.source = None
        self.full_resolution = full_resolution

    def set_frame(self, frame):
        # Preview only: the full-resolution output remains in the worker result.
        size = max(self.width(), self.height())
        h,w = frame.shape[:2]
        if not self.full_resolution and max(h,w)>size:
            frame = cv2.resize(frame, (max(1,round(w*size/max(h,w))), max(1,round(h*size/max(h,w)))))
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.source = QPixmap.fromImage(QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy())
        self.draw()

    def draw(self):
        if self.source is not None:
            ratio = self.devicePixelRatioF() if self.full_resolution else 1
            pixmap = self.source.scaled(round(self.width()*ratio),round(self.height()*ratio),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
            pixmap.setDevicePixelRatio(ratio)
            self.setPixmap(pixmap)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.draw()


class TVWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pepper's Cone — TV output")
        self.setStyleSheet("background:black;")
        self.setCursor(Qt.CursorShape.BlankCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.video = Preview(full_resolution=True)
        self.video.setStyleSheet("background:black;border:none;border-radius:0;")
        layout.addWidget(self.video)

    def open_on(self,screen):
        self.hide()
        self.winId()
        self.windowHandle().setScreen(screen)
        self.setGeometry(screen.geometry())
        self.showFullScreen()

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key.Key_Escape,Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)


class LiveWorker:
    """Own camera and MediaPipe off the GUI thread; keep only the newest result."""
    def __init__(self,index,resolution,fps,settings,mode):
        self.index,self.resolution,self.fps = index,resolution,fps
        self.settings,self.mode = settings,mode
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.latest = None
        self.error = ""
        self.segmentation_available = None
        self.thread = threading.Thread(target=self.run,daemon=True)
        self.camera_thread = None
        self.input_condition = threading.Condition()
        self.input_frame = None

    def read_camera(self,capture):
        sequence = 0
        try:
            while not self.done.is_set():
                ok,frame = capture.read()
                if not ok:
                    self.error = "Camera stopped returning frames. Stop and restart the preview."
                    return
                sequence += 1
                with self.input_condition:
                    self.input_frame = (sequence,frame)
                    self.input_condition.notify_all()
        except Exception as exc:
            self.error = str(exc)
        finally:
            capture.release()
            with self.input_condition:
                self.input_condition.notify_all()

    def update(self,settings,mode):
        with self.lock:
            self.settings,self.mode = settings,mode

    def run(self):
        capture = None
        projector = None
        try:
            projector = LiveProjector()
            self.segmentation_available = projector.segmenter is not None
            if self.mode == "Camera":
                capture = open_mac_camera(self.index)
                if capture is None:
                    raise RuntimeError("Camera unavailable. Check macOS Camera permission or another camera index.")
                w,h = self.resolution
                capture.set(cv2.CAP_PROP_FRAME_WIDTH,w)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
                capture.set(cv2.CAP_PROP_FPS,self.fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
                self.camera_thread = threading.Thread(target=self.read_camera,args=(capture,),daemon=True)
                self.camera_thread.start()
            card = test_card()
            sequence = 0
            timestamps = deque(maxlen=30)
            input_sequence = -1
            while not self.done.is_set():
                with self.lock:
                    settings,mode = self.settings,self.mode
                frame = card
                if capture is not None:
                    with self.input_condition:
                        self.input_condition.wait_for(lambda: self.done.is_set() or self.error or
                            (self.input_frame is not None and self.input_frame[0]!=input_sequence),timeout=1)
                        if self.done.is_set() or self.error:
                            break
                        if self.input_frame is None or self.input_frame[0]==input_sequence:
                            continue
                        input_sequence,frame = self.input_frame
                started = time.monotonic()
                output = alignment_pattern(settings) if mode == "Alignment rings" else projector.process(frame,settings,mode=="Camera")
                now = time.monotonic()
                timestamps.append(now)
                interval = now-timestamps[0]
                rate = (len(timestamps)-1)/interval if interval>=.5 else 0
                sequence += 1
                with self.lock:
                    self.latest = (sequence,frame,output,rate,(now-started)*1000)
                # Static patterns need no more than 15 Hz, avoiding unnecessary CPU.
                if capture is None:
                    self.done.wait(max(0,1/15-(time.monotonic()-started)))
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.done.set()
            if self.camera_thread is not None:
                self.camera_thread.join(timeout=2)
            elif capture is not None:
                capture.release()
            if projector is not None:
                projector.close()

    def stop(self):
        self.done.set()
        with self.input_condition:
            self.input_condition.notify_all()
        self.thread.join(timeout=2)
        return not self.thread.is_alive() and (self.camera_thread is None or not self.camera_thread.is_alive())


class LivePage(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.sequence = -1
        self.fullscreen = TVWindow()
        self.storage = QSettings("PeppersCone","MacLiveStudio")
        root = QVBoxLayout(self)
        root.setContentsMargins(0,4,0,0)
        root.setSpacing(14)
        title = QLabel("Live display")
        title.setObjectName("title")
        root.addWidget(title)
        subtitle = QLabel("Preview your camera and send the cone image to your TV.")
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)
        note = QLabel("Tip: connect your TV as a separate display in Mac Display settings, not a mirror of your desktop.")
        note.setWordWrap(True)
        note.setObjectName("tip")
        body = QHBoxLayout()
        body.setSpacing(18)
        panel = QWidget()
        panel.setObjectName("controlCard")
        form = QFormLayout(panel)
        form.setContentsMargins(20,20,20,20)
        form.setVerticalSpacing(12)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0,10)
        self.resolution = QComboBox()
        self.resolution.addItems(["1920x1080","1280x720","640x480"])
        self.mode = QComboBox()
        self.mode.addItems(["Camera","Orientation test card","Alignment rings"])
        self.screen = QComboBox()
        self.quality = QComboBox()
        self.quality.addItems(["Sharper picture","Smoother motion (recommended)","Highest detail (may be slower)"])
        self.quality.setCurrentIndex(1)
        self.views = QComboBox()
        for label,n in [("4 repeated views",4),("6 repeated views — experimental",6),("8 repeated views — experimental",8),("Single 200° arc — comparison",1)]:
            self.views.addItem(label,n)
        self.views.setCurrentIndex(max(0,self.views.findData(self.storage.value("views",4,type=int))))
        for combo in (self.mode,self.resolution,self.screen,self.quality,self.views):
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.background = QCheckBox("Remove camera background")
        self.background.setChecked(self.storage.value("background",True,type=bool))
        self.mirror = QCheckBox("Mirror for reflection")
        self.mirror.setChecked(self.storage.value("mirror",True,type=bool))
        self.invert = QCheckBox("Reverse head and feet radius")
        self.invert.setChecked(self.storage.value("invert",False,type=bool))
        self.portrait_crop = QCheckBox("Center portrait crop (larger subject)")
        self.portrait_crop.setChecked(self.storage.value("portrait_crop",True,type=bool))
        settings_title = QLabel("Set up your display")
        settings_title.setObjectName("section")
        form.addRow(settings_title)
        form.addRow("Show on",self.screen)
        form.addRow("Picture quality",self.quality)
        form.addRow(self.background)
        self.background.toggled.connect(self.update_settings)
        refresh = QPushButton("Find my TV")
        refresh.setObjectName("quiet")
        refresh.clicked.connect(self.refresh_screens)
        form.addRow(refresh)
        self.advanced_button = QPushButton("Advanced settings ▸")
        self.advanced_button.setObjectName("quiet")
        self.advanced_button.setCheckable(True)
        form.addRow(self.advanced_button)
        self.advanced_panel = QWidget()
        basic_form = form
        form = QFormLayout(self.advanced_panel)
        form.setContentsMargins(0,0,0,0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        explanation = QLabel("Cone fitting and experimental layouts. Repeated camera views do not create true 360° depth.")
        explanation.setWordWrap(True)
        form.addRow(explanation)
        self.spins = {}
        for label,widget in [("Camera or test pattern",self.mode),("Camera number",self.camera_index),("Camera resolution",self.resolution),("Repeated views",self.views)]:
            form.addRow(label,widget)
        for name,label,minimum,maximum,value,step in [
            ("diameter","Cone diameter %",10,100,94,1),
            ("inner","Inner radius %",0,85,12,1),
            ("rotation","Rotate layout °",0,359,270,5),
            ("center_x","Center across TV %",10,90,50,1),
            ("center_y","Center down TV %",10,90,50,1),
            ("zoom","Subject size %",20,150,85,5),
            ("gap","Sector gap °",0,8,2,.5),
            ("gain","Brightness gain",.5,2,1.15,.05)]:
            spin = QDoubleSpinBox()
            spin.setRange(minimum,maximum)
            spin.setValue(float(self.storage.value(name,value)))
            spin.setSingleStep(step)
            spin.setDecimals(2 if name=="gain" else 1)
            self.spins[name] = spin
            form.addRow(label,spin)
            spin.valueChanged.connect(self.update_settings)
        for widget in (self.mirror,self.invert,self.portrait_crop):
            form.addRow(widget)
            widget.toggled.connect(self.update_settings)
        save = QPushButton("Save cone fit")
        save.clicked.connect(self.save_fit)
        form.addRow(save)
        self.technical_status = QLabel("Camera and rendering details appear here while running.")
        self.technical_status.setWordWrap(True)
        form.addRow(self.technical_status)
        basic_form.addRow(self.advanced_panel)
        basic_form.addRow(note)
        self.advanced_panel.hide()
        self.advanced_button.toggled.connect(self.toggle_advanced)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(325)
        body.addWidget(scroll)
        previews = QVBoxLayout()
        previews.setSpacing(14)
        self.preview = Preview("Press Start camera to begin")
        self.output_preview = Preview("Your cone image will appear here")
        for title,caption,preview in [("Your camera","LIVE SOURCE",self.preview),("Cone preview","TV OUTPUT",self.output_preview)]:
            card = QFrame()
            card.setObjectName("previewCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14,12,14,14)
            card_layout.setSpacing(10)
            heading = QHBoxLayout()
            label = QLabel(title)
            label.setObjectName("section")
            heading.addWidget(label)
            heading.addStretch()
            tag = QLabel(caption)
            tag.setObjectName("eyebrow")
            heading.addWidget(tag)
            card_layout.addLayout(heading)
            card_layout.addWidget(preview,1)
            previews.addWidget(card,1)
        body.addLayout(previews,1)
        root.addLayout(body,1)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.output_button = QPushButton("Show on TV")
        self.output_button.setObjectName("tvAction")
        self.output_button.setEnabled(False)
        for button,callback in [(self.start_button,self.start),(self.output_button,self.open_output),(self.stop_button,self.stop)]:
            button.clicked.connect(callback)
            button.setMinimumHeight(44)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            actions.addWidget(button)
        actions.addStretch()
        root.addLayout(actions)
        self.status = QLabel("Ready when you are. Press Start camera.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(33)
        self.refresh_screens()
        QApplication.instance().screenAdded.connect(self.refresh_screens)
        QApplication.instance().screenRemoved.connect(self.refresh_screens)
        for combo in (self.views,self.quality,self.screen):
            combo.currentIndexChanged.connect(self.update_settings)
        self.mode.currentIndexChanged.connect(self.source_changed)

    def toggle_advanced(self,visible):
        self.advanced_panel.setVisible(visible)
        self.advanced_button.setText("Advanced settings ▾" if visible else "Advanced settings ▸")

    def refresh_screens(self,*args):
        selected = self.screen.currentData()
        self.screen.clear()
        screens = QApplication.screens()
        for screen in screens:
            geometry = screen.geometry()
            name = "This Mac" if screen==QApplication.primaryScreen() else screen.name()
            self.screen.addItem(name,screen)
            self.screen.setItemData(self.screen.count()-1,f"{screen.name()} • {geometry.width()}×{geometry.height()} logical pixels",Qt.ItemDataRole.ToolTipRole)
        index = next((i for i,s in enumerate(screens) if s==selected),1 if len(screens)>1 else 0)
        self.screen.setCurrentIndex(index)

    def settings(self):
        dimensions = [(1920,1080),(1280,720)]
        if self.quality.currentIndex()<2:
            w,h = dimensions[self.quality.currentIndex()]
        else:
            screen = self.screen.currentData() or QApplication.primaryScreen()
            geometry = screen.geometry()
            w,h = round(geometry.width()*screen.devicePixelRatio()),round(geometry.height()*screen.devicePixelRatio())
            scale = min(1,3840/w,2160/h)
            w,h = max(320,round(w*scale)),max(240,round(h*scale))
        values = {name:spin.value() for name,spin in self.spins.items()}
        for name in ("diameter","inner","center_x","center_y","zoom"):
            values[name] /= 100
        return ProjectionSettings(width=w,height=h,views=self.views.currentData(),
            mirror=self.mirror.isChecked(),invert=self.invert.isChecked(),portrait_crop=self.portrait_crop.isChecked(),remove_background=self.background.isChecked(),**values)

    def update_settings(self,*args):
        if self.worker is not None:
            self.worker.update(self.settings(),self.mode.currentText())

    def source_changed(self,*args):
        if self.worker is not None:
            self.start()

    def save_fit(self):
        for name,spin in self.spins.items():
            self.storage.setValue(name,spin.value())
        self.storage.setValue("views",self.views.currentData())
        for name in ("background","mirror","invert","portrait_crop"):
            self.storage.setValue(name,getattr(self,name).isChecked())
        self.status.setText("Cone fit saved on this Mac.")

    def start(self):
        if not self.stop():
            return
        resolution = tuple(map(int,self.resolution.currentText().split("x")))
        self.worker = LiveWorker(self.camera_index.value(),resolution,30,self.settings(),self.mode.currentText())
        self.sequence = -1
        self.worker.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.output_button.setEnabled(False)
        self.status.setText("Starting your camera…" if self.mode.currentText()=="Camera" else "Preparing the test pattern…")

    def next_frame(self):
        worker = self.worker
        if worker is None:
            return
        if worker.error:
            error = worker.error
            self.stop()
            self.status.setText(error)
            return
        with worker.lock:
            result = worker.latest
        if result is None or result[0]==self.sequence:
            return
        self.sequence,frame,output,fps,ms = result
        self.output_button.setEnabled(True)
        self.output_button.setText("Hide TV image" if self.fullscreen.isVisible() else "Show on TV")
        self.preview.set_frame(frame)
        self.output_preview.set_frame(output)
        if self.fullscreen.isVisible():
            self.fullscreen.video.set_frame(output)
        seg = " • background removal unavailable" if worker.segmentation_available is False and self.background.isChecked() else ""
        slow = " • For smoother motion try Balanced or disable background removal" if self.mode.currentText()=="Camera" and 0<fps<24 else ""
        self.technical_status.setText(f"Camera {frame.shape[1]}×{frame.shape[0]} • render {output.shape[1]}×{output.shape[0]} • {fps:.1f} processed fps • {ms:.0f} ms processing{seg}{slow}")
        message = "Showing on your selected display. Press Esc there to return." if self.fullscreen.isVisible() else "Preview is ready. Press Show on TV when you're ready."
        if self.mode.currentText()!="Camera":
            message = "Test pattern ready. Open Advanced settings to adjust the cone fit."
        if worker.segmentation_available is False and self.background.isChecked():
            message += " Background removal isn't available on this Mac."
        elif self.mode.currentText()=="Camera" and 0<fps<24:
            message += " Motion slow? Choose Smoother motion."
        self.status.setText(message)

    def open_output(self):
        if self.fullscreen.isVisible():
            self.fullscreen.close()
            self.output_button.setText("Show on TV")
            return
        screen = self.screen.currentData()
        if screen is not None:
            self.fullscreen.open_on(screen)

    def stop(self):
        if self.worker is not None and not self.worker.stop():
            self.status.setText("Camera is still stopping. Wait a moment, then press Stop again.")
            return False
        self.worker = None
        self.fullscreen.close()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.output_button.setEnabled(False)
        self.output_button.setText("Show on TV")
        self.status.setText("Camera stopped. Press Start camera to begin again.")
        return True

    def shutdown(self):
        self.stop()
        self.timer.stop()
