"""Interactive 3D model proof for the Mac Pepper's Cone application."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QSettings
from PySide6.QtGui import QImage, QKeyEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider,
    QVBoxLayout, QWidget, QApplication)

from mesh_renderer import RenderSettings, checker_texture, four_view_proof, render_model
from model_warp import warp_model_views
from projection import ProjectionSettings


class ModelPreview(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300,220)
        self.setStyleSheet("background:#070b13;color:#8998b2;border-radius:11px;")
        self.source = None

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w = rgb.shape[:2]
        self.source = QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],
                                               QImage.Format.Format_RGB888).copy())
        self.draw()

    def draw(self):
        if self.source is not None:
            self.setPixmap(self.source.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.draw()


class ModelTVWindow(QWidget):
    closed = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Output")
        self.setStyleSheet("background:black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.preview = ModelPreview()
        self.preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self.preview)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape),self)
        self.escape_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.escape_shortcut.activated.connect(self.dismiss)
        self.q_shortcut = QShortcut(QKeySequence("Q"),self)
        self.q_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.q_shortcut.activated.connect(self.dismiss)

    def open_on(self,screen,fullscreen=True):
        self.winId()  # Create the native window before assigning its display.
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(screen)
        if fullscreen:
            self.setGeometry(screen.geometry())
            self.showFullScreen()
        else:
            self.showNormal()
            area = screen.availableGeometry()
            width = min(1100,round(area.width()*.72))
            height = min(700,round(area.height()*.72))
            self.setGeometry(area.x()+area.width()-width-30,
                             area.y()+30,width,height)
            self.show()
            self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def dismiss(self):
        if not self.isVisible():
            return
        if self.isFullScreen():
            self.showNormal()
        self.hide()
        self.closed.emit()

    def closeEvent(self,event):
        self.closed.emit()
        super().closeEvent(event)

    def keyPressEvent(self,event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Escape,Qt.Key.Key_Q):
            self.dismiss()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self,event):
        self.dismiss()
        event.accept()


class ModelPage(QWidget):
    def __init__(self):
        super().__init__()
        self.texture = checker_texture()
        self.storage = QSettings("PeppersCone","MacModelStudio")
        self.tv = ModelTVWindow()
        self.angle = 25
        self.current_raw = None
        self.current_output = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0,4,0,0)
        title = QLabel("3D model")
        title.setObjectName("title")
        root.addWidget(title)
        subtitle = QLabel("Render real geometry from different viewpoints, then test the cone warp.")
        subtitle.setObjectName("subtitle")
        root.addWidget(subtitle)

        body = QHBoxLayout()
        body.setSpacing(18)
        panel = QWidget()
        panel.setObjectName("controlCard")
        panel.setMinimumWidth(285)
        panel.setMaximumWidth(325)
        form = QFormLayout(panel)
        form.setContentsMargins(20,20,20,20)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        section = QLabel("Model and viewpoint")
        section.setObjectName("section")
        form.addRow(section)
        self.model = QComboBox()
        self.model.addItems(["Cube","Sphere"])
        form.addRow("Shape",self.model)
        self.layout_mode = QComboBox()
        self.layout_mode.addItems(["Single perspective view","Four different angles"])
        form.addRow("View test",self.layout_mode)
        self.output_position = QComboBox()
        for label,rotation in (("Top",270),("Right",0),("Bottom",90),("Left",180)):
            self.output_position.addItem(label,rotation)
        form.addRow("Cone output side",self.output_position)
        self.yaw = QSlider(Qt.Orientation.Horizontal)
        self.yaw.setRange(0,359)
        self.yaw.setValue(25)
        form.addRow("Viewing angle",self.yaw)
        turn_row = QWidget()
        turn_layout = QHBoxLayout(turn_row)
        turn_layout.setContentsMargins(0,0,0,0)
        turn_layout.setSpacing(6)
        self.turn_left_button = QPushButton("↶ 45°")
        self.turn_left_button.setToolTip("Turn the cube 45 degrees left")
        self.turn_left_button.clicked.connect(lambda: self.turn_model(-45))
        self.flip_button = QPushButton("Flip 180°")
        self.flip_button.setToolTip("Show the opposite side of the cube")
        self.flip_button.clicked.connect(lambda: self.turn_model(180))
        self.turn_right_button = QPushButton("45° ↷")
        self.turn_right_button.setToolTip("Turn the cube 45 degrees right")
        self.turn_right_button.clicked.connect(lambda: self.turn_model(45))
        for button in (self.turn_left_button,self.flip_button,self.turn_right_button):
            turn_layout.addWidget(button)
        form.addRow("Turn model",turn_row)
        self.elevation = QSlider(Qt.Orientation.Horizontal)
        self.elevation.setRange(-45,45)
        self.elevation.setValue(15)
        form.addRow("Elevation",self.elevation)
        self.auto_rotate = QCheckBox("Rotate model automatically")
        self.auto_rotate.setChecked(True)
        form.addRow(self.auto_rotate)
        self.advanced_button = QPushButton("Advanced settings ▸")
        self.advanced_button.setObjectName("quiet")
        self.advanced_button.setCheckable(True)
        form.addRow(self.advanced_button)
        load = QPushButton("Use image as texture")
        load.clicked.connect(self.load_texture)
        form.addRow(load)
        restore = QPushButton("Restore checker texture")
        restore.clicked.connect(self.restore_texture)
        form.addRow(restore)
        self.output_size = QComboBox()
        self.output_size.addItem("720p output",(1280,720))
        self.output_size.addItem("1080p output",(1920,1080))
        form.addRow("TV output",self.output_size)
        self.screen = QComboBox()
        self.refresh_screens()
        form.addRow("Show on",self.screen)
        refresh = QPushButton("Refresh displays")
        refresh.clicked.connect(self.refresh_screens)
        form.addRow(refresh)
        self.fullscreen_output = QCheckBox("Fullscreen TV output")
        self.fullscreen_output.setChecked(len(QApplication.screens())>1)
        self.fullscreen_output.setToolTip("Leave off when the TV is the only detected display so the model controls stay accessible.")
        form.addRow(self.fullscreen_output)
        self.mirror = QCheckBox("Flip horizontally for reflection")
        self.mirror.setChecked(self.storage.value("mirror",True,type=bool))
        self.invert = QCheckBox("Reverse inner and outer edge")
        self.invert.setChecked(self.storage.value("invert",False,type=bool))
        form.addRow(self.mirror)
        form.addRow(self.invert)
        self.show_button = QPushButton("Show model on TV")
        self.show_button.setObjectName("tvAction")
        self.show_button.clicked.connect(self.toggle_tv)
        form.addRow(self.show_button)
        self.advanced_panel = QWidget()
        advanced = QFormLayout(self.advanced_panel)
        advanced.setContentsMargins(0,0,0,0)
        advanced.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        model_section = QLabel("3D model")
        model_section.setObjectName("section")
        advanced.addRow(model_section)
        self.model_scale = QDoubleSpinBox()
        self.model_scale.setRange(30,180)
        self.model_scale.setSingleStep(5)
        self.model_scale.setValue(float(self.storage.value("model_scale",100)))
        self.model_scale.valueChanged.connect(self.render)
        advanced.addRow("3D model size %",self.model_scale)
        self.model_fov = QDoubleSpinBox()
        self.model_fov.setRange(25,80)
        self.model_fov.setSingleStep(1)
        self.model_fov.setValue(float(self.storage.value("model_fov",42)))
        self.model_fov.valueChanged.connect(self.render)
        advanced.addRow("Camera field of view °",self.model_fov)
        cone_section = QLabel("Cone layout")
        cone_section.setObjectName("section")
        advanced.addRow(cone_section)
        self.spins = {}
        for name,label,minimum,maximum,value,step in [
            ("diameter","Cone diameter %",10,100,94,1),
            ("inner","Inner radius %",0,85,12,1),
            ("rotation","Rotate layout °",0,359,270,5),
            ("center_x","Center across TV %",10,90,50,1),
            ("center_y","Center down TV %",10,90,50,1),
            ("zoom","Model size inside cone %",20,150,85,5),
            ("gap","Sector gap °",0,8,2,.5),
            ("gain","Brightness gain",.5,2,1.15,.05)]:
            spin = QDoubleSpinBox()
            spin.setRange(minimum,maximum)
            spin.setSingleStep(step)
            spin.setDecimals(2 if name=="gain" else 1)
            spin.setValue(float(self.storage.value(name,value)))
            spin.valueChanged.connect(self.render)
            self.spins[name] = spin
            advanced.addRow(label,spin)
        for option in (self.mirror,self.invert):
            option.toggled.connect(self.render)
        save_fit = QPushButton("Save model cone fit")
        save_fit.clicked.connect(self.save_fit)
        advanced.addRow(save_fit)
        reset_fit = QPushButton("Reset advanced settings")
        reset_fit.clicked.connect(self.reset_advanced)
        advanced.addRow(reset_fit)
        form.addRow(self.advanced_panel)
        self.advanced_panel.hide()
        self.advanced_button.toggled.connect(self.toggle_advanced)
        note = QLabel("Research milestone: this is real 3D geometry. The current cone warp is only a starting approximation; measured optical calibration comes next.")
        note.setObjectName("tip")
        note.setWordWrap(True)
        form.addRow(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(325)
        scroll.setWidget(panel)
        self.control_scroll = scroll
        body.addWidget(scroll)

        previews = QVBoxLayout()
        previews.setSpacing(14)
        self.raw_preview = ModelPreview("Rendering model…")
        self.warp_preview = ModelPreview("Preparing cone output…")
        for heading,caption,widget in (("Perspective render","REAL 3D VIEW",self.raw_preview),
                                       ("Cone output","UNCALIBRATED PRE-WARP",self.warp_preview)):
            card = QFrame()
            card.setObjectName("previewCard")
            card_layout = QVBoxLayout(card)
            top = QHBoxLayout()
            label = QLabel(heading)
            label.setObjectName("section")
            top.addWidget(label)
            top.addStretch()
            tag = QLabel(caption)
            tag.setObjectName("eyebrow")
            top.addWidget(tag)
            card_layout.addLayout(top)
            card_layout.addWidget(widget,1)
            previews.addWidget(card,1)
        body.addLayout(previews,1)
        root.addLayout(body,1)
        self.status = QLabel("The cube and sphere are generated in the app—not repeated camera snapshots.")
        self.status.setObjectName("status")
        root.addWidget(self.status)

        for widget in (self.model,self.layout_mode,self.output_size):
            widget.currentIndexChanged.connect(self.render)
        self.output_position.currentIndexChanged.connect(self.change_output_side)
        for slider in (self.yaw,self.elevation):
            slider.valueChanged.connect(self.render)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)
        self.tv.closed.connect(self.tv_closed)
        QApplication.instance().screenAdded.connect(self.refresh_screens)
        QApplication.instance().screenRemoved.connect(self.refresh_screens)
        self.render()

    def settings(self,width=960,height=540,yaw=None):
        return RenderSettings(width=width,height=height,
            yaw=self.yaw.value() if yaw is None else yaw,
            elevation=self.elevation.value(),model=self.model.currentText(),
            scale=self.model_scale.value()/100,fov=self.model_fov.value())

    def cone_settings(self):
        values = {name:spin.value() for name,spin in self.spins.items()}
        for name in ("diameter","inner","center_x","center_y","zoom"):
            values[name] /= 100
        return ProjectionSettings(mirror=self.mirror.isChecked(),
            invert=self.invert.isChecked(),portrait_crop=False,
            remove_background=False,**values)

    def render(self,*args):
        settings = self.settings(640,360)
        views = four_view_proof(settings,self.texture) if self.layout_mode.currentIndex() else [render_model(settings,self.texture)]
        self.current_raw = views[0]
        self.current_output = warp_model_views(views,960,540,self.cone_settings())
        if len(views)==4:
            comparison = np.zeros((540,960,3),np.uint8)
            labels = ("FRONT","RIGHT","BACK","LEFT")
            for i,(view,label) in enumerate(zip(views,labels)):
                small = cv2.resize(view,(480,270),interpolation=cv2.INTER_AREA)
                y,x = divmod(i,2)
                comparison[y*270:(y+1)*270,x*480:(x+1)*480] = small
                cv2.putText(comparison,label,(x*480+16,y*270+30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
            self.current_raw = comparison
        self.raw_preview.set_frame(self.current_raw)
        self.warp_preview.set_frame(self.current_output)
        if self.tv.isVisible():
            self.render_tv()

    def render_tv(self):
        width,height = self.output_size.currentData()
        # The model source only needs enough pixels for its share of the cone.
        # Rendering every source at the full TV size wastes most of the work.
        settings = self.settings(640,360)
        views = four_view_proof(settings,self.texture) if self.layout_mode.currentIndex() else [render_model(settings,self.texture)]
        self.tv.preview.set_frame(warp_model_views(views,width,height,self.cone_settings()))

    def tick(self):
        self.timer.setInterval(250 if self.tv.isVisible() and self.layout_mode.currentIndex() else 100)
        if self.auto_rotate.isChecked():
            self.yaw.setValue((self.yaw.value()+1)%360)

    def turn_model(self,degrees):
        self.yaw.setValue((self.yaw.value()+degrees)%360)

    def change_output_side(self,*args):
        self.spins["rotation"].setValue(self.output_position.currentData())

    def load_texture(self):
        path,_ = QFileDialog.getOpenFileName(self,"Choose a texture image","","Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            image = cv2.imread(path)
            if image is not None:
                self.texture = image
                self.render()

    def restore_texture(self):
        self.texture = checker_texture()
        self.render()

    def refresh_screens(self,*args):
        selected = self.screen.currentData()
        self.screen.clear()
        screens = QApplication.screens()
        for screen in screens:
            label = screen.name() or "Display"
            if screen==QApplication.primaryScreen():
                label += " (primary)"
            self.screen.addItem(label,screen)
        index = next((i for i,screen in enumerate(screens) if screen==selected),
                     1 if len(screens)>1 else 0)
        self.screen.setCurrentIndex(index)
        if hasattr(self,"status") and len(screens)==1:
            self.status.setText("One display detected. TV output will open in a movable window unless Fullscreen TV output is enabled.")

    def toggle_tv(self):
        if self.tv.isVisible():
            self.tv.dismiss()
            return
        screen = self.screen.currentData() or QApplication.primaryScreen()
        fullscreen = self.fullscreen_output.isChecked()
        self.tv.open_on(screen,fullscreen)
        self.render_tv()
        self.show_button.setText("Hide TV output")
        mode = "fullscreen" if fullscreen else "in a movable window"
        self.status.setText(f"TV output is live {mode}. Keep changing the model here, or press Hide TV output. Escape/Q also closes it.")

    def tv_closed(self):
        self.show_button.setText("Show model on TV")

    def toggle_advanced(self,visible):
        self.advanced_panel.setVisible(visible)
        self.advanced_button.setText("Advanced settings ▾" if visible else "Advanced settings ▸")
        if visible:
            QTimer.singleShot(0,lambda: self.control_scroll.ensureWidgetVisible(self.advanced_panel))

    def save_fit(self):
        for name,spin in self.spins.items():
            self.storage.setValue(name,spin.value())
        self.storage.setValue("mirror",self.mirror.isChecked())
        self.storage.setValue("invert",self.invert.isChecked())
        self.storage.setValue("model_scale",self.model_scale.value())
        self.storage.setValue("model_fov",self.model_fov.value())
        self.status.setText("3D model cone fit saved on this Mac.")

    def reset_advanced(self):
        defaults = {"diameter":94,"inner":12,"rotation":270,"center_x":50,
                    "center_y":50,"zoom":85,"gap":2,"gain":1.15}
        self.model_scale.setValue(100)
        self.model_fov.setValue(42)
        for name,value in defaults.items():
            self.spins[name].setValue(value)
        self.mirror.setChecked(True)
        self.invert.setChecked(False)
        self.output_position.setCurrentText("Top")
        self.status.setText("Advanced settings reset.")

    def shutdown(self):
        self.timer.stop()
        self.tv.close()
