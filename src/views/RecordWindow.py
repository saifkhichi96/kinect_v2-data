from typing import Optional

import cv2
import numpy as np
from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QMainWindow,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QCheckBox,
    QSlider
)

from models import KinectV2


class RecordWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(np.ndarray)

    def __init__(self, config: KinectV2.Config, device, listener, registration):
        super(RecordWorker, self).__init__()
        self.config = config
        self.device = device
        self.listener = listener
        self.registration = registration

    # noinspection PyArgumentList,PyUnresolvedReferences
    def run(self):
        """Long-running task."""

        def on_new_frame(frame):
            self.progress.emit(frame)
            pass

        KinectV2.record(on_new_frame, self.config, self.device, self.listener, self.registration)
        self.finished.emit()


class RecordWindow(QMainWindow):
    """Main Window."""

    def __init__(self):
        """Initializer."""
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[RecordWorker] = None

        self.init_ui(title="Kinect Scanner")
        self.center()

        # self.button_record.clicked.connect(self.record)

    # noinspection PyAttributeOutsideInit
    def init_ui(self, title):
        self.setWindowTitle(title)
        self.resize(1024, 768)
        self.centralWidget = QLabel()
        self.centralWidget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setCentralWidget(self.centralWidget)

        self.root = QVBoxLayout(self.centralWidget)

        self.row1 = QHBoxLayout()
        self.root.addLayout(self.row1)

        self.row2 = QHBoxLayout()
        self.root.addLayout(self.row2)

        # # Main area (shows video feed)
        # self.video = QLabel()
        # self.root.addWidget(self.video)
        # self.current_frame = np.zeros((648, 824, 3)).astype(np.uint8)
        # self.update_feed(self.current_frame)

        # Viewport group
        viewport_layout = QVBoxLayout()
        self.row1.addLayout(viewport_layout)

        clip = QLabel()
        clip.setFixedWidth(300)
        clip.setText("Viewport")
        clip.setFont(QFont('Arial', 16))
        viewport_layout.addWidget(clip)

        self.left_label = QLabel()
        self.left_label.setText("Left: 0")
        self.left_label.setFont(QFont('Arial', 12))
        viewport_layout.addWidget(self.left_label)

        self.left = QSlider(Qt.Horizontal)
        self.left.setFixedWidth(300)
        self.left.setRange(0, 411)
        self.left.valueChanged.connect(self.on_left_changed)
        self.left.setValue(0)
        viewport_layout.addWidget(self.left)

        self.right_label = QLabel()
        self.right_label.setText("Right: WIDTH")
        self.right_label.setFont(QFont('Arial', 12))
        viewport_layout.addWidget(self.right_label)

        self.right = QSlider(Qt.Horizontal)
        self.right.setFixedWidth(300)
        self.right.setRange(412, 823)
        self.right.valueChanged.connect(self.on_right_changed)
        self.right.setValue(823)
        viewport_layout.addWidget(self.right)

        self.top_label = QLabel()
        self.top_label.setText("Top: 0")
        self.top_label.setFont(QFont('Arial', 12))
        viewport_layout.addWidget(self.top_label)

        self.top = QSlider(Qt.Horizontal)
        self.top.setFixedWidth(300)
        self.top.setRange(0, 323)
        self.top.valueChanged.connect(self.on_top_changed)
        self.top.setValue(0)
        viewport_layout.addWidget(self.top)

        self.bottom_label = QLabel()
        self.bottom_label.setText("Bottom: HEIGHT")
        self.bottom_label.setFont(QFont('Arial', 12))
        viewport_layout.addWidget(self.bottom_label)

        self.bottom = QSlider(Qt.Horizontal)
        self.bottom.setFixedWidth(300)
        self.bottom.setRange(324, 647)
        self.bottom.valueChanged.connect(self.on_bottom_changed)
        self.bottom.setValue(647)
        viewport_layout.addWidget(self.bottom)

        self.z_min_label = QLabel()
        self.z_min_label.setText("Near: 500")
        self.z_min_label.setFont(QFont('Arial', 12))
        viewport_layout.addWidget(self.z_min_label)

        self.z_min = QSlider(Qt.Horizontal)
        self.z_min.setFixedWidth(300)
        self.z_min.setRange(500, 1250)
        self.z_min.valueChanged.connect(self.on_z_min_changed)
        self.z_min.setValue(500)
        viewport_layout.addWidget(self.z_min)

        self.z_max_label = QLabel()
        self.z_max_label.setText("Far: 1250")
        self.z_max_label.setFont(QFont('Arial', 12))
        viewport_layout.addWidget(self.z_max_label)

        self.z_max = QSlider(Qt.Horizontal)
        self.z_max.setFixedWidth(300)
        self.z_max.setRange(500, 4500)
        self.z_max.valueChanged.connect(self.on_z_max_changed)
        self.z_max.setValue(1250)
        viewport_layout.addWidget(self.z_max)

        masks_layout = QVBoxLayout()
        self.row1.addLayout(masks_layout)

        masks = QLabel()
        masks.setFixedWidth(300)
        masks.setText("Masks")
        masks.setFont(QFont('Arial', 16))
        masks_layout.addWidget(masks)

        self.skin_mask = QCheckBox("Skin")
        self.skin_mask.setChecked(True)
        masks_layout.addWidget(self.skin_mask)

        self.artefact_mask = QCheckBox("Noise")
        self.artefact_mask.setChecked(True)
        masks_layout.addWidget(self.artefact_mask)

        masks = QLabel()
        masks.setText("Recording Settings")
        masks.setFont(QFont('Arial', 16))
        masks_layout.addWidget(masks)

        duration_label = QLabel()
        duration_label.setText("Duration:")
        duration_label.setFont(QFont('Arial', 12))
        masks_layout.addWidget(duration_label)

        self.duration = QLineEdit()
        self.duration.setFixedWidth(300)
        masks_layout.addWidget(self.duration)

        delay_label = QLabel()
        delay_label.setText("Delay:")
        delay_label.setFont(QFont('Arial', 12))
        masks_layout.addWidget(delay_label)

        self.delay = QLineEdit()
        self.delay.setFixedWidth(300)
        masks_layout.addWidget(self.delay)

        fps_label = QLabel()
        fps_label.setText("Frame Rate:")
        fps_label.setFont(QFont('Arial', 12))
        masks_layout.addWidget(fps_label)

        self.fps = QLineEdit()
        self.fps.setFixedWidth(300)
        masks_layout.addWidget(self.fps)

        self.button_record = QPushButton('Preview')
        masks_layout.addWidget(self.button_record)

        sequence_layout = QVBoxLayout()
        self.row2.addLayout(sequence_layout)

        surface_label = QLabel()
        surface_label.setText("Surface:")
        surface_label.setFont(QFont('Arial', 12))
        sequence_layout.addWidget(surface_label)

        self.surface = QLineEdit()
        self.surface.setFixedWidth(300)
        sequence_layout.addWidget(self.surface)

        self.button_save = QPushButton('Record')
        self.button_save.setFixedWidth(300)
        sequence_layout.addWidget(self.button_save)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def on_left_changed(self):
        left = self.left.value()
        self.left_label.setText("Left: " + str(left))

    def on_right_changed(self):
        right = 823 - self.right.value()
        if right > 0:
            self.right_label.setText("Right: WIDTH-" + str(right))
        else:
            self.right_label.setText("Right: WIDTH")

    def on_top_changed(self):
        top = self.top.value()
        self.top_label.setText("Top: " + str(top))

    def on_bottom_changed(self):
        bottom = 647 - self.bottom.value()
        if bottom > 0:
            self.bottom_label.setText("Bottom: HEIGHT-" + str(bottom))
        else:
            self.bottom_label.setText("Bottom: HEIGHT")

    def on_z_min_changed(self):
        z_min = self.z_min.value()
        self.z_min_label.setText("Near: " + str(z_min))
        self.z_max.setMinimum(z_min)
        # if self.worker is not None:
        #     self.worker.settings.z_min = z_min

    def on_z_max_changed(self):
        z_max = self.z_max.value()
        self.z_max_label.setText("Far: " + str(z_max))
        self.z_min.setMaximum(z_max)
        # if self.worker is not None:
        #     self.worker.settings.z_max = z_max

    # noinspection PyUnresolvedReferences
    def record(self):
        self.thread = QThread()

        config = KinectV2.Config(duration=0, delay=0, rate=5)
        self.worker = RecordWorker(config, self.device, self.listener, self.registration)

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.update_feed)
        self.thread.start()

        self.button_record.setEnabled(False)
        self.thread.finished.connect(lambda: self.button_record.setEnabled(True))

    def update_feed(self, frame):
        """Displays video feed with current frame.

        :param frame: A new frame from the camera.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

        self.current_frame = frame
        self.video.setPixmap(QPixmap(image))
