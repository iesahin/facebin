import sys

from PySide2 import QtCore as qtc
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2.QtCore import Signal, Slot

import datetime as dt

import numpy as np

from utils import *

log = init_logging()


class ShowImageDialog(qtw.QDialog):
    def __init__(self, image, parent=None):
        super(ShowImageDialog, self).__init__(parent)

        if isinstance(image, str):
            self.image = qtg.QImage(image)
        elif isinstance(image, np.ndarray):
            self.image = get_qimage(image)
        elif isinstance(image, qtg.QImage):
            self.image = image
        else:
            log.error("Unrecognized Image Parameter: %s", image)
            self.image = None

        if self.image is None:
            self.image_label = qtw.QLabel("Cannot Load Image")
        else:
            self.image_label = qtw.QLabel()
            self.image_label.setPixmap(qtg.QPixmap.fromImage(self.image))

        main_layout = qtw.QVBoxLayout()
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)


# image_profile = QtGui.QImage(image_path)  #QImage object
# image_profile = image_profile.scaled(
#     250,
#     250,
#     aspectRatioMode=QtCore.Qt.KeepAspectRatio,
#     transformMode=QtCore.Qt.SmoothTransformation
# )  # To scale image for example and keep its Aspect Ration
# label_Image.setPixmap(QtGui.QPixmap.fromImage(image_profile))
