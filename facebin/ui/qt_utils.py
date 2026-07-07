"""Qt helpers shared by the Facebin GUI widgets."""

import ctypes

import numpy as np

from facebin.server.utils import init_logging
from .qt_compat import qtg

log = init_logging()


def get_qimage(image: np.ndarray) -> "qtg.QImage":
    """Convert a BGR numpy image (as produced by OpenCV) to a QImage."""
    height, width, colors = image.shape
    # Work around QImage keeping a borrowed reference to the buffer.
    # See https://bugreports.qt.io/browse/PYSIDE-140
    ch = ctypes.c_char.from_buffer(image.data, 0)
    rcount = ctypes.c_long.from_address(id(ch)).value
    bytes_per_line = 3 * width
    qi = qtg.QImage(ch, width, height, bytes_per_line,
                    qtg.QImage.Format_RGB888)
    qi.rgbSwapped_inplace()
    ctypes.c_long.from_address(id(ch)).value = rcount
    return qi
