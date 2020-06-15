import os
import numpy as np
import subprocess as sp
import random
import cv2
import sys
import ctypes
import datetime as dt
import socket
import configparser as cp

from PySide2 import QtCore as qtc
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2.QtCore import Signal, Slot

from keras.callbacks import EarlyStopping, Callback


class NtfyProgress(Callback):
    def on_train_begin(self, logs={}):
        print(self.params)
        hostname = socket.gethostname()
        epoch = self.params["epochs"]
        steps = self.params["steps"]
        sp.run(
            "ntfy -t \"Begin Training\" send \"Epoch: {} Steps: {}\"".format(
                epoch, steps),
            shell=True)

    def on_epoch_end(self, epoch, logs={}):
        print(epoch)
        print(logs)
        hostname = socket.gethostname()
        cmd = """ntfy -t "End of Epoch {} in {}" send "

*loss:*  {:.2f}
*acc:* {:.2f}
*val_loss:* {:.2f}
*val_acc:* {:.2f}
"

""".format(epoch, hostname, logs["loss"], logs["acc"], logs["val_loss"],
           logs["val_acc"])
        sp.run(cmd, shell=True)


def tensorflow_config():
    from tensorflow import ConfigProto
    config = ConfigProto()
    #if tf.test.gpu_device_name():
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    return config


def shuffle_parallel(*args):
    for a in args:
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def config_filename():
    hostname = socket.gethostname()
    config_fn = "facebin-config-{}.ini".format(hostname)
    return config_fn


def get_configuration():
    fn = config_filename()
    config = cp.ConfigParser()
    if os.path.exists(fn):
        config.read(fn)
    else:
        config['general'] = {'dataset-dir': 'dataset-images/user'}
        config['camera1'] = {
            'id': 'camera1',
            'name': 'Default Camera',
            'device': '/dev/video0',
            'fps': '25',
            'command': ''
        }
        with open(fn, 'w') as f:
            config.write(f)

    return config


def save_configuration(config):
    fn = config_filename()
    with open(fn, 'w') as f:
        config.write(f)


@static_vars(LOG=None)
def init_logging():
    """configures logging for the application

    >>> ll1 = init_logging()
    >>> ll2 = init_logging()
    >>> assert(ll1 == ll2)
    >>> ll1.info("This is info")
    >>> ll2.debug("This is debug")
    >>> ll1.warning("This is warning")
    >>> ll2.error("This is error")
    """
    if init_logging.LOG is None:
        import logging
        init_logging.LOG = logging.getLogger("facebin")
        init_logging.LOG.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '{filename}::{lineno} \t {funcName} - {levelname} - {message}',
            style='{')

        fh = logging.FileHandler('/tmp/facebin-debug-{}.log'.format(
            dt.datetime.now().strftime("%F%H%M%S.%f")))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        init_logging.LOG.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        init_logging.LOG.addHandler(ch)

        init_logging.LOG.debug("Logging Initialized")
        return init_logging.LOG
    else:
        return init_logging.LOG


log = init_logging()


def extract_keyframes(video_filename, output_dir=None):
    """extract keyframes from a video file using ffmpeg and return a list of
filenames of keyframes. the output is placed normally under a random dir in
/tmp

    >>> fn = os.path.expandvars("$HOME/Repository/facebin/test-input/camera-2018-10-23-11-44-19-470021.mpeg")
    >>> keyframes = extract_keyframes(fn)
    >>> len(keyframes)
    64
    >>> keyframes = extract_keyframes(fn, output_dir="/tmp/camera-keyframes/")
    >>> filenames = os.listdir('/tmp/camera-keyframes/')
    >>> for k, f in zip(sorted(keyframes), sorted(filenames)):
    ...     assert k == f
    """

    if output_dir is None:
        output_dir = '/tmp/{}'.format(random.randint(100000, 1000000))

    vf = os.path.basename(video_filename)
    vfh, vfe = os.path.splitext(vf)
    command = """/usr/bin/ffmpeg -i {} -vf select=eq(pict_type\,PICT_TYPE_I) -vsync 2 -f image2 {}/keyframe_{}_%04d.jpeg""".format(
        video_filename, output_dir, vfh)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    sp.call(command.split())

    filenames = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]

    return filenames


def get_qimage(image: np.ndarray):
    global log
    # assert (np.max(image) <= 255)
    # swap rgb
    # image = image[..., ::-1]
    # image8 = image.astype(np.uint8, order='C', casting='unsafe')
    height, width, colors = image.shape
    log.debug("h: {} w: {} c: {}".format(height, width, colors))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # The following code is due to a bug in QImage memory usage.
    # We manually modify the Python reference count!
    # See https://bugreports.qt.io/browse/PYSIDE-140
    ch = ctypes.c_char.from_buffer(image.data, 0)
    rcount = ctypes.c_long.from_address(id(ch)).value
    bytesPerLine = 3 * width
    qi = qtg.QImage(ch, width, height, bytesPerLine, qtg.QImage.Format_RGB888)
    qi.rgbSwapped_inplace()
    ctypes.c_long.from_address(id(ch)).value = rcount

    # data = qtc.QByteArray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR).tostring())
    # log.debug("data.size(): %s", data.size())
    # qi = qtg.QImage(data.data(), width, height, bytesPerLine,
    #                 qtg.QImage.Format_RGB888)

    log.debug("qimage h: %s w: %s", qi.height(), qi.width())

    # data.clear()
    # del data
    return qi


if __name__ == "__main__":
    import doctest
    doctest.testmod()
