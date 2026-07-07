"""Shared helpers for Facebin server processes.

This module must stay importable in headless environments: it must not
import Qt, TensorFlow, or Keras at module level.  GUI helpers live in
:mod:`facebin.ui.qt_utils`.
"""

import datetime as dt
import logging
import os
import random
import subprocess as sp
import sys

import numpy as np

LOG_DIR = os.environ.get("FACEBIN_LOG_DIR", "/tmp/facebin-logs")


def static_vars(**kwargs):
    """Attach static variables to a function (used for memoization)."""

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(LOG=None)
def init_logging():
    """Configure and return the shared ``facebin`` logger.

    The logger writes DEBUG output to a timestamped file under
    ``$FACEBIN_LOG_DIR`` (default ``/tmp/facebin-logs``) and WARNING and
    above to stdout.  Repeated calls return the same logger.

    >>> ll1 = init_logging()
    >>> ll2 = init_logging()
    >>> assert ll1 is ll2
    """
    if init_logging.LOG is None:
        logger = logging.getLogger("facebin")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '{asctime} {filename}::{lineno} \t {funcName} - {levelname} - '
            '{message}',
            style='{')

        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            log_path = os.path.join(
                LOG_DIR, 'facebin-debug-{}-pid{}.log'.format(
                    dt.datetime.now().strftime("%F-%H%M%S.%f"), os.getpid()))
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError as e:
            # Logging must never crash the application; fall back to stdout.
            print("facebin: cannot open log file in {}: {}".format(
                LOG_DIR, e), file=sys.stderr)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.debug("Logging Initialized")
        init_logging.LOG = logger
    return init_logging.LOG


log = init_logging()


def tensorflow_config():
    """Build a TensorFlow session config that grows GPU memory on demand.

    TensorFlow is imported lazily so that headless installs without the
    ``ml`` extra can import this module.
    """
    try:
        from tensorflow.compat.v1 import ConfigProto
    except ImportError:
        from tensorflow import ConfigProto
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def shuffle_parallel(*args):
    """Shuffle several arrays in place with the same permutation."""
    for a in args:
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)


def extract_keyframes(video_filename, output_dir=None):
    """Extract keyframes from a video with ffmpeg.

    Returns the list of extracted image paths.  When *output_dir* is None a
    random directory under /tmp is used.
    """
    if output_dir is None:
        output_dir = '/tmp/facebin-keyframes-{}'.format(
            random.randint(100000, 1000000))

    vf = os.path.basename(video_filename)
    vfh, _ = os.path.splitext(vf)
    os.makedirs(output_dir, exist_ok=True)
    command = [
        'ffmpeg', '-i', video_filename, '-vf',
        r'select=eq(pict_type\,PICT_TYPE_I)', '-vsync', '2', '-f', 'image2',
        os.path.join(output_dir, 'keyframe_{}_%04d.jpeg'.format(vfh))
    ]

    result = sp.run(command, capture_output=True)
    if result.returncode != 0:
        log.error("ffmpeg failed (%s): %s", result.returncode,
                  result.stderr.decode("utf-8", errors="replace")[-1000:])

    return [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir))]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
