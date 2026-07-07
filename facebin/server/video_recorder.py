"""Footage recording to disk.

Writes camera frames into rolling video files (one file per
``video.video_seconds_per_file`` seconds, per camera) under
``video.video_record_dir``.
"""

import datetime as dt
import os
import time

import cv2

from facebin.config import load_config
from .utils import init_logging

log = init_logging()

_video_writers = {}
_last_checked = 0
_config = None


def configure(config):
    global _config
    _config = config


def _get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_video_index(t):
    period = _get_config().video.video_seconds_per_file
    return (t // period) * period


def get_video_filename(t, cam_id):
    video_index = get_video_index(t)
    tstr = dt.datetime.fromtimestamp(video_index).strftime("%F-%H-%M-%S")
    return os.path.join(_get_config().video.resolved_dir(),
                        "camera-{}-index-{}.avi".format(cam_id, tstr))


def record(t, cam_id, camera_image, video_filename):
    """Append a frame to its video file, opening a writer when needed."""
    fn = video_filename
    if fn not in _video_writers:
        os.makedirs(os.path.dirname(fn) or ".", exist_ok=True)
        size = (camera_image.shape[1], camera_image.shape[0])
        vw = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'XVID'), 30, size)
        if not vw.isOpened():
            log.error(
                "Cannot open video writer for '%s'; frames from camera %s "
                "will not be recorded. Check that the directory is writable "
                "and OpenCV has XVID support.", fn, cam_id)
            return
        _video_writers[fn] = (vw, time.time())

    vw, _ = _video_writers[fn]
    vw.write(camera_image)
    _video_writers[fn] = (vw, time.time())
    _close_stale_writers()


def _close_stale_writers():
    """Release writers that have not received frames recently."""
    global _last_checked
    if (time.time() - _last_checked) < 10:
        return
    _last_checked = time.time()
    period = _get_config().video.video_seconds_per_file
    for fn in list(_video_writers):
        vw, last_used = _video_writers[fn]
        if (time.time() - last_used) > (period * 2):
            log.info("Closing stale video writer: %s", fn)
            vw.release()
            del _video_writers[fn]
