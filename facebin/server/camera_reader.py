"""Camera reader worker process.

Reads frames from a camera device (or stream) with PyAV and pushes them to
the Redis ``CAMERA_QUEUE`` for the recognizer processes.  One reader process
runs per configured camera; it is started and supervised by
:class:`facebin.server.FacebinServer`.
"""

import datetime as dt
import os
import sys
import time

from facebin.config import load_config
from facebin.errors import CameraError, DependencyError
from . import redis_queue_utils as rqu
from .utils import init_logging

log = init_logging()


def _video_settings(config):
    video_dir = config.video.resolved_dir()
    return video_dir, config.video.video_seconds_per_file


def get_video_index(t, video_record_period):
    return (t // video_record_period) * video_record_period


def get_video_filename(t, cam_id, video_record_dir, video_record_period):
    """Return the video file a frame at timestamp ``t`` belongs to."""
    try:
        video_index = get_video_index(t, video_record_period)
        tstr = dt.datetime.fromtimestamp(video_index).strftime("%F-%H-%M-%S")
    except (ValueError, OSError, OverflowError):
        # Stream timestamps (dts) are not always valid UNIX times.
        video_index = get_video_index(time.time(), video_record_period)
        tstr = dt.datetime.fromtimestamp(video_index).strftime("%F-%H-%M-%S")
    return os.path.join(video_record_dir,
                        "camera-{}-index-{}.mp4".format(cam_id, tstr))


def open_container(camera_device):
    """Open a PyAV container for a device, with a meaningful error."""
    try:
        import av
    except ImportError as e:
        raise DependencyError(
            "PyAV is not installed; camera frames cannot be read.",
            hint="Install it with `pip install av`.") from e
    try:
        container = av.open(camera_device, 'r')
    except Exception as e:
        raise CameraError(
            "Cannot open camera device '{}': {}".format(camera_device, e),
            hint="Check that the device exists (e.g. `ls /dev/video*`), "
            "that the RTSP URL and credentials are correct, and that no "
            "other process is using the camera.") from e
    if not container.streams.video:
        raise CameraError(
            "Device '{}' has no video stream.".format(camera_device))
    return container


def reader_loop(camera_id, camera_device, max_fps=25, config=None):
    """Read frames from a camera and enqueue them until the process dies."""
    if config is None:
        config = load_config()
    rqu.configure(config.redis)
    video_record_dir, video_record_period = _video_settings(config)

    log_path = "/tmp/facebin-camera-reader-out-pid-{}.txt".format(os.getpid())
    outfile = open(log_path, "a")
    sys.stdout = outfile
    sys.stderr = outfile

    container = open_container(camera_device)
    container.streams.video[0].thread_type = 'AUTO'

    camera_prev_time = time.time()
    skip_threshold = 10
    skips_before_quit = 100
    skips = 0
    min_delay = 1.0 / max_fps

    for frame in container.decode(video=0):
        if rqu.queue_length(rqu.CAMERA_QUEUE) > skip_threshold:
            log.warning("Camera %s: queue is full; waiting for recognizers "
                        "to catch up.", camera_id)
            time.sleep(min_delay)
            skips += 1
            if skips >= skips_before_quit:
                raise CameraError(
                    "Camera {}: nothing consumed the frame queue for {} "
                    "cycles; giving up.".format(camera_id, skips),
                    hint="Check that recognizer processes are running and "
                    "that Redis is reachable.")
        else:
            skips = 0

        if frame.dts is None:
            continue
        camera_current_time = time.time()
        camera_diff = camera_current_time - camera_prev_time
        if camera_diff < min_delay:
            time.sleep(min_delay - camera_diff)
            continue

        current_video_filename = get_video_filename(
            frame.dts, camera_id, video_record_dir, video_record_period)

        rqu.init_frame(frame.to_ndarray(format='bgr24'), frame.dts,
                       current_video_filename, camera_id)
        camera_prev_time = time.time()

    raise CameraError(
        "Camera {}: stream '{}' ended unexpectedly.".format(
            camera_id, camera_device),
        hint="For live cameras this usually means the connection dropped; "
        "the server supervisor will restart this reader.")


if __name__ == '__main__':
    reader_loop('camera1', '/dev/video0')
