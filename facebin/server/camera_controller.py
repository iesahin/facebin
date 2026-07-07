"""Camera lifecycle management.

A :class:`CameraController` wraps one camera defined in a ``[[camera]]``
block of ``facebin.toml``.  When the camera needs a helper process (for
example an ffmpeg relay that converts an RTSP stream to a v4l2 device), the
controller starts and stops that process and captures its output under the
camera log directory.
"""

import datetime as dt
import os
import re
import subprocess as sp

from facebin.config import CameraConfig, load_config, save_config
from facebin.errors import CameraError
from .utils import init_logging

log = init_logging()

CAMLOGDIR = os.environ.get("FACEBIN_CAMERA_LOG_DIR", '/tmp/facebin-cam-logs/')


class CameraController:
    """Controls a single camera and its optional helper command."""

    def __init__(self, camera_id, name, device, command, fps=25):
        if not camera_id:
            raise CameraError("Camera id cannot be empty.")
        # An empty device is allowed here so the GUI can show placeholder
        # entries for unconfigured cameras; reading from such a camera
        # fails with a meaningful CameraError in camera_reader.
        self.camera_id = camera_id
        self.name = name
        self.device = device
        self.command = command
        self.fps = fps
        self.process = None

        ts = dt.datetime.now().strftime('%F-%H-%M-%S')
        os.makedirs(CAMLOGDIR, exist_ok=True)
        stem = re.sub('[^A-Za-z0-9]', '_', self.device) or 'unconfigured'
        self.stdout_filename = os.path.join(
            CAMLOGDIR, '{}.out.{}.log'.format(stem, ts))
        self.stderr_filename = os.path.join(
            CAMLOGDIR, '{}.err.{}.log'.format(stem, ts))

    @classmethod
    def from_config(cls, cam: CameraConfig):
        return cls(cam.id, cam.name, cam.device, cam.command, cam.fps)

    def run_command(self):
        """Start the helper command, if one is configured and not running."""
        if self.process is not None or not self.command.strip():
            return
        try:
            self.process = sp.Popen(self.command.split(),
                                    stdout=open(self.stdout_filename, 'w'),
                                    stderr=open(self.stderr_filename, 'w'))
        except (OSError, ValueError) as e:
            raise CameraError(
                "Cannot start helper command for camera '{}': {}\n"
                "Command: {}".format(self.camera_id, e, self.command),
                hint="Check that the executable exists and the command line "
                "in facebin.toml is valid.") from e
        log.info("Started helper command for %s (pid %s): %s",
                 self.camera_id, self.process.pid, self.command)

    def kill_command(self):
        """Stop the helper command if it is running."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except sp.TimeoutExpired:
            log.warning("Helper for %s did not terminate; killing it.",
                        self.camera_id)
            self.process.kill()
            self.process.wait()
        self.process = None

    def update_name(self, name):
        self.name = name

    def update_device(self, device):
        self.device = device

    def update_command(self, command):
        self.command = command


def get_camera_controllers(config=None):
    """Build controllers for every camera in the configuration.

    Returns a dict keyed by camera id.
    """
    if config is None:
        config = load_config()
    cams = {}
    for cam_cfg in config.cameras:
        cam = CameraController.from_config(cam_cfg)
        cams[cam.camera_id] = cam
    if not cams:
        log.warning(
            "No cameras defined in %s; the server will start without "
            "camera readers.", config.source or "the default configuration")
    return cams


def save_camera_config(cam, config=None, path=None):
    """Persist a camera's settings back into the TOML configuration file.

    Cameras without a device are placeholders (e.g. empty slots in the
    camera dialog) and are not saved.
    """
    if not cam.device:
        log.debug("Not saving camera '%s': it has no device.", cam.camera_id)
        return
    if config is None:
        config = load_config(path)
    target = path or config.source
    if target is None:
        raise CameraError(
            "Cannot save camera '{}': no configuration file is in use."
            .format(cam.camera_id),
            hint="Create one with `facebin init-config` first.")
    updated = CameraConfig(id=cam.camera_id,
                           name=cam.name,
                           device=cam.device,
                           command=cam.command,
                           fps=getattr(cam, "fps", 25))
    for i, existing in enumerate(config.cameras):
        if existing.id == cam.camera_id:
            config.cameras[i] = updated
            break
    else:
        config.cameras.append(updated)
    save_config(config, target)
