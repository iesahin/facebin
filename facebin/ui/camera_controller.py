"""Qt-facing wrapper around the server-side camera controller."""

from facebin.ui.qt_compat import qtc, Signal, Slot

import facebin.server.camera_controller as server_cc

Property = qtc.Property


class CameraController(qtc.QObject):
    def __init__(self, camera_id, name, device, command):
        super().__init__()
        self.cc = server_cc.CameraController(camera_id, name, device, command)

    @Slot()
    def run_command(self):
        self.cc.run_command()

    @Slot()
    def kill_command(self):
        self.cc.kill_command()

    def stdout_r(self):
        try:
            with open(self.cc.stdout_filename) as f:
                return f.read()
        except OSError:
            return ""

    def stderr_r(self):
        try:
            with open(self.cc.stderr_filename) as f:
                return f.read()
        except OSError:
            return ""

    stdout = Property(str, stdout_r, None)
    stderr = Property(str, stderr_r, None)

    @Slot(str)
    def update_name(self, name):
        self.cc.name = name

    @Slot(str)
    def update_device(self, device):
        self.cc.device = device

    @Slot(str)
    def update_command(self, command):
        self.cc.command = command


get_camera_controllers = server_cc.get_camera_controllers
save_camera_config = server_cc.save_camera_config
