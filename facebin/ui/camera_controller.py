from PySide2 import QtCore as qtc
from PySide2.QtCore import Signal, Slot, Property

import facebin.server.camera_controller as server_cc


class CameraController(qtc.QObject):
    def __init__(self, camera_id, name, device, command):
        self.cc = server_cc.CameraController(camera_id, name, device, command)

    @Slot()
    def run_command(self):
        self.cc.run_command()

    @Slot()
    def kill_command(self):
        self.cc.kill_command()

    def stdout_r(self):
        return self.cc.stdout_file.read()

    def stderr_r(self):
        return self.cc.stderr_file.read()

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
