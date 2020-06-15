import sys

from PySide2 import QtCore as qtc
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2.QtCore import Signal, Slot

import database_api as db
import camera_controller as cc


class CameraConfigurationDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super(CameraConfigurationDialog, self).__init__(parent)

        self.cams = cc.get_camera_controllers()

        print(self.cams)

        for k in ['camera1', 'camera2', 'camera3', 'camera4']:
            if k not in self.cams:
                self.cams[k] = cc.CameraController(k, k, "", "")

        self.cam1_name_edit = qtw.QLineEdit(self.cams['camera1'].name)
        self.cam1_name_edit.textChanged.connect(
            self.cams['camera1'].update_name)
        self.cam2_name_edit = qtw.QLineEdit(self.cams['camera2'].name)
        self.cam2_name_edit.textChanged.connect(
            self.cams['camera2'].update_name)
        self.cam3_name_edit = qtw.QLineEdit(self.cams['camera3'].name)
        self.cam3_name_edit.textChanged.connect(
            self.cams['camera3'].update_name)
        self.cam4_name_edit = qtw.QLineEdit(self.cams['camera4'].name)
        self.cam3_name_edit.textChanged.connect(
            self.cams['camera3'].update_name)

        self.cam1_device_edit = qtw.QLineEdit(self.cams['camera1'].device)
        self.cam1_device_edit.textChanged.connect(
            self.cams['camera1'].update_device)
        self.cam2_device_edit = qtw.QLineEdit(self.cams['camera2'].device)
        self.cam2_device_edit.textChanged.connect(
            self.cams['camera2'].update_device)
        self.cam3_device_edit = qtw.QLineEdit(self.cams['camera3'].device)
        self.cam3_device_edit.textChanged.connect(
            self.cams['camera3'].update_device)
        self.cam4_device_edit = qtw.QLineEdit(self.cams['camera4'].device)
        self.cam4_device_edit.textChanged.connect(
            self.cams['camera4'].update_device)

        self.cam1_command_edit = qtw.QLineEdit(self.cams['camera1'].command)
        self.cam2_command_edit = qtw.QLineEdit(self.cams['camera2'].command)
        self.cam3_command_edit = qtw.QLineEdit(self.cams['camera3'].command)
        self.cam4_command_edit = qtw.QLineEdit(self.cams['camera4'].command)

        self.cam1_command_edit.textChanged.connect(
            self.cams['camera1'].update_command)
        self.cam2_command_edit.textChanged.connect(
            self.cams['camera2'].update_command)
        self.cam3_command_edit.textChanged.connect(
            self.cams['camera3'].update_command)
        self.cam4_command_edit.textChanged.connect(
            self.cams['camera4'].update_command)

        self.cam1_command_run_button = qtw.QPushButton("Run")
        self.cam2_command_run_button = qtw.QPushButton("Run")
        self.cam3_command_run_button = qtw.QPushButton("Run")
        self.cam4_command_run_button = qtw.QPushButton("Run")

        self.cam1_command_run_button.clicked.connect(
            self.cams['camera1'].run_command)
        self.cam2_command_run_button.clicked.connect(
            self.cams['camera2'].run_command)
        self.cam3_command_run_button.clicked.connect(
            self.cams['camera3'].run_command)
        self.cam4_command_run_button.clicked.connect(
            self.cams['camera4'].run_command)

        self.cam1_command_kill_button = qtw.QPushButton("Kill")
        self.cam2_command_kill_button = qtw.QPushButton("Kill")
        self.cam3_command_kill_button = qtw.QPushButton("Kill")
        self.cam4_command_kill_button = qtw.QPushButton("Kill")

        self.cam1_command_kill_button.clicked.connect(
            self.cams['camera1'].kill_command)
        self.cam2_command_kill_button.clicked.connect(
            self.cams['camera2'].kill_command)
        self.cam3_command_kill_button.clicked.connect(
            self.cams['camera3'].kill_command)
        self.cam4_command_kill_button.clicked.connect(
            self.cams['camera4'].kill_command)

        self.cam1_command_stdout_button = qtw.QPushButton("Stdout")
        self.cam2_command_stdout_button = qtw.QPushButton("Stdout")
        self.cam3_command_stdout_button = qtw.QPushButton("Stdout")
        self.cam4_command_stdout_button = qtw.QPushButton("Stdout")

        self.cam1_command_stderr_button = qtw.QPushButton("Stderr")
        self.cam2_command_stderr_button = qtw.QPushButton("Stderr")
        self.cam3_command_stderr_button = qtw.QPushButton("Stderr")
        self.cam4_command_stderr_button = qtw.QPushButton("Stderr")

        layout_g1 = qtw.QGroupBox('Camera 1')
        layout_c1 = qtw.QFormLayout()
        layout_c1.addRow('Name', self.cam1_name_edit)
        layout_c1.addRow('Device', self.cam1_device_edit)
        layout_c1.addRow('Command', self.cam1_command_edit)
        button_layout_c1 = qtw.QHBoxLayout()
        button_layout_c1.addWidget(self.cam1_command_run_button)
        button_layout_c1.addWidget(self.cam1_command_kill_button)
        button_layout_c1.addWidget(self.cam1_command_stdout_button)
        button_layout_c1.addWidget(self.cam1_command_stderr_button)
        layout_c1.addRow(button_layout_c1)
        layout_g1.setLayout(layout_c1)

        layout_g2 = qtw.QGroupBox('Camera 2')
        layout_c2 = qtw.QFormLayout()
        layout_c2.addRow('Name', self.cam2_name_edit)
        layout_c2.addRow('Device', self.cam2_device_edit)
        layout_c2.addRow('Command', self.cam2_command_edit)
        button_layout_c2 = qtw.QHBoxLayout()
        button_layout_c2.addWidget(self.cam2_command_run_button)
        button_layout_c2.addWidget(self.cam2_command_kill_button)
        button_layout_c2.addWidget(self.cam2_command_stdout_button)
        button_layout_c2.addWidget(self.cam2_command_stderr_button)
        layout_c2.addRow(button_layout_c2)
        layout_g2.setLayout(layout_c2)

        layout_g3 = qtw.QGroupBox('Camera 3')
        layout_c3 = qtw.QFormLayout()
        layout_c3.addRow('Name', self.cam3_name_edit)
        layout_c3.addRow('Device', self.cam3_device_edit)
        layout_c3.addRow('Command', self.cam3_command_edit)
        button_layout_c3 = qtw.QHBoxLayout()
        button_layout_c3.addWidget(self.cam3_command_run_button)
        button_layout_c3.addWidget(self.cam3_command_kill_button)
        button_layout_c3.addWidget(self.cam3_command_stdout_button)
        button_layout_c3.addWidget(self.cam3_command_stderr_button)
        layout_c3.addRow(button_layout_c3)
        layout_g3.setLayout(layout_c3)

        layout_g4 = qtw.QGroupBox('Camera 4')
        layout_c4 = qtw.QFormLayout()
        layout_c4.addRow('Name', self.cam4_name_edit)
        layout_c4.addRow('Device', self.cam4_device_edit)
        layout_c4.addRow('Command', self.cam4_command_edit)
        button_layout_c4 = qtw.QHBoxLayout()
        button_layout_c4.addWidget(self.cam4_command_run_button)
        button_layout_c4.addWidget(self.cam4_command_kill_button)
        button_layout_c4.addWidget(self.cam4_command_stdout_button)
        button_layout_c4.addWidget(self.cam4_command_stderr_button)
        layout_c4.addRow(button_layout_c4)
        layout_g4.setLayout(layout_c4)

        self.cancel_button = qtw.QPushButton('Cancel')
        self.ok_button = qtw.QPushButton('OK')
        self.apply_button = qtw.QPushButton('Apply')
        dialog_button_layout = qtw.QHBoxLayout()
        dialog_button_layout.addWidget(self.ok_button)
        dialog_button_layout.addWidget(self.apply_button)
        dialog_button_layout.addWidget(self.cancel_button)

        layout_main = qtw.QVBoxLayout()
        layout_main.addWidget(layout_g1)
        layout_main.addWidget(layout_g2)
        layout_main.addWidget(layout_g3)
        layout_main.addWidget(layout_g4)
        layout_main.addLayout(dialog_button_layout)

        self.setLayout(layout_main)

        self.ok_button.clicked.connect(self.ok_clicked)
        self.apply_button.clicked.connect(self.accept_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked)

    @Slot()
    def ok_clicked(self):
        self.record_camera_data()
        self.accept()

    @Slot()
    def accept_clicked(self):
        self.record_camera_data()

    @Slot()
    def cancel_clicked(self):
        self.reject()

    def record_camera_data(self):

        for k in self.cams:
            cc.save_camera_config(self.cams[k])


def main():
    app = qtw.QApplication(sys.argv)
    ccd = CameraConfigurationDialog()
    ccd.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
