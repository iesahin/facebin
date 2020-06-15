import subprocess as sp
from . import database_api as db
import re
import os
import datetime as dt
import socket
import av

import configparser as cp

from facebin.server.utils import *
log = init_logging()

CAMLOGDIR = '/tmp/facebin-cam-logs/'


class CameraController:
    def __init__(self, camera_id, name, device, command):
        self.camera_id = camera_id
        self.name = name
        self.device = device
        self.command = command
        log.debug("self.device: %s", self.device)
        # self.container = av.open(self.device, 'r')
        # FASTER
        # self.container.streams.video[0].thread_type = 'AUTO'
        # SKIP NONKEY FRAMES
        # self.container.streams.video[0].codec_context.skip_frame = 'NONKEY'
        self.process = None
        ts = dt.datetime.now().strftime('%F-%H-%M-%S')
        if not os.path.isdir(CAMLOGDIR):
            os.mkdir(CAMLOGDIR)

        self.stdout_filename = os.path.join(
            CAMLOGDIR,
            re.sub('[^A-Za-z0-9]', '_', self.device) + '.out.' + ts + '.log')
        self.stdout_file = open(self.stdout_filename, 'w')
        self.stderr_filename = os.path.join(
            CAMLOGDIR,
            re.sub('[^A-Za-z0-9]', '_', self.device) + '.err.' + ts + '.log')
        self.stderr_file = open(self.stderr_filename, 'w')

    def run_command(self):
        log.debug("Running Command: %s", self.command)
        log.debug("self.process: %s", self.process)
        if self.process is None and self.command.strip() != "":
            self.process = sp.Popen(
                self.command.split(),
                stdout=self.stdout_file,
                stderr=self.stderr_file)
        log.debug("self.process: %s", self.process)

    def kill_command(self):
        if self.process is not None:
            self.process.terminate()
            if not self.process.poll():
                self.process.kill()
            self.process = None

    def stdout_r(self):
        return self.stdout_file.read()

    def stderr_r(self):
        return self.stderr_file.read()

    def update_name(self, name):
        self.name = name

    def update_device(self, device):
        self.device = device

    def update_command(self, command):
        self.command = command


def get_default_config_filename():
    hostname = socket.gethostname()
    config_filename = "camera-config-{}.ini".format(hostname)
    return config_filename


def get_camera_controllers(config_filename=None):

    if config_filename is None:
        config_filename = get_default_config_filename()

    config = cp.ConfigParser()

    log.debug(config_filename)
    if os.path.exists(config_filename):
        config.read(config_filename)
    else:
        config['camera1'] = {
            'id': 'camera1',
            'name': 'Default Camera',
            'device': '/dev/video0',
            'command': ''
        }
        with open(config_filename, 'w') as f:
            config.write(f)

    cams = {}
    for i in range(1, 10):
        cam_key = 'camera{}'.format(i)
        if config.has_section(cam_key):
            cam = CameraController(
                config[cam_key]['id'], config[cam_key]['name'],
                config[cam_key]['device'], config[cam_key]['command'])
            cams[cam.camera_id] = cam
    return cams


def save_camera_config(cam, config_filename=None):

    if config_filename is None:
        config_filename = get_default_config_filename()

    config = cp.ConfigParser(config_filename)

    config[cam.camera_id] = {
        'id': cam.camera_id,
        'name': cam.name,
        'device': cam.device,
        'command': cam.command
    }

    with open(config_filename, 'w') as f:
        config.write(f)
