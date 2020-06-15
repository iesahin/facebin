from . import redis_queue_utils as rqu
from . import camera_controller as cc
from . import face_recognition_v6 as fr6
from .utils import *
import multiprocessing as mp
from threading import Timer
import configparser as cp
import redis


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


log = init_logging()


class FacebinServer:

    def __init__(self, num_camera_reader=True, num_recognizer=True, num_history=True, health_check_interval=1, camera_config_file="../config/camera_config.ini"):
        self._num_camera_reader = num_camera_reader
        self._num_recognizer = num_recognizer
        self._num_history = num_history
        self._health_check_interval = health_check_interval
        self._camera_config_file = camera_config_file

        self.camera_controllers = cc.get_camera_controllers(
            config_filename=camera_config_file)

    def init_worker_processes(self):
        # Init Camera Readers

        R.flushall()

        self.camera_reader_processes = {}
        for k, cc in self.camera_controllers.items():
            self.camera_reader_processes[k] = mp.Process(
                target=camera_reader.reader_loop,
                args=(cc.camera_id, cc.device))
            self.camera_reader_processes[k].start()

        self.recognizer_processes = {}
        for k in range(
                len(self.camera_controllers) * self.recognizers_per_camera):
            self.recognizer_processes[k] = mp.Process(
                target=fr6.face_recognition_loop)
            self.recognizer_processes[k].start()
            log.debug("self.recognizer_processes:", self.recognizer_processes)

        self.history_processes = {}
        for k, cc in self.camera_controllers.items():
            self.history_processes[k] = mp.Process(
                target=history_recorder.record_loop)
            self.history_processes[k].start()

        self.heartbeat_timer = RepeatTimer(
            self._health_check_interval, self.check_process_health)

    def check_process_health(self):
        self.check_camera_processes()
        self.check_history_processes()
        self.check_recognizer_processes()

    def check_camera_processes(self, force_restart=False):
        for cpk, cp in self.camera_reader_processes.items():
            if not cp.is_alive():
                log.warning(
                    "Camera Process %s with PID %s is dead. Restarting.", cpk,
                    cp.pid)
                cc = self.camera_controllers[cpk]
                self.camera_reader_processes[cpk] = mp.Process(
                    target=camera_reader.reader_loop,
                    args=(cc.camera_id, cc.device))
                self.camera_reader_processes[cpk].start()
            elif force_restart:
                log.warning(
                    "Force Restarting Camera Process %s with PID %s.", cpk,
                    cp.pid)
                cp.terminate()
                cp.join()
                cc = self.camera_controllers[cpk]
                self.camera_reader_processes[cpk] = mp.Process(
                    target=camera_reader.reader_loop,
                    args=(cc.camera_id, cc.device))
                self.camera_reader_processes[cpk].start()
            else:
                log.debug("Camera Process %s with PID %s is alive", cpk,
                          cp.pid)

        log.debug("Camera Queue Length: %s",
                  rqu.queue_length(rqu.CAMERA_QUEUE))
        for cpk in self.camera_reader_processes:
            log.debug("Recognizer(%s) Queue Length: %s", cpk,
                      rqu.queue_length(rqu.RECOGNIZER_QUEUE(cpk)))

    def check_recognizer_processes(self, force_restart=False):

        for rk, rp in self.recognizer_processes.items():
            if not rp.is_alive():
                log.warning(
                    "Recognizer Process %s with PID %s is dead. Restarting.",
                    rk, rp.pid)
                self.recognizer_processes[rk] = mp.Process(
                    # target=fr.face_recognition_loop)
                    target=fr6.face_recognition_loop)
                self.recognizer_processes[rk].start()
            elif force_restart:
                log.warning(
                    "Force Restarting Recognizer Process %s with PID %s",
                    rk, rp.pid)
                rp.terminate()
                rp.join()
                self.recognizer_processes[rk] = mp.Process(
                    # target=fr.face_recognition_loop)
                    target=fr6.face_recognition_loop)
                self.recognizer_processes[rk].start()
            else:
                log.debug("Recognizer Process %s with PID %s is alive", rk,
                          rp.pid)

    def check_history_processes(self, force_restart=False):
        for hk, hp in self.history_processes.items():
            if not hp.is_alive():
                log.warning(
                    "History Process %s with PID %s is dead. Restarting.", hk, hp.pid)
                self.history_processes[hk] = mp.Process(
                    target=history_recorder.record_loop)
                self.history_processes[hk].start()
            elif force_restart:
                log.warning(
                    "Force Restarting History Process %s with PID %s", hk,
                    hp.pid)
                hp.terminate()
                hp.join()
                self.history_processes[hk] = mp.Process(
                    target=history_recorder.record_loop)
                self.history_processes[hk].start()
            else:
                log.debug("History Process %s with PID %s is alive", hk,
                          hp.pid)

        log.debug("History Queue Length: %s",
                  rqu.queue_length(rqu.HISTORY_QUEUE))
        log.debug("History Recording Queue Length: %s",
                  rqu.queue_length(rqu.HISTORY_RECORDING_QUEUE))
