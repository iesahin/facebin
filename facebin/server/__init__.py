"""Facebin server: supervises the worker processes of the pipeline.

The pipeline consists of three kinds of worker processes connected by Redis
queues:

1. **Camera readers** (one per camera) decode frames and push them to the
   camera queue.
2. **Recognizers** detect and recognize faces in queued frames.
3. **History recorders** aggregate recognized faces into appearance records
   and write them to the SQLite database.

:class:`FacebinServer` starts these processes, checks their health
periodically, and restarts any that die.  Worker entry points are imported
lazily (and can be injected) so that the supervisor itself does not require
TensorFlow or camera libraries — this keeps it importable in tests and in
minimal installs.
"""

import multiprocessing as mp
from threading import Timer

from facebin.config import Config, load_config
from facebin.errors import FacebinError
from . import redis_queue_utils as rqu
from .utils import init_logging

log = init_logging()


class RepeatTimer(Timer):
    """A Timer that fires repeatedly every ``interval`` seconds."""

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def _default_camera_reader_target():
    from . import camera_reader
    return camera_reader.reader_loop


def _default_recognizer_target():
    from . import face_recognition_v6
    return face_recognition_v6.face_recognition_loop


def _default_history_target():
    from . import history_recorder
    return history_recorder.record_loop


class FacebinServer:
    """Starts and supervises all Facebin worker processes.

    Args:
        config: A loaded :class:`facebin.config.Config`.  When None the
            configuration is loaded from the default locations.
        camera_reader_target / recognizer_target / history_target:
            Worker entry points; overridable for testing.  Camera readers
            are called as ``target(camera_id, device, max_fps, config)``,
            the others as ``target(config)``.
    """

    def __init__(self,
                 config: Config = None,
                 camera_reader_target=None,
                 recognizer_target=None,
                 history_target=None):
        self.config = config if config is not None else load_config()
        self._camera_reader_target = camera_reader_target
        self._recognizer_target = recognizer_target
        self._history_target = history_target

        self.camera_reader_processes = {}
        self.recognizer_processes = {}
        self.history_processes = {}
        self.heartbeat_timer = None
        self._running = False

        rqu.configure(self.config.redis)

    # --- Worker target resolution (lazy so heavy imports stay optional) ---

    def camera_reader_target(self):
        if self._camera_reader_target is None:
            self._camera_reader_target = _default_camera_reader_target()
        return self._camera_reader_target

    def recognizer_target(self):
        if self._recognizer_target is None:
            self._recognizer_target = _default_recognizer_target()
        return self._recognizer_target

    def history_target(self):
        if self._history_target is None:
            self._history_target = _default_history_target()
        return self._history_target

    # --- Process spawning --------------------------------------------------

    def _spawn_camera_reader(self, cam_cfg):
        p = mp.Process(target=self.camera_reader_target(),
                       args=(cam_cfg.id, cam_cfg.device, cam_cfg.fps,
                             self.config),
                       name="facebin-camera-{}".format(cam_cfg.id),
                       daemon=True)
        p.start()
        log.info("Started camera reader for %s (pid %s)", cam_cfg.id, p.pid)
        return p

    def _spawn_recognizer(self, index):
        p = mp.Process(target=self.recognizer_target(),
                       args=(self.config, ),
                       name="facebin-recognizer-{}".format(index),
                       daemon=True)
        p.start()
        log.info("Started recognizer %s (pid %s)", index, p.pid)
        return p

    def _spawn_history_recorder(self, index):
        p = mp.Process(target=self.history_target(),
                       args=(self.config, ),
                       name="facebin-history-{}".format(index),
                       daemon=True)
        p.start()
        log.info("Started history recorder %s (pid %s)", index, p.pid)
        return p

    # --- Lifecycle ----------------------------------------------------------

    def start(self):
        """Start every worker process and the health-check timer."""
        if self._running:
            raise FacebinError("FacebinServer is already running.")

        if self.config.server.flush_redis_on_start:
            log.info("Flushing Redis queues before start.")
            rqu.get_redis().flushdb()

        for cam_cfg in self.config.cameras:
            self.camera_reader_processes[cam_cfg.id] = \
                self._spawn_camera_reader(cam_cfg)

        n_recognizers = (len(self.config.cameras) *
                         self.config.server.recognizers_per_camera)
        # Recognition must run even in camera-less setups (e.g. imported
        # video files) when explicitly configured.
        if not self.config.cameras:
            n_recognizers = self.config.server.recognizers_per_camera
        for k in range(n_recognizers):
            self.recognizer_processes[k] = self._spawn_recognizer(k)

        for k in range(self.config.server.history_recorders):
            self.history_processes[k] = self._spawn_history_recorder(k)

        self.heartbeat_timer = RepeatTimer(
            self.config.server.health_check_interval,
            self.check_process_health)
        self.heartbeat_timer.daemon = True
        self.heartbeat_timer.start()
        self._running = True
        log.info(
            "Facebin server started: %s camera reader(s), %s recognizer(s), "
            "%s history recorder(s).", len(self.camera_reader_processes),
            len(self.recognizer_processes), len(self.history_processes))

    def stop(self):
        """Stop the health checker and terminate every worker process."""
        if self.heartbeat_timer is not None:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None
        for group in (self.camera_reader_processes,
                      self.recognizer_processes, self.history_processes):
            for key, proc in group.items():
                if proc.is_alive():
                    log.info("Terminating %s (pid %s)", proc.name, proc.pid)
                    proc.terminate()
            for key, proc in group.items():
                proc.join(timeout=5)
                if proc.is_alive():
                    log.warning("%s (pid %s) did not stop; killing it.",
                                proc.name, proc.pid)
                    proc.kill()
                    proc.join()
            group.clear()
        self._running = False
        log.info("Facebin server stopped.")

    def join(self):
        """Block while workers run (the health checker restarts the dead)."""
        import time
        while self._running:
            time.sleep(1)

    # --- Health checks --------------------------------------------------------

    def check_process_health(self):
        try:
            self.check_camera_processes()
            self.check_history_processes()
            self.check_recognizer_processes()
        except Exception:
            log.exception("Health check failed")

    def check_camera_processes(self, force_restart=False):
        for cpk, proc in list(self.camera_reader_processes.items()):
            if not proc.is_alive() or force_restart:
                if proc.is_alive():
                    log.warning("Force restarting camera reader %s (pid %s).",
                                cpk, proc.pid)
                    proc.terminate()
                    proc.join()
                else:
                    log.warning(
                        "Camera reader %s (pid %s) died with exit code %s; "
                        "restarting it.", cpk, proc.pid, proc.exitcode)
                cam_cfg = self.config.camera_by_id(cpk)
                self.camera_reader_processes[cpk] = \
                    self._spawn_camera_reader(cam_cfg)

        log.debug("Camera queue length: %s",
                  rqu.queue_length(rqu.CAMERA_QUEUE))

    def check_recognizer_processes(self, force_restart=False):
        for rk, proc in list(self.recognizer_processes.items()):
            if not proc.is_alive() or force_restart:
                if proc.is_alive():
                    log.warning("Force restarting recognizer %s (pid %s).",
                                rk, proc.pid)
                    proc.terminate()
                    proc.join()
                else:
                    log.warning(
                        "Recognizer %s (pid %s) died with exit code %s; "
                        "restarting it.", rk, proc.pid, proc.exitcode)
                self.recognizer_processes[rk] = self._spawn_recognizer(rk)

    def check_history_processes(self, force_restart=False):
        for hk, proc in list(self.history_processes.items()):
            if not proc.is_alive() or force_restart:
                if proc.is_alive():
                    log.warning(
                        "Force restarting history recorder %s (pid %s).",
                        hk, proc.pid)
                    proc.terminate()
                    proc.join()
                else:
                    log.warning(
                        "History recorder %s (pid %s) died with exit code "
                        "%s; restarting it.", hk, proc.pid, proc.exitcode)
                self.history_processes[hk] = self._spawn_history_recorder(hk)

        log.debug("History queue length: %s",
                  rqu.queue_length(rqu.HISTORY_QUEUE))
