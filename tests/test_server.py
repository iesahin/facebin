"""Tests for the FacebinServer process supervisor.

Worker entry points are replaced with lightweight dummies, so these tests
exercise process spawning, health checking, restarting, and shutdown
without cameras, TensorFlow, or Redis.
"""

import os
import sys
import time

import pytest

import facebin.server.redis_queue_utils as rqu
from facebin.config import CameraConfig, Config
from facebin.errors import FacebinError
from facebin.server import FacebinServer, RepeatTimer


def sleepy_worker(*args, **kwargs):
    time.sleep(60)


def dying_worker(*args, **kwargs):
    sys.exit(3)


def recording_camera_worker(camera_id, device, fps, config):
    out_dir = os.environ["FACEBIN_TEST_OUT"]
    with open(os.path.join(out_dir, "camera-args.txt"), "w") as f:
        f.write("{}|{}|{}".format(camera_id, device, fps))
    time.sleep(60)


def make_config(n_cameras=1, recognizers_per_camera=1, history_recorders=1):
    config = Config()
    config.server.flush_redis_on_start = False
    config.server.recognizers_per_camera = recognizers_per_camera
    config.server.history_recorders = history_recorders
    config.cameras = [
        CameraConfig(id="camera{}".format(i + 1),
                     name="Test camera {}".format(i + 1),
                     device="/dev/video{}".format(i),
                     fps=10 + i) for i in range(n_cameras)
    ]
    return config


def make_server(config, **targets):
    targets.setdefault("camera_reader_target", sleepy_worker)
    targets.setdefault("recognizer_target", sleepy_worker)
    targets.setdefault("history_target", sleepy_worker)
    return FacebinServer(config=config, **targets)


def wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def server(request):
    servers = []

    def factory(config, **targets):
        s = make_server(config, **targets)
        servers.append(s)
        return s

    yield factory
    for s in servers:
        s.stop()


def test_start_spawns_expected_processes(server, fake_redis):
    config = make_config(n_cameras=2, recognizers_per_camera=2,
                         history_recorders=1)
    s = server(config)
    rqu.set_redis(fake_redis)
    s.start()

    assert len(s.camera_reader_processes) == 2
    assert len(s.recognizer_processes) == 4
    assert len(s.history_processes) == 1
    for group in (s.camera_reader_processes, s.recognizer_processes,
                  s.history_processes):
        for proc in group.values():
            assert proc.is_alive()


def test_start_twice_raises(server, fake_redis):
    s = server(make_config())
    rqu.set_redis(fake_redis)
    s.start()
    with pytest.raises(FacebinError):
        s.start()


def test_stop_terminates_everything(server, fake_redis):
    s = server(make_config(n_cameras=1))
    rqu.set_redis(fake_redis)
    s.start()
    procs = (list(s.camera_reader_processes.values()) +
             list(s.recognizer_processes.values()) +
             list(s.history_processes.values()))
    s.stop()
    for proc in procs:
        assert not proc.is_alive()
    assert s.camera_reader_processes == {}
    # stop() is idempotent
    s.stop()


def test_camera_reader_receives_camera_config(server, fake_redis, tmp_path,
                                              monkeypatch):
    monkeypatch.setenv("FACEBIN_TEST_OUT", str(tmp_path))
    config = make_config(n_cameras=1, recognizers_per_camera=0,
                         history_recorders=0)
    s = server(config, camera_reader_target=recording_camera_worker)
    rqu.set_redis(fake_redis)
    s.start()

    out_file = tmp_path / "camera-args.txt"
    assert wait_until(out_file.exists)
    assert out_file.read_text() == "camera1|/dev/video0|10"


def test_dead_camera_reader_is_restarted(server, fake_redis):
    config = make_config(n_cameras=1, recognizers_per_camera=0,
                         history_recorders=0)
    s = server(config, camera_reader_target=dying_worker)
    rqu.set_redis(fake_redis)
    s.start()

    first = s.camera_reader_processes["camera1"]
    assert wait_until(lambda: not first.is_alive())
    s.check_camera_processes()
    second = s.camera_reader_processes["camera1"]
    assert second is not first


def test_dead_recognizer_is_restarted(server, fake_redis):
    config = make_config(n_cameras=1, recognizers_per_camera=1,
                         history_recorders=0)
    s = server(config, recognizer_target=dying_worker)
    rqu.set_redis(fake_redis)
    s.start()

    first = s.recognizer_processes[0]
    assert wait_until(lambda: not first.is_alive())
    s.check_recognizer_processes()
    assert s.recognizer_processes[0] is not first


def test_dead_history_recorder_is_restarted(server, fake_redis):
    config = make_config(n_cameras=1, history_recorders=1)
    s = server(config, history_target=dying_worker)
    rqu.set_redis(fake_redis)
    s.start()

    first = s.history_processes[0]
    assert wait_until(lambda: not first.is_alive())
    s.check_history_processes()
    assert s.history_processes[0] is not first


def test_force_restart_replaces_live_processes(server, fake_redis):
    s = server(make_config(n_cameras=1))
    rqu.set_redis(fake_redis)
    s.start()
    first = s.camera_reader_processes["camera1"]
    assert first.is_alive()
    s.check_camera_processes(force_restart=True)
    second = s.camera_reader_processes["camera1"]
    assert second is not first
    assert second.is_alive()
    assert not first.is_alive()


def test_flush_redis_on_start(server, fake_redis):
    config = make_config(n_cameras=0, recognizers_per_camera=0,
                         history_recorders=0)
    config.server.flush_redis_on_start = True
    s = server(config)
    rqu.set_redis(fake_redis)
    fake_redis.set("leftover", "value")
    s.start()
    assert not fake_redis.exists("leftover")


def test_repeat_timer_fires_repeatedly():
    calls = []
    timer = RepeatTimer(0.02, lambda: calls.append(1))
    timer.daemon = True
    timer.start()
    try:
        assert wait_until(lambda: len(calls) >= 3, timeout=5)
    finally:
        timer.cancel()
