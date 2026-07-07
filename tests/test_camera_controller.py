"""Tests for the camera controller and its TOML persistence."""

import os

import pytest

import facebin.server.camera_controller as cc
from facebin.config import (CameraConfig, Config, load_config, parse_config,
                            save_config)
from facebin.errors import CameraError


@pytest.fixture(autouse=True)
def camlogdir(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CAMLOGDIR", str(tmp_path / "cam-logs"))


def make_config(tmp_path):
    config = parse_config(
        '[[camera]]\nid = "cam1"\nname = "One"\ndevice = "/dev/video0"\n'
        '[[camera]]\nid = "cam2"\nname = "Two"\ndevice = "rtsp://cam/live"\n',
        source=str(tmp_path / "facebin.toml"))
    return config


def test_controllers_from_config(tmp_path):
    cams = cc.get_camera_controllers(make_config(tmp_path))
    assert set(cams) == {"cam1", "cam2"}
    assert cams["cam1"].device == "/dev/video0"
    assert cams["cam2"].name == "Two"


def test_empty_config_gives_no_controllers():
    assert cc.get_camera_controllers(Config()) == {}


def test_camera_id_required():
    with pytest.raises(CameraError):
        cc.CameraController("", "name", "/dev/video0", "")


def test_run_command_missing_executable():
    cam = cc.CameraController("cam1", "n", "/dev/video0",
                              "/no/such/binary --flag")
    with pytest.raises(CameraError) as e:
        cam.run_command()
    assert "cam1" in str(e.value)
    assert "/no/such/binary" in str(e.value)


def test_run_and_kill_command():
    cam = cc.CameraController("cam1", "n", "/dev/video0", "sleep 60")
    cam.run_command()
    assert cam.process is not None
    assert cam.process.poll() is None
    cam.kill_command()
    assert cam.process is None


def test_empty_command_is_noop():
    cam = cc.CameraController("cam1", "n", "/dev/video0", "  ")
    cam.run_command()
    assert cam.process is None
    cam.kill_command()  # must not raise


def test_save_camera_config_updates_existing(tmp_path):
    config = make_config(tmp_path)
    path = tmp_path / "facebin.toml"
    save_config(config, str(path))

    cam = cc.CameraController("cam1", "Renamed", "/dev/video7", "")
    cc.save_camera_config(cam, path=str(path))

    reloaded = load_config(str(path))
    assert reloaded.camera_by_id("cam1").device == "/dev/video7"
    assert reloaded.camera_by_id("cam1").name == "Renamed"
    assert reloaded.camera_by_id("cam2").device == "rtsp://cam/live"


def test_save_camera_config_appends_new(tmp_path):
    config = make_config(tmp_path)
    path = tmp_path / "facebin.toml"
    save_config(config, str(path))

    cam = cc.CameraController("cam3", "Third", "/dev/video2", "")
    cc.save_camera_config(cam, path=str(path))
    reloaded = load_config(str(path))
    assert len(reloaded.cameras) == 3
    assert reloaded.camera_by_id("cam3").name == "Third"


def test_save_camera_config_skips_placeholders(tmp_path):
    config = make_config(tmp_path)
    path = tmp_path / "facebin.toml"
    save_config(config, str(path))

    placeholder = cc.CameraController("cam9", "empty slot", "", "")
    cc.save_camera_config(placeholder, path=str(path))
    reloaded = load_config(str(path))
    assert len(reloaded.cameras) == 2
