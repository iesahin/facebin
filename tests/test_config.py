"""Tests for the TOML configuration loader and validator."""

import os

import pytest

from facebin.config import (DEFAULT_CONFIG_TOML, Config, dump_toml,
                            find_config_file, load_config, parse_config,
                            save_config, write_default_config)
from facebin.errors import ConfigError


def test_default_template_parses():
    config = parse_config(DEFAULT_CONFIG_TOML, source="<default>")
    assert config.redis.host == "localhost"
    assert config.redis.port == 6379
    assert config.redis.autostart is True
    assert config.server.recognizers_per_camera == 1
    assert len(config.cameras) == 1
    assert config.cameras[0].id == "camera1"


def test_defaults_when_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FACEBIN_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config()
    assert config.source is None
    assert config.redis.port == 6379
    assert config.cameras == []


def test_load_explicit_path(tmp_path):
    path = tmp_path / "my.toml"
    path.write_text('[redis]\nport = 7000\n')
    config = load_config(str(path))
    assert config.redis.port == 7000
    assert config.source == str(path)


def test_missing_explicit_path_raises():
    with pytest.raises(ConfigError) as e:
        load_config("/nonexistent/facebin.toml")
    assert "does not exist" in str(e.value)
    assert "init-config" in str(e.value)


def test_env_var_location(tmp_path, monkeypatch):
    path = tmp_path / "env.toml"
    path.write_text('[redis]\nhost = "envhost"\n')
    monkeypatch.setenv("FACEBIN_CONFIG", str(path))
    assert find_config_file() == str(path)
    assert load_config().redis.host == "envhost"


def test_env_var_pointing_nowhere_raises(monkeypatch):
    monkeypatch.setenv("FACEBIN_CONFIG", "/nope/nope.toml")
    with pytest.raises(ConfigError) as e:
        load_config()
    assert "FACEBIN_CONFIG" in str(e.value)


def test_invalid_toml_syntax():
    with pytest.raises(ConfigError) as e:
        parse_config("this is not toml [", source="bad.toml")
    assert "bad.toml" in str(e.value)
    assert "TOML" in str(e.value)


def test_unknown_section():
    with pytest.raises(ConfigError) as e:
        parse_config("[nonsense]\nx = 1\n", source="t.toml")
    assert "[nonsense]" in str(e.value)
    assert "Valid sections" in str(e.value)


def test_unknown_key():
    with pytest.raises(ConfigError) as e:
        parse_config("[redis]\nhosty = 'x'\n", source="t.toml")
    assert "hosty" in str(e.value)
    assert "Valid keys" in str(e.value)


def test_wrong_type():
    with pytest.raises(ConfigError) as e:
        parse_config('[redis]\nport = "not a number"\n', source="t.toml")
    assert "port" in str(e.value)
    assert "int" in str(e.value)


def test_port_out_of_range():
    with pytest.raises(ConfigError) as e:
        parse_config("[redis]\nport = 70000\n", source="t.toml")
    assert "out of range" in str(e.value)


@pytest.mark.parametrize("section,key,value", [
    ("server", "recognizers_per_camera", -1),
    ("server", "history_recorders", -2),
    ("server", "health_check_interval", 0),
    ("history", "image_record_period", 0),
    ("video", "video_seconds_per_file", -5),
])
def test_invalid_numeric_settings(section, key, value):
    text = "[{}]\n{} = {}\n".format(section, key, value)
    with pytest.raises(ConfigError) as e:
        parse_config(text, source="t.toml")
    assert key in str(e.value)


def test_camera_requires_id():
    with pytest.raises(ConfigError) as e:
        parse_config('[[camera]]\ndevice = "/dev/video0"\n', source="t.toml")
    assert "id" in str(e.value)


def test_camera_requires_device():
    with pytest.raises(ConfigError) as e:
        parse_config('[[camera]]\nid = "cam1"\n', source="t.toml")
    assert "device" in str(e.value)


def test_duplicate_camera_ids():
    text = ('[[camera]]\nid = "cam1"\ndevice = "/dev/video0"\n'
            '[[camera]]\nid = "cam1"\ndevice = "/dev/video1"\n')
    with pytest.raises(ConfigError) as e:
        parse_config(text, source="t.toml")
    assert "Duplicate" in str(e.value)


def test_camera_fps_must_be_positive():
    text = '[[camera]]\nid = "c"\ndevice = "/dev/video0"\nfps = 0\n'
    with pytest.raises(ConfigError) as e:
        parse_config(text, source="t.toml")
    assert "fps" in str(e.value)


def test_camera_by_id():
    config = parse_config(DEFAULT_CONFIG_TOML, source="<default>")
    assert config.camera_by_id("camera1").device == "/dev/video0"
    with pytest.raises(ConfigError):
        config.camera_by_id("nope")


def test_path_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = parse_config('[database]\npath = "~/data/f.db"\n', source="t")
    assert config.database.resolved_path() == str(tmp_path / "data" / "f.db")


def test_write_default_config(tmp_path):
    target = tmp_path / "sub" / "facebin.toml"
    written = write_default_config(str(target))
    assert os.path.exists(written)
    # And the written file round-trips.
    config = load_config(written)
    assert config.redis.port == 6379


def test_write_default_config_refuses_overwrite(tmp_path):
    target = tmp_path / "facebin.toml"
    write_default_config(str(target))
    with pytest.raises(ConfigError) as e:
        write_default_config(str(target))
    assert "Refusing to overwrite" in str(e.value)
    write_default_config(str(target), overwrite=True)  # --force works


def test_dump_toml_roundtrip(tmp_path):
    config = parse_config(DEFAULT_CONFIG_TOML, source="<default>")
    config.redis.port = 6390
    config.cameras[0].name = 'Cam "quoted" \\ name'
    reparsed = parse_config(dump_toml(config), source="<dump>")
    assert reparsed.redis.port == 6390
    assert reparsed.cameras[0].name == 'Cam "quoted" \\ name'
    assert reparsed.history.image_record_period == \
        config.history.image_record_period


def test_save_config(tmp_path):
    config = parse_config(DEFAULT_CONFIG_TOML, source="<default>")
    target = tmp_path / "saved.toml"
    save_config(config, str(target))
    assert config.source == str(target)
    assert load_config(str(target)).redis.port == 6379
