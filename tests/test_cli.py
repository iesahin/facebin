"""Tests for the `facebin` command line interface."""

import os

import pytest

import facebin.server.database_api as db
import facebin.server.redis_queue_utils as rqu
from facebin.cli import build_parser, main
from facebin.config import RedisConfig, load_config


@pytest.fixture(autouse=True)
def reset_module_state():
    yield
    rqu.set_redis(None)
    rqu.configure(RedisConfig())
    db.set_database_path(None)


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    from facebin import __version__
    assert "facebin {}".format(__version__) in capsys.readouterr().out


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("run", "server", "gui", "init-config", "init-db",
                    "check"):
        assert command in out


def test_init_config(tmp_path, capsys):
    target = tmp_path / "facebin.toml"
    rc = main(["init-config", "--path", str(target)])
    assert rc == 0
    assert target.exists()
    config = load_config(str(target))
    assert config.redis.port == 6379


def test_init_config_refuses_overwrite(tmp_path, capsys):
    target = tmp_path / "facebin.toml"
    assert main(["init-config", "--path", str(target)]) == 0
    rc = main(["init-config", "--path", str(target)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Refusing to overwrite" in err
    assert "--force" in err
    assert main(["init-config", "--path", str(target), "--force"]) == 0


def test_init_db(tmp_path, capsys):
    config_path = tmp_path / "facebin.toml"
    db_path = tmp_path / "data" / "facebin.db"
    config_path.write_text('[database]\npath = "{}"\n'.format(db_path))
    rc = main(["init-db", "-c", str(config_path)])
    assert rc == 0
    assert db_path.exists()
    assert "Database ready" in capsys.readouterr().out


def test_missing_config_file_reports_error(capsys):
    rc = main(["check", "-c", "/does/not/exist.toml"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "does not exist" in err


def test_check_reports_problems(tmp_path, capsys):
    config_path = tmp_path / "facebin.toml"
    config_path.write_text(
        '[redis]\nhost = "localhost"\nport = 1\nautostart = false\n'
        '[database]\npath = "{db}"\n'
        '[models]\ndir = "{models}"\n'
        '[[camera]]\nid = "cam1"\ndevice = "/dev/video99"\n'.format(
            db=tmp_path / "facebin.db", models=tmp_path / "models"))
    rc = main(["check", "-c", str(config_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Problems found:" in out
    assert "Cannot connect to Redis" in out
    assert "/dev/video99" in out
    assert "Detection model" in out


def test_check_passes_with_fakeredis(tmp_path, capsys, fake_redis,
                                     monkeypatch):
    # Make every referenced path exist and Redis reachable.
    models = tmp_path / "models"
    models.mkdir()
    (models / "frozen_inference_graph_face.pb").write_bytes(b"")
    (models / "face_label_map.pbtxt").write_text("")
    db_path = tmp_path / "facebin.db"
    db_path.write_bytes(b"")
    config_path = tmp_path / "facebin.toml"
    config_path.write_text(
        '[database]\npath = "{db}"\n'
        '[models]\ndir = "{models}"\n'.format(db=db_path, models=models))

    # `check` calls rqu.configure, which resets the injected client; put
    # it back so the connectivity probe hits fakeredis.
    monkeypatch.setattr(rqu, "configure", lambda *_: None)
    rc = main(["check", "-c", str(config_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Everything looks good" in out


def test_default_command_is_run():
    parser = build_parser()
    args = parser.parse_args(["run", "--no-gui"])
    assert args.command == "run"
    assert args.no_gui is True
