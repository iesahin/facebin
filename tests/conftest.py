"""Shared fixtures for the Facebin test suite.

The tests exercise the headless core (configuration, database, queues,
process supervision, CLI); they do not require TensorFlow, Qt, a camera,
or a real Redis server.
"""

import sys

import fakeredis
import pytest

import facebin.server.database_api as db
import facebin.server.redis_queue_utils as rqu
from facebin.config import Config


@pytest.fixture
def fake_redis():
    """Inject a fresh fakeredis client into the queue module."""
    client = fakeredis.FakeRedis()
    rqu.set_redis(client)
    yield client
    rqu.set_redis(None)


@pytest.fixture
def temp_db(tmp_path):
    """Point the database module at a fresh temporary SQLite file."""
    path = tmp_path / "facebin-test.db"
    db.set_database_path(str(path))
    db.init_db()
    yield str(path)
    db.set_database_path(None)


@pytest.fixture
def default_config(tmp_path):
    """A default Config whose paths all live under tmp_path."""
    config = Config()
    config.database.path = str(tmp_path / "facebin.db")
    config.history.image_record_dir = str(tmp_path / "image-store")
    config.video.video_record_dir = str(tmp_path / "video-store")
    config.models.dir = str(tmp_path / "models")
    config.dataset.dir = str(tmp_path / "dataset")
    return config
