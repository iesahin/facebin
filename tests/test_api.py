"""Tests for the HTTP API and mobile web app (FastAPI TestClient).

Run against fakeredis and a temporary SQLite database; no network, camera,
or ML dependencies are needed.
"""

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import facebin.server.database_api as db
import facebin.server.redis_queue_utils as rqu
from facebin.api.server import STREAM_BOUNDARY, create_app
from facebin.config import CameraConfig


@pytest.fixture
def client(default_config, temp_db, fake_redis, monkeypatch):
    """A TestClient over an app wired to fakeredis and a temp database."""
    default_config.cameras = [
        CameraConfig(id="camera1", name="Entrance", device="/dev/video0")
    ]
    # create_app calls rqu.configure/db.configure, which would reconnect to
    # real services; pin them to the test doubles instead.
    monkeypatch.setattr(rqu, "configure", lambda *_: None)
    monkeypatch.setattr(db, "configure", lambda *_: None)
    app = create_app(default_config)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth(client):
    """Login as the default admin; returns auth headers."""
    response = client.post("/api/login",
                           json={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    token = response.json()["token"]
    return {"headers": {"Authorization": "Bearer " + token},
            "token": token}


def test_login_wrong_password(client):
    response = client.post("/api/login",
                           json={"username": "admin", "password": "nope"})
    assert response.status_code == 401
    assert "password" in response.json()["detail"]


def test_endpoints_require_auth(client):
    for path in ("/api/me", "/api/cameras", "/api/history", "/api/persons",
                 "/api/status", "/api/stream/camera1"):
        response = client.get(path)
        assert response.status_code == 401, path
        assert "Bearer" in response.json()["detail"]


def test_me_and_logout(client, auth):
    response = client.get("/api/me", **{"headers": auth["headers"]})
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin"
    assert "training" in body["permissions"]

    assert client.post("/api/logout",
                       headers=auth["headers"]).status_code == 200
    # The token is dead now.
    assert client.get("/api/me",
                      headers=auth["headers"]).status_code == 401


def test_token_accepted_as_query_parameter(client, auth):
    response = client.get("/api/me?token=" + auth["token"])
    assert response.status_code == 200


def test_cameras(client, auth):
    response = client.get("/api/cameras", headers=auth["headers"])
    assert response.status_code == 200
    cams = response.json()
    assert cams == [{"id": "camera1", "name": "Entrance", "fps": 25,
                     "stream_url": "/api/stream/camera1"}]


def test_stream_unknown_camera(client, auth):
    response = client.get("/api/stream/nope", headers=auth["headers"])
    assert response.status_code == 404
    assert "camera1" in response.json()["detail"]


def test_stream_serves_mjpeg_frames(client, auth, fake_redis):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8)
    key = rqu.add_frame(image, 1.0, None, "camera1")
    fake_redis.zadd(rqu.RECOGNIZER_QUEUE("camera1"), {key: 1.0})

    response = client.get("/api/stream/camera1?frames=1",
                          headers=auth["headers"])
    assert response.status_code == 200
    assert STREAM_BOUNDARY in response.headers["content-type"]
    body = response.content
    assert body.startswith(b"--" + STREAM_BOUNDARY.encode())
    assert b"Content-Type: image/jpeg" in body
    assert b"\xff\xd8" in body  # JPEG magic

    # The frame moved from the recognizer queue to the history queue, the
    # same hand-off the desktop GUI performs.
    assert rqu.queue_length(rqu.RECOGNIZER_QUEUE("camera1")) == 0
    assert rqu.queue_length(rqu.HISTORY_QUEUE) == 1


def test_status(client, auth, fake_redis):
    response = client.get("/api/status", headers=auth["headers"])
    assert response.status_code == 200
    body = response.json()
    assert "recognizer:camera1" in body["queues"]
    assert body["queues"]["camera"] == 0


def test_person_crud_and_history(client, auth, tmp_path):
    created = client.post(
        "/api/persons",
        headers=auth["headers"],
        json={"name": "Ada", "title": "Dr.", "notes": ""})
    assert created.status_code == 201
    pid = created.json()["id"]

    listed = client.get("/api/persons", headers=auth["headers"]).json()
    assert listed[0]["name"] == "Ada"
    assert listed[0]["face_count"] == 0
    assert listed[0]["image_url"] is None

    # Record an appearance and a face snapshot on disk.
    face_file = tmp_path / "face.png"
    face_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    db.record_history_data(1, pid, time.time(), None, str(face_file), None)

    detail = client.get("/api/persons/{}".format(pid),
                        headers=auth["headers"]).json()
    assert detail["name"] == "Ada"
    assert len(detail["history"]) == 1

    history = client.get("/api/history", headers=auth["headers"]).json()
    assert history[0]["person_name"] == "Ada"
    assert history[0]["known"] is True

    image = client.get(history[0]["face_image_url"],
                       headers=auth["headers"])
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")


def test_person_needs_name(client, auth):
    response = client.post("/api/persons", headers=auth["headers"],
                           json={"name": "   "})
    assert response.status_code == 422


def test_person_not_found(client, auth):
    response = client.get("/api/persons/999", headers=auth["headers"])
    assert response.status_code == 404


def test_history_unknown_only(client, auth):
    pid = db.insert_person("", "Known", "")
    db.record_history_data(1, pid, time.time(), None, None, None)
    db.record_history_data(1, None, time.time(), None, None, None)

    everyone = client.get("/api/history", headers=auth["headers"]).json()
    assert len(everyone) == 2
    unknown = client.get("/api/history?unknown_only=true",
                         headers=auth["headers"]).json()
    assert len(unknown) == 1
    assert unknown[0]["known"] is False


def test_history_snapshot_missing_from_disk(client, auth):
    pid = db.insert_person("", "Gone", "")
    row = db.record_history_data(1, pid, time.time(), "/no/cam.png",
                                 "/no/face.png", None)
    response = client.get("/api/history/{}/face".format(row),
                          headers=auth["headers"])
    assert response.status_code == 404
    assert "not on disk" in response.json()["detail"]


def test_webapp_shell_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Facebin</title>" in response.text
    for asset in ("app.js", "style.css", "manifest.webmanifest", "sw.js",
                  "icons/icon-192.png"):
        assert client.get("/" + asset).status_code == 200, asset


def test_session_expiry(default_config, temp_db, fake_redis, monkeypatch):
    monkeypatch.setattr(rqu, "configure", lambda *_: None)
    monkeypatch.setattr(db, "configure", lambda *_: None)
    default_config.api.session_ttl = 1
    app = create_app(default_config)
    with TestClient(app) as test_client:
        token = test_client.post(
            "/api/login",
            json={"username": "admin", "password": "admin"}).json()["token"]
        headers = {"Authorization": "Bearer " + token}
        assert test_client.get("/api/me", headers=headers).status_code == 200
        time.sleep(1.2)
        assert test_client.get("/api/me", headers=headers).status_code == 401
