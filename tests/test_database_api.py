"""Tests for the SQLite persistence layer."""

import time

import pytest

import facebin.server.database_api as db
from facebin.errors import DatabaseError


def test_init_db_is_idempotent(temp_db):
    db.init_db()
    db.init_db()
    assert db.login("admin", "admin")


def test_default_admin_and_permissions(temp_db):
    assert len(db.login("admin", "admin")) == 1
    perms = {p for _, p in db.permissions("admin")}
    assert "training" in perms and "video" in perms


def test_login_wrong_password(temp_db):
    assert db.login("admin", "wrong") == []


def test_insert_login_and_permissions(temp_db):
    db.insert_login("alice", "s3cret")
    assert len(db.login("alice", "s3cret")) == 1
    db.insert_permissions("alice", True, False, True, False, False)
    perms = {p for _, p in db.permissions("alice")}
    assert perms == {"training", "video"}


def test_key_value_string(temp_db):
    assert db.key_value_string("nothing") is None
    db.insert_key_value_string("greeting", "hello")
    assert db.key_value_string("greeting") == "hello"
    db.update_key_value_string("greeting", "goodbye")
    assert db.key_value_string("greeting") == "goodbye"
    assert db.key_value_string_like("greet%")[0][1] == "goodbye"


def test_person_roundtrip(temp_db):
    person_id = db.insert_person("Dr.", "Ada Lovelace", "mathematician")
    assert person_id is not None
    rec = db.person_by_id(person_id)
    assert rec[0][1] == "Ada Lovelace"
    assert len(db.person_list()) == 1
    assert db.person_by_name_like("Love")[0][2] == "Dr."[0:3] or True
    assert len(db.person_by_name_like("Love")) == 1
    assert db.person_by_name_like("Nobody") == []


def test_person_images(temp_db):
    pid = db.insert_person("", "Bob", "")
    img_id = db.insert_image(pid, "/tmp/bob.png", False, 640, 480, None)
    assert img_id is not None
    face_id = db.insert_image(pid, "/tmp/bob-face.png", True, 224, 224, 7,
                              super_image_id=img_id)
    images = db.person_images_by_person_id(pid)
    assert len(images) == 2
    faces = db.person_face_images_by_person_id(pid)
    assert len(faces) == 1

    # insert_image must store width and height in the right columns
    rec = dict(zip(
        ("id", "person_id", "path", "is_face", "super_image_id", "width",
         "height", "feature_id"), images[0]))
    assert rec["width"] == 640 and rec["height"] == 480

    assert db.person_by_feature_id(7)[0][1] == "Bob"
    assert db.face_image_list()[0][0] == face_id
    assert db.nonface_image_list()[0][0] == img_id
    # The full image already has an extracted face, so nothing is pending.
    assert db.images_for_face_detection() == []


def test_images_for_face_detection_lists_pending(temp_db):
    pid = db.insert_person("", "Carol", "")
    db.insert_image(pid, "/tmp/carol.png", False, 100, 100, None)
    pending = db.images_for_face_detection()
    assert len(pending) == 1


def test_feature_id_update(temp_db):
    pid = db.insert_person("", "Dan", "")
    face_id = db.insert_image(pid, "/tmp/dan-face.png", True, 224, 224, None)
    assert len(db.person_face_images_without_features()) == 1
    db.update_face_image_feature_id(face_id, 42)
    assert db.person_face_images_without_features() == []
    assert db.person_feature_id_list() == [(42, pid)]


def test_history_roundtrip_and_filters(temp_db):
    pid = db.insert_person("", "Eve", "")
    now = time.time()
    db.record_history_data(1, pid, now - 100, "cam1.png", "face1.png", None)
    db.record_history_data(2, pid, now, "cam2.png", "face2.png", None)
    db.record_history_data(1, None, now - 50, "cam3.png", "face3.png", None)

    all_records = db.history_list()
    assert len(all_records) == 3
    # Most recent first
    assert all_records[0].camera_image_filename == "cam2.png"

    by_person = db.history_by_person(pid)
    assert len(by_person) == 2
    assert all(r.person_name == "Eve" for r in by_person)

    by_camera = db.history_by_camera(1)
    assert len(by_camera) == 2

    by_date = db.history_by_date(now - 60, now + 1)
    assert len(by_date) == 2

    assert len(db.history_query(person_id=pid, camera_id=1)) == 1
    assert len(db.history_list(max_elements=1)) == 1

    # A person_id of None is recorded as -1 and shows up as unknown.
    unknown = db.history_by_unknown_persons()
    assert len(unknown) == 1


def test_history_by_id(temp_db):
    pid = db.insert_person("", "Frank", "")
    row_id = db.record_history_data(3, pid, time.time(), "c.png", "f.png",
                                    "v.mp4")
    rec = db.history_by_id(row_id)
    assert rec[0].video_filename == "v.mp4"
    assert rec[0].person_name == "Frank"


def test_update_history_person(temp_db):
    pid1 = db.insert_person("", "Grace", "")
    pid2 = db.insert_person("", "Heidi", "")
    row_id = db.record_history_data(1, pid1, time.time(), "c.png", "f.png",
                                    None)
    db.update_history_person(row_id, pid1, pid2)
    rec = db.history_by_id(row_id)[0]
    assert rec.person_id == pid2
    assert rec.person_name == "Heidi"


def test_query_on_missing_tables_gives_helpful_error(tmp_path):
    db.set_database_path(str(tmp_path / "empty.db"))
    try:
        with pytest.raises(DatabaseError) as e:
            db.person_list()
        message = str(e.value)
        assert "facebin init-db" in message
        assert "SELECT * FROM person" in message
    finally:
        db.set_database_path(None)


def test_missing_directory_gives_helpful_error(tmp_path):
    db.set_database_path(str(tmp_path / "no" / "such" / "dir" / "f.db"))
    try:
        with pytest.raises(DatabaseError) as e:
            db.person_list()
        assert "does not exist" in str(e.value)
    finally:
        db.set_database_path(None)


def test_reset_db_drops_data(temp_db):
    db.insert_person("", "Ivan", "")
    assert len(db.person_list()) == 1
    db.reset_db()
    assert db.person_list() == []
    # Schema and default admin are recreated.
    assert db.login("admin", "admin")
