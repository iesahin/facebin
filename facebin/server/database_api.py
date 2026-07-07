"""SQLite persistence layer for Facebin.

Stores persons, their images, recognition history, users/permissions, and a
small key-value area.  The database location comes from the ``[database]``
section of ``facebin.toml``; call :func:`configure` (done automatically by
the CLI) or :func:`set_database_path` before using the query helpers.

All helpers raise :class:`facebin.errors.DatabaseError` with the failing
query attached instead of silently returning empty results.
"""

import os
import sqlite3
import time
from collections import namedtuple

from facebin.errors import DatabaseError
from .utils import init_logging

log = init_logging()

_database_path = None

SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS login (username text, password text)""",
    """CREATE TABLE IF NOT EXISTS permissions (username text, permission text)""",
    """CREATE TABLE IF NOT EXISTS key_value_string (key text, value text)""",
    """CREATE TABLE IF NOT EXISTS key_value_int (key text, value integer)""",
    """CREATE TABLE IF NOT EXISTS person (
        id integer primary key, name text, title text, notes text)""",
    """CREATE TABLE IF NOT EXISTS person_image (
        id integer primary key, person_id integer, path text, is_face bool,
        super_image_id integer, width integer, height integer,
        feature_id integer)""",
    """CREATE TABLE IF NOT EXISTS history (
        id integer primary key, camera_id integer, person_id integer,
        time real, camera_image_filename text, face_image_filename text,
        video_filename text, original_person_id integer,
        original_record_change_timestamp real,
        original_record_change_user integer)""",
]

DEFAULT_ADMIN = ("admin", "admin")
DEFAULT_ADMIN_PERMISSIONS = ("training", "tespit", "video", "ipcamerasetting",
                             "loginsetting")


def configure(config):
    """Point this module at the database defined in a loaded Config."""
    set_database_path(config.database.resolved_path())


def set_database_path(path):
    global _database_path
    _database_path = path


def get_database_path():
    """Return the configured database path (default: ./facebin.db)."""
    return _database_path if _database_path is not None else "facebin.db"


def _connect(isolation_level=""):
    path = get_database_path()
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise DatabaseError(
            "Database directory '{}' does not exist.".format(parent),
            hint="Run `facebin init-db` to create the database, or fix "
            "the [database] path in facebin.toml.")
    try:
        if isolation_level == "":
            return sqlite3.connect(path)
        return sqlite3.connect(path, isolation_level=isolation_level)
    except sqlite3.Error as e:
        raise DatabaseError(
            "Cannot open SQLite database '{}': {}".format(path, e),
            hint="Check that the file is a readable, writable SQLite "
            "database.") from e


def init_db():
    """Create missing tables and the default admin user.

    Safe to call repeatedly; existing data is preserved.
    """
    path = get_database_path()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    conn = _connect()
    try:
        c = conn.cursor()
        for statement in SCHEMA_STATEMENTS:
            c.execute(statement)
        c.execute("SELECT COUNT(*) FROM login")
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO login (username, password) VALUES (?, ?)",
                      DEFAULT_ADMIN)
            for permission in DEFAULT_ADMIN_PERMISSIONS:
                c.execute(
                    "INSERT INTO permissions (username, permission) "
                    "VALUES (?, ?)", (DEFAULT_ADMIN[0], permission))
            log.warning(
                "Created default admin user with password 'admin'; "
                "change it before deploying.")
        conn.commit()
    except sqlite3.Error as e:
        raise DatabaseError(
            "Cannot initialize database '{}': {}".format(path, e),
            hint="Check disk space and file permissions.") from e
    finally:
        conn.close()


def reset_db():
    """DESTRUCTIVE: drop all Facebin tables and recreate them."""
    conn = _connect()
    try:
        c = conn.cursor()
        for table in ("login", "permissions", "key_value_string",
                      "key_value_int", "person", "person_image", "history"):
            c.execute("DROP TABLE IF EXISTS {}".format(table))
        conn.commit()
    except sqlite3.Error as e:
        raise DatabaseError(
            "Cannot reset database '{}': {}".format(get_database_path(), e)
        ) from e
    finally:
        conn.close()
    init_db()


def run_select_query(query, params, fetchsize=None):
    conn = _connect()
    try:
        c = conn.cursor()
        c.execute(query, params)
        if fetchsize is None:
            return c.fetchall()
        return c.fetchmany(fetchsize)
    except sqlite3.Error as e:
        raise DatabaseError(
            "Query failed on '{}': {}\nQuery: {}\nParameters: {!r}".format(
                get_database_path(), e, query.strip(), params),
            hint="If tables are missing, run `facebin init-db` first."
        ) from e
    finally:
        conn.close()


def run_insert_update_query(query, params):
    conn = _connect(isolation_level='EXCLUSIVE')
    try:
        c = conn.cursor()
        c.execute(query, params)
        conn.commit()
        return c.lastrowid
    except sqlite3.Error as e:
        raise DatabaseError(
            "Write failed on '{}': {}\nQuery: {}\nParameters: {!r}".format(
                get_database_path(), e, query.strip(), params),
            hint="If tables are missing, run `facebin init-db` first."
        ) from e
    finally:
        conn.close()


def run_insert_update_many(query, params):
    conn = _connect()
    try:
        c = conn.cursor()
        c.executemany(query, params)
        conn.commit()
        return c.lastrowid
    except sqlite3.Error as e:
        raise DatabaseError(
            "Bulk write failed on '{}': {}\nQuery: {}\n{} parameter rows."
            .format(get_database_path(), e, query.strip(), len(params)),
            hint="If tables are missing, run `facebin init-db` first."
        ) from e
    finally:
        conn.close()


# --- Users and permissions ---------------------------------------------------


def login(username, password):
    q = "SELECT * FROM login WHERE username=? AND password=?"
    return run_select_query(q, (username, password))


def permissions(username):
    q = "SELECT * FROM permissions WHERE username=?"
    return run_select_query(q, (username, ))


def insert_login(username, password):
    q = "INSERT INTO login (username, password) VALUES (?, ?)"
    run_insert_update_query(q, (username, password))


def insert_permissions(username, training, tespit, video, ipcamerasetting,
                       loginsetting):
    granted = [
        name for name, flag in (("training", training), ("tespit", tespit),
                                ("video", video),
                                ("ipcamerasetting", ipcamerasetting),
                                ("loginsetting", loginsetting)) if flag
    ]
    for permission in granted:
        q = "INSERT INTO permissions (username, permission) VALUES (?, ?)"
        run_insert_update_query(q, (username, permission))


# --- Key-value store ----------------------------------------------------------


def key_value_string(key):
    q = "SELECT value FROM key_value_string WHERE key=?"
    values = run_select_query(q, (key, ))
    return values[0][0] if values else None


def insert_key_value_string(key, value):
    q = "INSERT INTO key_value_string (key, value) VALUES (?, ?)"
    return run_insert_update_query(q, (key, value))


def update_key_value_string(key, value):
    q = "UPDATE key_value_string SET value = ? WHERE key = ?"
    return run_insert_update_query(q, (value, key))


def key_value_int(key):
    q = "SELECT value FROM key_value_int WHERE key=?"
    values = run_select_query(q, (key, ))
    return values[0] if values else None


def update_key_value_int(key, value):
    q = "UPDATE key_value_int set value = ? WHERE key = ?"
    run_insert_update_query(q, (value, key))


def key_value_string_like(key):
    q = "SELECT * FROM key_value_string WHERE key LIKE ?"
    return run_select_query(q, (key, ))


# --- Person images ------------------------------------------------------------


def image_list():
    """Return all person image records."""
    return run_select_query("SELECT * FROM person_image", tuple())


def face_image_list():
    """Return the person images that are cropped face images."""
    return run_select_query("SELECT * FROM person_image WHERE is_face = 1",
                            tuple())


def face_image_by_person_id(person_id):
    q = "SELECT * FROM person_image WHERE person_id = ?"
    return run_select_query(q, (person_id, ))


def nonface_image_list():
    return run_select_query("SELECT * FROM person_image WHERE is_face = 0",
                            tuple())


def image_by_path(path):
    q = "SELECT * FROM person_image WHERE path = ?"
    return run_select_query(q, (path, ))


def insert_image(person_id,
                 path,
                 is_face,
                 width,
                 height,
                 feature_id,
                 super_image_id=None):
    q = """INSERT INTO person_image(person_id, path, is_face, width, height,
           feature_id, super_image_id) VALUES (?, ?, ?, ?, ?, ?, ?)"""
    params = (person_id, path, is_face, width, height, feature_id,
              super_image_id)
    return run_insert_update_query(q, params)


def images_for_face_detection():
    """Return images that do not have an extracted face record yet."""
    q = """SELECT * FROM person_image WHERE (is_face = 0) AND (id NOT IN
           (SELECT super_image_id FROM person_image WHERE is_face = 1
            AND super_image_id IS NOT NULL))"""
    return run_select_query(q, tuple())


def person_by_feature_id(feature_id):
    q = """SELECT person.id AS person_id,
                  person.name AS name,
                  person.title AS title,
                  person.notes AS notes
           FROM person JOIN person_image ON person.id = person_image.person_id
           WHERE person_image.is_face = 1 AND person_image.feature_id = ?"""
    return run_select_query(q, (feature_id, ))


def person_face_images_without_features():
    q = """SELECT id, person_id, path, is_face, feature_id
           FROM person_image
           WHERE is_face = 1 AND feature_id is NULL"""
    return run_select_query(q, tuple())


def person_images_by_person_id(person_id):
    q = "SELECT * FROM person_image WHERE person_image.person_id = ?"
    return run_select_query(q, (person_id, ))


def person_face_images_by_person_id(person_id):
    q = """SELECT * FROM person_image
           WHERE is_face = 1 AND person_image.person_id = ?"""
    return run_select_query(q, (person_id, ))


def person_face_image_by_feature_id(feature_id):
    q = "SELECT * FROM person_image WHERE is_face = 1 AND feature_id = ?"
    return run_select_query(q, (feature_id, ))


def person_feature_id_list():
    q = """SELECT feature_id, person_id FROM person_image
           WHERE NOT (feature_id IS NULL) ORDER BY feature_id"""
    return run_select_query(q, tuple())


def update_face_image_feature_id(image_id, feature_id):
    q = "UPDATE person_image SET feature_id = ? WHERE id = ?"
    return run_insert_update_query(q, (feature_id, image_id))


# --- Persons ------------------------------------------------------------------


def person_list():
    """Return all person records."""
    return run_select_query("SELECT * FROM person", tuple())


def person_by_id(person_id):
    q = "SELECT * FROM person WHERE id=?"
    return run_select_query(q, (person_id, ))


def person_by_name_like(name):
    q = "SELECT * FROM person WHERE name LIKE ?"
    return run_select_query(q, ("%{}%".format(name), ))


def insert_person(title, name, notes):
    q = "INSERT INTO person (title, name, notes) VALUES (?, ?, ?)"
    return run_insert_update_query(q, (title, name, notes))


# --- History ------------------------------------------------------------------

HistoryRecord = namedtuple("HistoryRecord", [
    "row_id", "camera_id", "person_id", "time", "camera_image_filename",
    "face_image_filename", "video_filename", "person_name", "person_title",
    "person_notes"
])


def record_history_data(camera_id, person_id, time_, camera_image_filename,
                        face_image_filename, video_filename):
    if person_id is None:
        person_id = -1
    q = """INSERT INTO history(camera_id, person_id, time,
           camera_image_filename, face_image_filename, video_filename)
           VALUES (?, ?, ?, ?, ?, ?)"""
    params = (camera_id, person_id, time_, camera_image_filename,
              face_image_filename, video_filename)
    return run_insert_update_query(q, params)


def make_history_list_light(history_records):
    return [HistoryRecord._make(hr) for hr in history_records]


_HISTORY_SELECT = """
    SELECT history.id as id,
           history.camera_id as camera_id,
           history.person_id as person_id,
           history.time as time,
           history.camera_image_filename as camera_image_filename,
           history.face_image_filename as face_image_filename,
           history.video_filename as video_filename,
           person.name as person_name,
           person.title as person_title,
           person.notes as person_notes
    FROM history LEFT JOIN person ON history.person_id = person.id
"""


def history_by_id(history_id):
    query = _HISTORY_SELECT + " WHERE history.id = ?"
    results = run_select_query(query, (history_id, ))
    return make_history_list_light(results)


def history_query(person_id=None,
                  camera_id=None,
                  datetime_begin=None,
                  datetime_end=None,
                  max_elements=None):
    """Query history with optional filters.

    ``datetime_begin`` and ``datetime_end`` are UTC UNIX timestamps.
    """
    criteria = []
    params = []
    if person_id is not None:
        criteria.append("history.person_id = ?")
        params.append(person_id)
    if camera_id is not None:
        criteria.append("history.camera_id = ?")
        params.append(camera_id)
    if datetime_begin is not None:
        criteria.append("history.time >= ?")
        params.append(datetime_begin)
    if datetime_end is not None:
        criteria.append("history.time <= ?")
        params.append(datetime_end)

    where_clause = " WHERE " + " AND ".join(criteria) if criteria else ""
    limit_clause = "" if max_elements is None else " LIMIT {}".format(
        int(max_elements))

    query = (_HISTORY_SELECT + where_clause + " ORDER BY time DESC" +
             limit_clause)
    results = run_select_query(query, tuple(params))
    return make_history_list_light(results)


def history_list(max_elements=1000):
    return history_query(max_elements=max_elements)


def history_by_person(person_id, max_elements=1000):
    return history_query(person_id=person_id, max_elements=max_elements)


def history_by_camera(camera_id, max_elements=1000):
    return history_query(camera_id=camera_id, max_elements=max_elements)


def history_by_date(begin, end, max_elements=1000):
    return history_query(datetime_begin=begin,
                         datetime_end=end,
                         max_elements=max_elements)


def history_by_unknown_persons(max_elements=1000):
    limit_clause = "" if max_elements is None else " LIMIT {}".format(
        int(max_elements))
    query = ("""SELECT id, camera_id, person_id, time, camera_image_filename,
                face_image_filename, video_filename
                FROM history WHERE history.person_id < 0
                ORDER BY time DESC""" + limit_clause)
    return run_select_query(query, tuple())


def update_history_person(rowid, current_person_id, new_person_id):
    """Reassign a history record to another person, keeping the original."""
    query = """UPDATE history
               SET person_id = ?,
                   original_person_id = ?,
                   original_record_change_timestamp = ?,
                   original_record_change_user = ?
               WHERE rowid = ? AND person_id = ?"""
    ts = time.time()
    params = (new_person_id, current_person_id, ts, 0, rowid,
              current_person_id)
    return run_insert_update_query(query, params)
