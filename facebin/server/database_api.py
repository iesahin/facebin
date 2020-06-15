import sqlite3
from .utils import *
import random
import cv2
import datetime as dt
import time
from collections import namedtuple

init_statements = [
    """DROP TABLE login""",
    """CREATE TABLE login (username text, password text)""",
    """INSERT INTO login (username, password) VALUES ("admin", "admin")""",
    """DROP TABLE permissions""",
    """CREATE TABLE permissions (username text, permission text)""",
    """INSERT INTO permissions (username, permission) VALUES ("admin", "training")""",
    """INSERT INTO permissions (username, permission) VALUES ("admin", "tespit")""",
    """INSERT INTO permissions (username, permission) VALUES ("admin", "video")""",
    """INSERT INTO permissions (username, permission) VALUES ("admin", "ipcamerasetting")""",
    """INSERT INTO permissions (username, permission) VALUES ("admin", "loginsetting")""",
    """DROP TABLE key_value_string""",
    """CREATE TABLE key_value_string (key text, value text)""",
    # """INSERT INTO key_value_string VALUES ("camera1.camera_id", "camera1")""",
    # """INSERT INTO key_value_string VALUES ("camera1.name", "Webcam")""",
    # """INSERT INTO key_value_string VALUES ("camera1.device", "/dev/video0")""",
    # """INSERT INTO key_value_string VALUES ("camera1.command", "")""",
    """INSERT INTO key_value_string VALUES ("camera1.camera_id", "camera1")""",
    """INSERT INTO key_value_string VALUES ("camera1.name", "IP Camera at 192.168.1.65")""",
    """INSERT INTO key_value_string VALUES ("camera1.device", "/dev/video0")""",
    """INSERT INTO key_value_string VALUES ("camera1.command", "ffmpeg -loglevel debug -rtsp_transport tcp -i rtsp://admin:euddue1967@192.168.1.65:554/live -c:v rawvideo -pix_fmt yuv420p  -f  v4l2 /dev/video0")""",
    # """INSERT INTO key_value_string VALUES ("camera2.camera_id", "camera2")""",
    # """INSERT INTO key_value_string VALUES ("camera2.name", "IP Camera at 192.168.1.67")""",
    # """INSERT INTO key_value_string VALUES ("camera2.device", "/dev/video1")""",
    # """INSERT INTO key_value_string VALUES ("camera2.command", "ffmpeg -loglevel debug -rtsp_transport tcp -i rtsp://admin:euddue1967@192.168.1.67:554/live -c:v rawvideo -pix_fmt yuv420p  -f  v4l2 /dev/video2")""",
    """INSERT INTO key_value_string VALUES ("camera2.camera_id", "camera2")""",
    """INSERT INTO key_value_string VALUES ("camera2.name", "")""",
    """INSERT INTO key_value_string VALUES ("camera2.device", "")""",
    """INSERT INTO key_value_string VALUES ("camera2.command", "")""",
    """INSERT INTO key_value_string VALUES ("camera3.camera_id", "camera3")""",
    """INSERT INTO key_value_string VALUES ("camera3.name", "")""",
    """INSERT INTO key_value_string VALUES ("camera3.device", "")""",
    """INSERT INTO key_value_string VALUES ("camera3.command", "")""",
    """INSERT INTO key_value_string VALUES ("camera4.camera_id", "camera4")""",
    """INSERT INTO key_value_string VALUES ("camera4.name", "")""",
    """INSERT INTO key_value_string VALUES ("camera4.device", "")""",
    """INSERT INTO key_value_string VALUES ("camera4.command", "")""",
    """DROP TABLE key_value_int""",
    """CREATE TABLE key_value_int (key text, value integer)""",
    """DROP TABLE person""",
    """CREATE TABLE person(id integer primary key, name text, title text, notes text)""",
    """DROP TABLE person_image""",
    """CREATE TABLE person_image(id integer primary key, person_id integer, path
text, is_face bool, super_image_id integer, width integer, height integer,
feature_id integer)""",
    """INSERT INTO key_value_string VALUES ("history.image_store", "/home/iesahin/facebin-data/image-store")""",
    """DROP TABLE history""",
    """CREATE TABLE history(id integer primary key, camera_id integer, person_id integer, time real,
    camera_image_filename text, face_image_filename text, video_filename
    text, original_person_id integer, original_record_change_timestamp real, original_record_change_user integer)""",
]

log = init_logging()


def run_database_query(query, params):
    conn = sqlite3.connect("facebin.db")
    return conn.cursor()


def init_db():
    for s in init_statements:
        print(s)
        try:
            conn = sqlite3.connect("facebin.db")
            c = conn.cursor()
            c.execute(s)
            conn.commit()
            conn.close()
        except Exception as e:
            print(e)


def run_select_query(query, params, fetchsize=None):
    try:
        conn = sqlite3.connect("facebin.db")
        conn.set_trace_callback(print)
        c = conn.cursor()
        c.execute(query, params)
        if fetchsize is None:
            rows = c.fetchall()
        else:
            rows = c.fetchmany(fetchsize)
        log.debug("rows: %s", rows)
        return rows
    except Exception as e:
        log.error(e)
        return []


def run_insert_update_query(query, params):
    try:
        conn = sqlite3.connect("facebin.db", isolation_level='exclusive')
        log.debug("conn: %s", conn)
        conn.set_trace_callback(print)
        c = conn.cursor()
        log.debug("c: %s", c)
        c.execute(query, params)
        log.debug("(query, params): %s", (query, params))
        conn.commit()
        last_row_id = c.lastrowid
        log.debug("last_row_id: %s", last_row_id)
        conn.close()
        log.debug("conn: %s", conn)
        return last_row_id
    except Exception as e:
        print(e)
        return None


def run_insert_update_many(query, params):
    try:
        conn = sqlite3.connect("facebin.db")
        conn.set_trace_callback(print)
        c = conn.cursor()
        c.executemany(query, params)
        conn.commit()
        last_row_id = c.lastrowid
        conn.close()
        return last_row_id
    except Exception as e:
        print(e)
        return None


def login(username, password):
    q = "SELECT * FROM login WHERE username=? AND password=?"
    params = (username, password)
    return run_select_query(q, params)


def permissions(username):
    q = "SELECT * FROM permissions WHERE username=?"
    params = (username, )
    return run_select_query(q, params)


def key_value_string(key):

    q = "SELECT value FROM key_value_string WHERE key=?"
    params = (key, )
    values = run_select_query(q, params)
    if len(values) > 0:
        return values[0][0]
    else:
        return None


def insert_key_value_string(key, value):
    q = "INSERT INTO key_value_string (key, value) VALUES (?, ?)"
    params = (key, value)
    return run_insert_update_query(q, params)


def update_key_value_string(key, value):
    q = "UPDATE key_value_string SET value = ? WHERE key = ?"
    params = (value, key)
    return run_insert_update_query(q, params)


def key_value_int(key):
    q = "SELECT value FROM key_value_int WHERE key=?"
    params = (key, )
    values = run_select_query(q, params)
    if len(values) > 0:
        return values[0]
    else:
        return None


def update_key_value_int(key, value):
    q = "UPDATE key_value_int set value = ? WHERE key = ?"
    params = (value, key)
    run_insert_update_query(q, params)


def key_value_string_like(key):
    q = "SELECT * FROM key_value_string WHERE key LIKE ?"
    params = (key, )
    values = run_select_query(q, params)
    return values


def insert_login(username, password):
    q = "INSERT INTO login (username, password) VALUES (?, ?)"
    params = (username, password)
    run_insert_update_query(q, params)


def insert_permissions(username, training, tespit, video, ipcamerasetting,
                       loginsetting):
    if training:
        q = "INSERT INTO permissions (username, permission) VALUES (?, \"training\")"
        params = (username, )
        run_insert_update_query(q, params)

    if tespit:
        q = "INSERT INTO permissions (username, permission) VALUES (?, \"tespit\")"
        params = (username, )
        run_insert_update_query(q, params)

    if video:
        q = "INSERT INTO permissions (username, permission) VALUES (?, \"video\")"
        params = (username, )
        run_insert_update_query(q, params)

    if ipcamerasetting:
        q = "INSERT INTO permissions (username, permission) VALUES (?, \"ipcamerasetting\")"
        params = (username, )
        run_insert_update_query(q, params)

    if loginsetting:
        q = "INSERT INTO permissions (username, permission) VALUES (?, \"loginsetting\")"
        params = (username, )
        run_insert_update_query(q, params)


def image_list():
    """Returns all person image records """

    q = """SELECT * FROM person_image"""
    params = tuple([])
    return run_select_query(q, params)


def face_image_list():
    """

    """

    q = """SELECT * FROM person_image WHERE is_face = 1"""
    params = tuple([])
    return run_select_query(q, params)


def face_image_by_person_id(person_id):
    q = """SELECT * FROM person_image WHERE person_id = ?"""
    params = (person_id, )
    return run_select_query(q, params)


def nonface_image_list():
    q = """SELECT * FROM person_image WHERE is_face = 0"""
    params = tuple([])
    return run_select_query(q, params)


def image_by_path(path):
    q = """SELECT * FROM person_image WHERE path = ?"""
    params = tuple(path)
    return run_select_query(q, params)


def insert_image(person_id,
                 path,
                 is_face,
                 width,
                 height,
                 feature_id,
                 super_image_id=None):
    q = """INSERT INTO person_image(person_id, path, is_face, width, height,
feature_id, super_image_id) VALUES (?, ?, ?, ?, ?, ?, ?)"""
    params = (person_id, path, is_face, height, width, feature_id,
              super_image_id)
    return run_insert_update_query(q, params)


def images_for_face_detection():
    """Returns the list of images where a face record doesn't exist."""

    q = """SELECT * FROM person_image WHERE (is_face = 0) AND (id NOT IN (SELECT super_image_id FROM person_image WHERE is_face = 1))"""
    params = tuple([])
    return run_select_query(q, params)


def person_by_feature_id(feature_id):
    q = """SELECT person.id AS person_id,
                  person.name AS name,
                  person.title AS title,
                  person.notes AS notes
           FROM person JOIN person_image ON person.id = person_image.person_id
           WHERE person_image.is_face = 1 AND person_image.feature_id = ?"""
    params = (feature_id, )
    return run_select_query(q, params)


def person_face_images_without_features():
    q = """SELECT id, person_id, path, is_face, feature_id
           FROM person_image
           WHERE is_face = 1 AND feature_id is NULL"""
    params = tuple([])
    return run_select_query(q, params)


def person_images_by_person_id(person_id):
    q = """SELECT * FROM person_image WHERE person_image.person_id = ?"""
    params = (person_id, )
    return run_select_query(q, params)


def person_face_images_by_person_id(person_id):
    q = """SELECT * FROM person_image WHERE is_face = 1 AND person_image.person_id = ?"""
    params = (person_id, )
    return run_select_query(q, params)


def person_face_image_by_feature_id(feature_id):
    q = """SELECT * FROM person_image WHERE is_face = 1 AND feature_id = ?"""
    params = (feature_id, )
    return run_select_query(q, params)


def person_feature_id_list():
    q = """SELECT feature_id, person_id FROM person_image WHERE NOT (feature_id IS NULL) ORDER BY feature_id"""
    params = tuple([])
    return run_select_query(q, params)


def update_face_image_feature_id(image_id, feature_id):
    log.debug("image_id: %s", image_id)
    log.debug("feature_id: %s", feature_id)
    q = """UPDATE person_image
    SET feature_id = ?
    WHERE id = ?
    """
    params = (feature_id, image_id)

    return run_insert_update_query(q, params)



def person_list():
    """Returns all person records in the dataset

    >>> pl = person_list()
    >>> pl[0]
    (1, 'firewoman', 'asude', 'avs')
    >>>
    """
    q = """SELECT * FROM person"""
    params = tuple([])
    return run_select_query(q, params)


def person_by_id(person_id):
    """Retrieves a person record by id
    >>> person_by_id(5)[0][1]
    'ceo'
    >>> person_by_id(2)[0][2]
    'affan-ali-ferzan-sahin'
    >>> person_by_id(4)[0][3]
    'hk'
    """
    log.debug(person_id)
    q = """SELECT * FROM person WHERE id=?"""
    params = (person_id, )
    res = run_select_query(q, params)
    log.debug(res)
    return res


def person_by_name_like(name):
    q = """SELECT * FROM person WHERE name LIKE ?"""
    params = ("%{}%".format(name), )
    return run_select_query(q, params)


def insert_person(title, name, notes):
    q = """INSERT INTO person (title, name, notes) VALUES (?, ?, ?)"""
    params = (title, name, notes)
    log.debug(params)
    return run_insert_update_query(q, params)


# HISTORY_IMAGE_STORE = key_value_string("history.image_store")
# if HISTORY_IMAGE_STORE is None:
#     HISTORY_IMAGE_STORE = '/tmp/facebin-cam-logs'
#     insert_key_value_string('history.image_store', HISTORY_IMAGE_STORE)

# log.debug(HISTORY_IMAGE_STORE)
# if not os.path.isdir(HISTORY_IMAGE_STORE):
#     os.makedirs(HISTORY_IMAGE_STORE)


def record_history_data(camera_id: int, person_id: int, time: float,
                        camera_image_filename: str, face_image_filename: str,
                        video_filename: str):
    if person_id is None:
        person_id = -1

    query = """INSERT INTO history(camera_id, person_id, time, camera_image_filename,
    face_image_filename, video_filename) VALUES (?, ?, ?, ?, ?, ?)"""
    params = (camera_id, person_id, time, camera_image_filename,
              face_image_filename, video_filename)

    return run_insert_update_query(query, params)


def record_multiple_history_data(record_dict: dict):
    record_list = []
    for k, v in record_dict.items():
        camera_id, person_id, camera_image, face_image, time, video_filename = v
        r = random.randint(10000, 100000)
        face_image_filename = "{}/face-{}-{}-{}-{}.png".format(
            HISTORY_IMAGE_STORE, person_id, camera_id, int(time), r)
        camera_image_filename = "{}/cam-{}-{}-{}-{}.png".format(
            HISTORY_IMAGE_STORE, camera_id, person_id, int(time), r)
        cv2.imwrite(face_image_filename, face_image)
        cv2.imwrite(camera_image_filename, camera_image)
        record_list.append((camera_id, person_id, time, camera_image_filename,
                            face_image_filename, video_filename))

    query = """INSERT INTO history(camera_id, person_id, time, camera_image_filename,
    face_image_filename, video_filename) VALUES (?, ?, ?, ?, ?, ?)"""
    return run_insert_update_many(query, record_list)


# def make_history_list(history_records):

#     history_list = []

#     for hr in history_records:
#         (rowid, camera_id, person_id, time, camera_image_filename,
#          face_image_filename, video_filename) = hr

#         person_rec = person_by_id(person_id)
#         log.debug(person_rec)
#         if len(person_rec) == 0:
#             person_name = "Unknown Unknown"
#         elif len(person_rec) == 1:
#             person_name = person_rec[0][1]
#         else:
#             log.error("More than 1 person with ID: %s", person_id)

#         # camera_key = "{}.name".format(camera_id)
#         # camera_name = key_value_string(camera_key)

#         camera_name = "Camera {}".format(camera_id)

#         if os.path.exists(camera_image_filename):
#             camera_image = cv2.imread(camera_image_filename)
#         else:
#             camera_image = None

#         if os.path.exists(face_image_filename):
#             face_image = cv2.imread(face_image_filename)
#         else:
#             face_image = None

#         history_list.append((camera_name, person_name, time, camera_image,
#                              face_image, video_filename, rowid))

#     return history_list


def make_history_list_light(history_records):

    HistoryRecord = namedtuple("HistoryRecord", [
        "row_id", "camera_id", "person_id", "time", "camera_image_filename",
        "face_image_filename", "video_filename", "person_name", "person_title",
        "person_notes"
    ])

    history_list = [HistoryRecord._make(hr) for hr in history_records]
    return history_list


def history_by_id(history_id):
    query = """
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
    WHERE history.id = ? ;
        """
    params = (history_id, )

    results = run_select_query(query, params)
    history_list = make_history_list_light(results)
    return history_list


def history_query(person_id=None,
                  camera_id=None,
                  datetime_begin=None,
                  datetime_end=None,
                  max_elements=None):
    """General history query with multiple criteria
    WARNING! @datetime_begin @datetime_end should be in UTC.

    Args:
        person_id: ID of the person to be searched in history or None for all

    """
    general_query = """
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
    FROM history LEFT JOIN person ON history.person_id = person.id {where_clause}
    ORDER BY time DESC {limit_clause};
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
        # datetime_begin = datetime_begin.replace(
        #     tzinfo=dt.timezone.utc).timestamp()
        params.append(datetime_begin)
    if datetime_end is not None:
        criteria.append("history.time <= ?")
        # datetime_end = datetime_end.replace(tzinfo=dt.timezone.utc).timestamp()
        params.append(datetime_end)

    if len(criteria) > 0:
        where_clause = "WHERE " + " AND ".join(criteria)
        params = tuple(params)
    else:
        where_clause = ""
        params = tuple([])

    if max_elements is None:
        limit_clause = ""
    else:
        limit_clause = "LIMIT {}".format(max_elements)

    query = general_query.format(
        where_clause=where_clause, limit_clause=limit_clause)

    log.debug("query: %s", query)

    results = run_select_query(query, params)
    history_list = make_history_list_light(results)
    return history_list


def history_list(max_elements=1000):
    return history_query(max_elements=max_elements)


def history_by_person(person_id, max_elements=1000):
    return history_query(person_id=person_id, max_elements=max_elements)


def history_by_camera(camera_id, max_elements=1000):
    return history_query(camera_id=camera_id, max_elements=max_elements)


def history_by_date(begin, end, max_elements=1000):
    return history_query(
        datetime_begin=begin, datetime_end=end, max_elements=max_elements)


def history_by_unknown_persons(max_elements=1000):
    general_query = """
    SELECT id, camera_id, person_id, time, camera_image_filename, face_image_filename, video_filename
    FROM history WHERE history.person_id < 0
    ORDER BY time DESC {limit_clause};
        """
    if max_elements is None:
        limit_clause = ""
    else:
        limit_clause = "LIMIT {}".format(max_elements)

    query = general_query.format(limit_clause=limit_clause)
    params = tuple([])

    return run_select_query(query, params)


def update_history_person(rowid, current_person_id, new_person_id):
    query = """UPDATE history
    SET person_id = ? ,
    original_person_id = ? ,
    original_record_change_timestamp = ? , 
    original_record_change_user = ? 
    WHERE rowid = ?
    AND person_id = ?
    """
    ts = time.time()

    params = (new_person_id, current_person_id, ts, 0, rowid,
              current_person_id)

    return run_insert_update_query(query, params)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
