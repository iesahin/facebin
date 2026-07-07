"""Redis-backed frame queues.

Frames captured from cameras are stored as Redis hashes and their keys are
queued in sorted sets, scored by frame timestamp:

- ``CAMERA_QUEUE``: raw frames waiting for face recognition.
- ``RECOGNIZER_QUEUE(camera_id)``: processed frames per camera, consumed by
  the GUI.
- ``HISTORY_QUEUE``: frames with recognized faces, waiting for the history
  recorder.
- ``HISTORY_RECORDING_QUEUE``: aggregated person appearances waiting to be
  written to the SQLite database.

The connection is created lazily from the loaded configuration; call
:func:`configure` to point this module at a specific Redis instance (tests
inject a ``fakeredis`` client with :func:`set_redis`).
"""

import time

import numpy as np
import redis

from facebin.errors import RedisConnectionError
from .utils import init_logging

log = init_logging()

CAMERA_QUEUE = 'framekeys'
HISTORY_QUEUE = 'history'
VIDEO_QUEUE = 'video'
HISTORY_RECORDING_QUEUE = 'record'

STANDARD_EXPIRATION = 3000

_redis = None
_redis_params = {"host": "localhost", "port": 6379, "db": 0}


def configure(redis_config):
    """Set connection parameters from a :class:`facebin.config.RedisConfig`.

    Resets any existing connection so the next access uses the new
    parameters.
    """
    global _redis, _redis_params
    _redis_params = {
        "host": redis_config.host,
        "port": redis_config.port,
        "db": redis_config.db,
    }
    _redis = None


def set_redis(client):
    """Inject a Redis client directly (used by tests)."""
    global _redis
    _redis = client


def get_redis():
    """Return the shared Redis client, connecting on first use."""
    global _redis
    if _redis is None:
        client = redis.Redis(**_redis_params)
        try:
            client.ping()
        except redis.exceptions.ConnectionError as e:
            raise RedisConnectionError(
                "Cannot connect to Redis at {host}:{port} (db {db}): {err}"
                .format(err=e, **_redis_params),
                hint="Start it with `redis-server`, enable "
                "`autostart = true` in the [redis] section of facebin.toml, "
                "or fix the host/port settings.") from e
        _redis = client
    return _redis


class _RedisProxy:
    """Module-level ``R`` object that defers connecting until first use."""

    def __getattr__(self, name):
        return getattr(get_redis(), name)


R = _RedisProxy()


def RECOGNIZER_QUEUE(camera_id):
    if isinstance(camera_id, bytes):
        camera_id = camera_id.decode("utf-8")
    return 'recognizer:{}'.format(camera_id)


def queue_length(queue):
    return R.zcount(queue, 0, "+inf")


def get_next_key(queue, delete=False):
    """Pop (or peek, when ``delete`` is False) the oldest key in a queue.

    Returns ``(None, None)`` when the queue is empty.
    """
    res = R.zrange(queue, 0, 0, withscores=True)
    if not res:
        return (None, None)
    key, score = res[0]
    if delete:
        # When several consumers race for the same key, zrem returns 0 for
        # the losers; move on to the next key until we win one.
        while R.zrem(queue, key) == 0:
            res = R.zrange(queue, 0, 0, withscores=True)
            if not res:
                return (None, None)
            key, score = res[0]
    return (key.decode("utf-8"), float(score))


def init_frame(image_data, dts, video_filename, camera_id):
    """Store a frame and enqueue it for recognition."""
    key = add_frame(image_data, dts, video_filename, camera_id)
    R.zadd(CAMERA_QUEUE, {key: dts})
    R.expire(key, STANDARD_EXPIRATION)


def add_frame(image_data, dts, video_filename, camera_id):
    """Store a frame image and its metadata as a Redis hash."""
    key = 'frame:{}:{}'.format(camera_id, dts)
    R.hset(
        key,
        mapping={
            time_k(): dts,
            camera_id_k(): camera_id,
            image_data_k(): image_data.tobytes(),
            image_shape_x_k(): image_data.shape[0],
            image_shape_y_k(): image_data.shape[1],
            image_shape_z_k(): image_data.shape[2],
            image_dtype_k(): str(image_data.dtype),
            filename_k(): video_filename if video_filename is not None else ""
        })
    return key


def get_frame_image(key, name='image'):
    """Rebuild the numpy image stored under ``key`` (None when missing)."""
    if key is None:
        return None
    fields = [
        '{}_{}'.format(name, f)
        for f in ['data', 'shape_x', 'shape_y', 'shape_z', 'dtype']
    ]
    v = get_frame(key, fields=fields)
    if v[fields[0]] is None:
        return None
    return make_image_from_redis_data(v, name)


def make_image_from_redis_data(v, name):
    """Create a numpy image from the fields of a Redis frame hash."""
    dtype = v['{}_dtype'.format(name)]
    if isinstance(dtype, bytes):
        dtype = dtype.decode("utf-8")
    image = np.frombuffer(v['{}_data'.format(name)], dtype=dtype)
    image.shape = (int(v['{}_shape_x'.format(name)]),
                   int(v['{}_shape_y'.format(name)]),
                   int(v['{}_shape_z'.format(name)]))
    return image


def get_hash_keys(key):
    return [k.decode("utf-8") for k in R.hkeys(key)]


def get_frame(key, fields=None):
    """Fetch a frame hash; all fields when ``fields`` is None."""
    if fields is None:
        return {k.decode("utf-8"): v for k, v in R.hgetall(key).items()}
    return {f: R.hget(key, f) for f in fields}


def fix_keys(frame_dict):
    """Decode byte keys of a frame dictionary to strings."""
    return {
        (k.decode("utf-8") if isinstance(k, bytes) else k): v
        for k, v in frame_dict.items()
    }


def del_frame(key):
    log.debug("Deleting frame: %s", key)
    R.delete(key)


def make_array(string, dtype, shape):
    arr = np.frombuffer(string, dtype=dtype)
    arr.shape = shape
    return arr


def make_string(arr):
    return (arr.tobytes(), arr.dtype, arr.shape)


# --- Hash field name helpers -------------------------------------------------
# Frame hashes hold a flat namespace of fields; these helpers keep the field
# names consistent between producers and consumers.


def time_k():
    return 'time'


def timestamp_k():
    return 'timestamp'


def camera_id_k():
    return 'camera_id'


def image_data_k():
    return 'image_data'


def image_shape_x_k():
    return 'image_shape_x'


def image_shape_y_k():
    return 'image_shape_y'


def image_shape_z_k():
    return 'image_shape_z'


def image_dtype_k():
    return 'image_dtype'


def filename_k():
    return 'filename'


def processed_image_data_k():
    return 'processed_image_data'


def processed_image_shape_x_k():
    return 'processed_image_shape_x'


def processed_image_shape_y_k():
    return 'processed_image_shape_y'


def processed_image_shape_z_k():
    return 'processed_image_shape_z'


def processed_image_dtype_k():
    return 'processed_image_dtype'


def face_k(face_i):
    return "face_{}".format(face_i)


def face_x_k(face_i):
    return "face_{}_x".format(face_i)


def face_y_k(face_i):
    return "face_{}_y".format(face_i)


def face_w_k(face_i):
    return "face_{}_w".format(face_i)


def face_h_k(face_i):
    return "face_{}_h".format(face_i)


def face_person_id_k(face_i):
    return "face_{}_person_id".format(face_i)


def face_encoding_k(face_i):
    return 'face_{}_encoding'.format(face_i)


def face_encoding_dtype_k(face_i):
    return 'face_{}_encoding_dtype'.format(face_i)


def face_image_k(face_i):
    return 'face_image_{}'.format(face_i)


def face_image_data_k(face_i):
    return 'face_image_{}_data'.format(face_i)


def face_image_shape_x_k(face_i):
    return 'face_image_{}_shape_x'.format(face_i)


def face_image_shape_y_k(face_i):
    return 'face_image_{}_shape_y'.format(face_i)


def face_image_shape_z_k(face_i):
    return 'face_image_{}_shape_z'.format(face_i)


def face_image_dtype_k(face_i):
    return 'face_image_{}_dtype'.format(face_i)


def face_encoding_index_k(face_i):
    return 'face_{}_encoding_index'.format(face_i)


def face_timestamp_k(face_i):
    return 'face_{}_timestamp'.format(face_i)


def face_count_k(face_i):
    return 'face_{}_count'.format(face_i)


def face_feature_id_k(face_i):
    return 'face_{}_feature_id'.format(face_i)
