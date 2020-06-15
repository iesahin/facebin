import redis
import time
import numpy as np

from .utils import *

log = init_logging()

R = redis.Redis(host='localhost', port=6379)
CAMERA_QUEUE = 'framekeys'


def RECOGNIZER_QUEUE(camera_id):
    if type(camera_id) == bytes:
        camera_id = camera_id.decode("utf-8")
    return 'recognizer:{}'.format(camera_id)


HISTORY_QUEUE = 'history'
VIDEO_QUEUE = 'video'
HISTORY_RECORDING_QUEUE = 'record'

STANDARD_EXPIRATION = 3000


def queue_length(queue):
    return R.zcount(queue, 0, "inf")


def get_next_key(queue, delete=False):
    try:
        log.debug("queue: %s", queue)
        res = R.zrange(queue, 0, 0, withscores=True)
        log.debug("res: %s", res)
        if res == []:
            return None, None
        key, score = res[0]
        log.debug("key: %s", key)
        log.debug("score: %s", score)
        log.debug("delete: %s", delete)
        if delete:
            while R.zrem(queue, key) == 0:
                key, score = R.zrange(queue, 0, 0, withscores=True)[0]
                log.debug("%s len: %s, key: %s, score: %s", queue,
                          R.zcount(queue, 0, "inf"), key, score)
        return (key.decode("utf-8"), float(score))
    except Exception as err:
        log.debug("error: %s", err)
        return (None, None)


def init_frame(image_data, dts, video_filename, camera_id):
    key = add_frame(image_data, dts, video_filename, camera_id)
    R.zadd(CAMERA_QUEUE, {key: dts})
    R.expire(key, STANDARD_EXPIRATION)


def add_frame(image_data, dts, video_filename, camera_id):
    key = 'frame:{}:{}'.format(camera_id, dts)
    log.debug("key: %s", key)
    R.hmset(
        key, {
            time_k(): dts,
            camera_id_k(): camera_id,
            image_data_k(): image_data.tostring(),
            image_shape_x_k(): image_data.shape[0],
            image_shape_y_k(): image_data.shape[1],
            image_shape_z_k(): image_data.shape[2],
            image_dtype_k(): str(image_data.dtype),
            filename_k(): video_filename
        })
    return key


def get_frame_image(key, name='image'):
    if key is None:
        return None
    bb = time.time()
    fields = [
        '{}_{}'.format(name, f)
        for f in ['data', 'shape_x', 'shape_y', 'shape_z', 'dtype']
    ]
    log.debug("key: %s", key)
    v = get_frame(key, fields=fields)
    if v[fields[0]] is None:
        return None

    log.debug("v.keys(): %s", v.keys())
    ee = time.time()

    image = make_image_from_redis_data(v, name)
    log.debug("get_frame_image time: %s", ee - bb)

    return image


def make_image_from_redis_data(v, name):
    "This is a helper function to create an numpy image from redis data."

    image = np.frombuffer(
        v['{}_data'.format(name)], dtype=v['{}_dtype'.format(name)])
    image.shape = (int(v['{}_shape_x'.format(name)]),
                   int(v['{}_shape_y'.format(name)]),
                   int(v['{}_shape_z'.format(name)]))
    return image


def get_hash_keys(key):
    vv = [k.decode("utf-8") for k in R.hkeys(key)]
    return vv


def get_frame(key, fields=None):
    if fields is None:
        # decode keys from bytes to utf-8
        vv = {k.decode("utf-8"): v for k, v in R.hgetall(key).items()}
    else:
        vv = {}
        for f in fields:
            vv[f] = R.hget(key, f)
    # log.debug('vv.keys: %s', vv.keys())
    # No need to decode the keys to utf-8, as we receive them upstream
    # return {k.decode('utf-8'): v for k, v in vv.items()}
    return vv


def fix_keys(frame_dict):
    res = {}
    for k, v in frame_dict.items():
        if type(k) == bytes:
            res[k.decode("utf-8")] = v
        else:
            res[k] = v

    return res


def del_frame(key):
    log.debug("Deleting frame: %s", key)
    R.hdel(key)


def make_array(string, dtype, shape):
    arr = np.frombuffer(string, dtype=dtype)
    arr.shape = shape
    return arr


def make_string(arr):
    return (arr.tostring(), arr.dtype, arr.shape)


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
