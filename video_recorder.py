import os
import cv2
import random
import datetime as dt
import io
import time

from PySide2 import QtCore as qtc
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg

import configparser as cp

from utils import *

log = init_logging()

cfg = cp.ConfigParser()
cfg.read("facebin.ini")

video_record_dir = os.path.expandvars(cfg['video']['video_record_dir'])
video_record_period = int(cfg['video']['video_seconds_per_file'])

video_writers = {}
last_checked = 0


def get_video_index(t):
    return (t // video_record_period) * video_record_period


def get_video_filename(t, cam_id):
    video_index = get_video_index(t)
    tstr = dt.datetime.fromtimestamp(video_index).strftime("%F-%H-%M-%S")
    fn = "{}/camera-{}-index-{}.xvid".format(video_record_dir, cam_id, tstr)
    return fn


def record(t, cam_id, camera_image, video_filename):
    global video_writers
    global last_checked
    fn = video_filename
    log.debug("video_writers.keys(): %s", video_writers.keys())
    if fn not in video_writers:
        log.debug("Opening: %s", fn)
        size = (camera_image.shape[1], camera_image.shape[0])
        # log.debug("Size: %s", size)
        vw = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc('X', 'V', 'I',
                                                        'D'), 30,
                             (camera_image.shape[1], camera_image.shape[0]))
        log.debug("video_writer created: %s", vw)
        video_writers[fn] = (vw, time.time())
        log.debug("video_writers[%s]: %s", fn, video_writers[fn])

    vw, _ = video_writers[fn]
    # log.debug("vw: %s", vw)
    vw.write(camera_image)
    # log.debug("Written: %s", camera_image.shape)
    video_writers[fn] = (vw, time.time())
    # log.debug("video_writers[%s]: %s", fn, video_writers[fn])
    # close long unused files

    if (time.time() - last_checked) > 10:
        log.debug("Entered File Close Check")
        last_checked = time.time()
        for fn in list(video_writers.keys()):
            vw, last_used = video_writers[fn]
            diff = (time.time() - last_used)
            log.debug("last_used: %s diff: %s fn: %s ", last_used, diff, fn)
            if diff > (video_record_period * 2):
                log.debug("Closing: {}".format(fn))
                vw.release()
                del video_writers[fn]
                log.debug("Closed: {}".format(fn))


# def write_loop(R):

#     print("Beginning to Write Videos")

#     while True:
#         key = R.lpop('framekeys')
#         l = R.llen('framekeys')
#         log.debug("len: %s, key: %s", l, key)
#         if key is None:
#             # log.debug("key is None")
#             continue
#         # skip if we are too behind:
#         if l > 1000:
#             R.delete(key)
#             continue

#         v = R.hgetall(key)
#         # log.debug("v: %s", v)
#         R.delete(key)
#         t = float(v[b'time'])
#         cam_id = v[b'camera_id'].decode('utf-8')
#         # img_str = v[b'image_string']
#         # image_from_str = np.load(io.BytesIO(img_str))
#         image_data = v[b'image_data']
#         image_shape_x = v[b'image_shape_x']
#         image_shape_y = v[b'image_shape_y']
#         image_shape_z = v[b'image_shape_z']
#         image_dtype = v[b'image_dtype']
#         image = np.frombuffer(image_data, dtype=image_dtype)
#         image.shape = (int(image_shape_x), int(image_shape_y),
#                        int(image_shape_z))
#         # assert (image_from_str.mean() == image.mean())
#         filename = v[b'filename'].decode('utf-8')
#         record(t, cam_id, image, filename)
#         # log.debug("t: %s", t)

# def main():
#     import redis
#     R = redis.Redis()
#     write_loop(R)

# if __name__ == '__main__':
#     main()
