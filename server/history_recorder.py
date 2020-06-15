import os
import cv2
import random
import time
import datetime as dt
import configparser as cp
import sys

from PySide2 import QtCore as qtc
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg

import database_api as db
from dataset_manager_v3 import FEATURE_SIZE
import redis_queue_utils as rqu
import redis

from utils import *

log = init_logging()


class VideoHistoryRecorder(qtc.QObject):
    """Records the history data for the face recognition system

    Args:

    encoding_delta_limit (float): The similarity metric limit to determine
    whether two consecutive face images are the same. 

    """

    def __init__(self, encoding_delta_limit=500):
        cfg = cp.ConfigParser()
        cfg.read("facebin.ini")

        self.record_dir = os.path.expandvars(
            cfg['history']['image_record_dir'])
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        self.record_period = int(cfg['history']['image_record_period'])

        self.period_begin = int(time.time())
        self.current_records = []
        self.current_encodings = np.empty((0, FEATURE_SIZE), dtype=np.float32)
        self.encoding_delta_limit = encoding_delta_limit

        self._record_person = {}
        self._record_encoding = {}

        self._db_write_timeout = 30
        self._last_db_write = time.time()

    def record_to_database(self, index):
        # Get data

        r = self.current_records[i]
        del self.current_records[i]

        log.debug("the_data: %s", the_data)

        # Save images
        tstr = dt.datetime.fromtimestamp(t).strftime("%F-%H-%M-%S-%f")
        face_image_filename = "{}/face-{}-{}-{}.png".format(
            self.record_dir, r["person_id"], r["cam_id"], tstr)
        camera_image_filename = "{}/cam_img-{}-{}-{}.png".format(
            self.record_dir, cam_id, person_id, tstr)

        cv2.imwrite(face_image_filename, r["face_image"])
        cv2.imwrite(camera_image_filename, r["image"])
        # Save DB
        res = db.record_history_data(cam_id, person_id, t,
                                     camera_image_filename,
                                     face_image_filename, None)

    # def record(self,
    #            t,
    #            cam_id,
    #            person_id,
    #            camera_image,
    #            face_image,
    #            video_filename=None):

    #     log.debug("t: %s", t)
    #     log.debug("cam_id: %s", cam_id)
    #     log.debug("person_id: %s", person_id)
    #     log.debug("camera_image.shape: %s", camera_image.shape)
    #     log.debug("face_image.shape: %s", face_image.shape)
    #     log.debug("video_filename: %s", video_filename)

    #     if person_id == None or person_id < 0:
    #         person_id = -1 * random.randint(1000000, 100000000)

    #     period_counter = (t - self.period_begin) // self.record_period
    #     triplet = (cam_id, person_id, period_counter)
    #     log.debug("triplet: %s", triplet)

    #     if triplet not in self.current_records:
    #         log.debug("t: %s", t)
    #         tstr = dt.datetime.fromtimestamp(t).strftime("%F-%H-%M-%S-%f")
    #         face_image_filename = "{}/face-{}-{}-{}.png".format(
    #             self.record_dir, person_id, cam_id, tstr)
    #         camera_image_filename = "{}/cam_img-{}-{}-{}.png".format(
    #             self.record_dir, cam_id, person_id, tstr)

    #         cv2.imwrite(face_image_filename, face_image)
    #         cv2.imwrite(camera_image_filename, camera_image)

    #         res = db.record_history_data(cam_id, person_id, t,
    #                                      camera_image_filename,
    #                                      face_image_filename, video_filename)
    #         self.current_records[triplet] = res

    def _camera_person_id_k(self, face_record):
        return "{}-{}".format(face_record['camera_id'],
                              face_record['person_id'])

    def _record_by_camera_person_id(self, face_record):
        k = self._camera_person_id_k(face_record)
        if k in self._record_person:
            return self._record_person[k]
        else:
            return None

    def _record_by_encoding(self, face_record):
        encoding = face_record['encoding']
        log.debug("type(encoding): %s", type(encoding))
        encoding = np.frombuffer(encoding, dtype=np.float32)
        log.debug("encoding.dtype: %s", encoding.dtype)
        log.debug("encoding.shape: %s", encoding.shape)
        encoding.shape = (FEATURE_SIZE, )
        log.debug("encoding.dtype: %s", encoding.dtype)
        log.debug("encoding.shape: %s", encoding.shape)
        log.debug("self.current_encodings.shape: %s",
                  self.current_encodings.shape)
        log.debug("self.current_encodings.dtype: %s",
                  self.current_encodings.dtype)
        if self.current_encodings.shape[0] > 0:
            dists = np.linalg.norm(self.current_encodings - encoding, axis=1)
            log.debug("dists.shape: %s", dists.shape)
            i = np.argmin(dists)
            log.debug("i: %s", i)
            log.debug("dists[i]: %s", dists[i]) 
            if dists[i] < self.encoding_delta_limit:
                log.debug("self._record_encoding.keys(): %s",
                          self._record_encoding.keys())
                # we refresh the encoding in _update_data
                # self.current_encodings[i] = encoding
                return self._record_encoding[i]
            else:
                return None
        else:
            return None

    # def _key_face_i_k(self, key, face_i):
    #     return "{}-{}".format(key, face_i)

    # def _record_by_key_face_i(self, key, face_i):
    #     k = self._key_face_i_k(key, face_i)
    #     if k in self._record_key_face:
    #         return self._record_key_face[k]
    #     else:
    #         return None

    def insert_update_records(self, key, score, face_i):
        """Checks whether we already have a record for this person appearing in previous frames.

        Gets the encoding, face coordinates, person id from redis and compares

        Encoding should be similar to a previous frame by a margin of
        `self.encoding_delta_limit`, person_id should be identical. If
        person_id is negative, i.e. person is unknown, then we only check encoding.

        Face coords are not used in comparison. They are used the select the
        larger face among the consecutive set of faces.

        WARNING: Any twins who have similar face encodings will be recorded
        once if they appear in the same frame. If this happens frequently we
        may also need to add face coordinates to the similarity metric.

        If face coordinates are larger than previous records, the record is
        updated.

        This function may update the record's timestamp, face coordinates, face
        image.

        """

        face_data = rqu.get_frame(
            key,
            fields=[
                rqu.face_encoding_k(face_i),
                rqu.face_encoding_dtype_k(face_i),
                rqu.face_x_k(face_i),
                rqu.face_y_k(face_i),
                rqu.face_w_k(face_i),
                rqu.face_h_k(face_i),
                rqu.face_person_id_k(face_i),
                rqu.camera_id_k(),
                rqu.face_feature_id_k(face_i),
            ])

        log.debug("face_data[rqu.face_encoding_dtype_k(face_i)]: %s",
                  face_data[rqu.face_encoding_dtype_k(face_i)])

        encoding = face_data[rqu.face_encoding_k(face_i)]
        person_id = face_data[rqu.face_person_id_k(face_i)]
        face_x = int(face_data[rqu.face_x_k(face_i)])
        face_y = int(face_data[rqu.face_y_k(face_i)])
        face_w = int(face_data[rqu.face_w_k(face_i)])
        face_h = int(face_data[rqu.face_h_k(face_i)])
        face_size = face_w * face_h
        camera_id = face_data[rqu.camera_id_k()]
        feature_id = face_data[rqu.face_feature_id_k(face_i)]

        face_record = {
            'encoding': encoding,
            'person_id': person_id,
            'face_x': face_x,
            'face_y': face_y,
            'face_w': face_w,
            'face_h': face_h,
            'face_size': face_size,
            'camera_id': camera_id,
            'face_i': face_i,
            'face_size': face_size,
            'timestamp': time.time(),
            'count': 1,
            'face_key': key,
            'face_score': score,
            'feature_id': feature_id
        }
        # TODO: We also need to consider multiple cameras seeing the same person

        log.debug("face_record: %s", face_record) 

        log.debug("R.zcount(rqu.HISTORY_RECORDING_QUEUE): %s", rqu.R.zcount(rqu.HISTORY_RECORDING_QUEUE, 0, "inf"))
        if int(person_id) >= 0:  # that is, we found the face in the database
            r = self._record_by_camera_person_id(face_record)
            log.debug("r: %s", r) 
            if r is None:
                log.debug("inserting face_record: %s", face_record) 
                self._insert_data(face_record)
            else:
                log.debug("updating face_record: %s", face_record) 
                self._update_data(r, face_record)
        else:  # we don't know the face
            r = self._record_by_encoding(face_record)
            log.debug("type(r): %s", type(r))
            log.debug("r is None: %s", r is None)
            if r is None:
                log.debug("inserting face_record: %s", face_record) 
                self._insert_data(face_record)
            else:
                log.debug("updating face_record: %s", face_record) 
                self._update_data(r, face_record)

    def _generate_hash_key(self, face_record):
        "Generates a hash key for the record. "
        return "{}-{}-{}".format(face_record['face_key'],
                                 int(face_record['face_i']),
                                 time.time())

    def _insert_data(self, face_record):
        """Inserts a new record to the currently following data."""
        encoding = rqu.make_array(face_record['encoding'], np.float32,
                                  (1, FEATURE_SIZE))
        log.debug("encoding.shape: %s", encoding.shape)
        log.debug("self.current_encodings.shape: %s",
                  self.current_encodings.shape)
        self.current_encodings = np.vstack([self.current_encodings, encoding])
        log.debug("self.current_encodings.shape: %s",
                  self.current_encodings.shape)
        enc_index = self.current_encodings.shape[0] - 1
        face_record['enc_index'] = enc_index
        record_key = self._generate_hash_key(face_record)
        face_record['record_key'] = record_key

        log.debug("face_record['person_id']): %s", face_record['person_id'])
        person_id = int(face_record['person_id'])
        log.debug("person_id: %s", person_id)

        self._record_encoding[enc_index] = face_record
        log.debug("enc_index: %s", enc_index) 

        if person_id >= 0:
            pc_key = self._camera_person_id_k(face_record)
            self._record_person[pc_key] = face_record
        log.debug("record_key: %s", record_key) 
        r_res = rqu.R.hmset(record_key, face_record)
        log.debug("r_res: %s", r_res) 
        log.debug("R.zcount(rqu.HISTORY_RECORDING_QUEUE): %s", rqu.R.zcount(rqu.HISTORY_RECORDING_QUEUE, 0, "inf"))
        r_res = rqu.R.zadd(rqu.HISTORY_RECORDING_QUEUE, {record_key: time.time()})
        log.debug("r_res: %s", r_res) 
        log.debug("R.zcount(rqu.HISTORY_RECORDING_QUEUE): %s", rqu.R.zcount(rqu.HISTORY_RECORDING_QUEUE, 0, "inf"))
        r_res = rqu.R.expire(record_key, rqu.STANDARD_EXPIRATION)
        log.debug("r_res: %s", r_res) 
        # r_res= rqu.R.zrem(rqu.HISTORY_QUEUE, face_record['face_key'])
        # log.debug("r_res: %s", r_res) 

    def _update_data(self, current_record, new_record):
        """Updates a record with the new information. When a new frame of a person
comes, we update the current record with a new timestamp and (if the new face
image is larger) face image."""
        try:
            record_key = current_record['record_key'].decode('utf-8')
        except:
            record_key = current_record['record_key']
        log.debug("record_key: %s", record_key) 

        current_record['timestamp'] = new_record['timestamp']
        current_record['count'] += 1
        current_record['encoding'] = new_record['encoding']

        enc_index = current_record['enc_index']
        log.debug("enc_index: %s", enc_index)

        self.current_encodings[enc_index] = rqu.make_array(
            new_record['encoding'], np.float32, (1, FEATURE_SIZE))
        # Check the current face size and previous one
        if current_record['face_size'] < new_record['face_size']:
            current_record['face_x'] = new_record['face_x']
            current_record['face_y'] = new_record['face_y']
            current_record['face_w'] = new_record['face_w']
            current_record['face_h'] = new_record['face_h']
            current_record['face_size'] = new_record['face_size']
            current_record['face_key'] = new_record['face_key']
            current_record['face_i'] = new_record['face_i']

        log.debug("record_key: %s", record_key) 
        r_res = rqu.R.hmset(record_key, current_record)
        log.debug("r_res: %s", r_res) 
        log.debug("R.zcount(rqu.HISTORY_RECORDING_QUEUE): %s", rqu.R.zcount(rqu.HISTORY_RECORDING_QUEUE, 0, "inf"))
        r_res = rqu.R.zadd(rqu.HISTORY_RECORDING_QUEUE, {record_key: time.time()})
        log.debug("r_res: %s", r_res) 
        log.debug("R.zcount(rqu.HISTORY_RECORDING_QUEUE): %s", rqu.R.zcount(rqu.HISTORY_RECORDING_QUEUE, 0, "inf"))
        r_res = rqu.R.zrem(rqu.HISTORY_QUEUE, new_record['face_key'])
        log.debug("r_res: %s", r_res) 

        if (time.time() - self._db_write_timeout) > self._last_db_write:
            self.write_records()
            self._last_db_write = time.time()

    def _record_to_database(self, rec):
        """Saves a single face record to the database and related images disk."""
        person_id = int(
            rec['person_id']) if int(rec['person_id']) >= 0 else None
        tstr = dt.datetime.fromtimestamp(
            rec['timestamp']).strftime("%F-%H-%M-%S-%f")
        face_image_filename = "{}/face-{}-{}-{}.png".format(
            self.record_dir, person_id, rec["camera_id"], tstr)
        camera_image_filename = "{}/cam_img-{}-{}-{}.png".format(
            self.record_dir, rec['camera_id'], person_id, tstr)

        image_record_key = rec['record_key']
        image = rqu.get_frame_image(image_record_key, 'image')

        face_image = rqu.get_frame_image(image_record_key,
                                         'face_image_{}'.format(rec['face_i']))

        cv2.imwrite(face_image_filename, face_image)
        cv2.imwrite(camera_image_filename, image)
        # Save DB
        res = db.record_history_data(rec['camera_id'], person_id,
                                     rec['timestamp'], camera_image_filename,
                                     face_image_filename, None)

    def write_records(self):
        """Write the records that are "older" (1 minute) enough not to be updated to the database."""
        remaining_recs = []
        written_recs = []

        threshold = time.time() - self._db_write_timeout

        new_record_person = {}

        for k, rec in self._record_person.items():
            if rec['timestamp'] < threshold:
                self._record_to_database(rec)
            else:
                new_record_person[k] = rec

        self._record_person = new_record_person

        new_record_encoding = {}
        for k, rec in self._record_encoding.items():
            if rec['timestamp'] < threshold:
                self._record_to_database(rec)
            else:
                new_record_encoding[k] = rec

        # renew the current_encodings
        new_current_encodings = np.empty(
            shape=(len(new_record_encoding), FEATURE_SIZE))
        for i, k in enumerate(new_record_encoding):
            rec = new_record_encoding[k]
            new_current_encodings[i] = self.current_encodings[rec['enc_index']]
            rec['enc_index'] = i
            rqu.R.hmset(rec['record_key'], {'enc_index': i})


        updated_new_record_encoding = {}

        for k, rec in new_record_encoding.items():
            updated_new_record_encoding[rec['enc_index']] = rec

        self.current_encodings = new_current_encodings
        self._record_encoding = updated_new_record_encoding


VHR = VideoHistoryRecorder()
VIDEO_RECORD_TIMEOUT = 3600


def record_loop():
    outfile = open("/tmp/facebin-history-out-pid-{}.txt".format(os.getpid()),
                   "a")
    sys.stdout = outfile
    sys.stderr = outfile
    input_queue = rqu.HISTORY_QUEUE
    output_queue = rqu.HISTORY_RECORDING_QUEUE
    R = rqu.R
    while True:
        log.debug("alive: %s", time.time()) 
        key, score = rqu.get_next_key(input_queue, delete=True)
        if key is None:
            log.debug("key: %s", key) 
            time.sleep(0.1)
            continue
        log.debug("alive: %s", time.time()) 
        fields = [f.decode("utf-8") for f in R.hkeys(key)]
        log.debug("fields: %s", fields)
        for face_i in range(0, 10):
            k = rqu.face_image_shape_x_k(face_i)
            if k in fields:
                log.debug("Found %s", k)
                VHR.insert_update_records(key, score, face_i)
            else:
                break
        log.debug("alive: %s", time.time()) 

        # log.debug("Deleting Frame: %s (%s)", key, score)
        # R.delete(key)
