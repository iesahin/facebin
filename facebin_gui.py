import redis_queue_utils as rqu
import rq
import redis
import video_recorder
import history_recorder
import camera_reader
import camera_controller as cc
import history_dialog as hd
import person_dialog as pd
import camera_dialog as cd
import database_api as db
import face_recognition_v6 as fr6
from PySide2.QtCore import Signal, Slot
from PySide2 import QtGui as qtg
from PySide2 import QtWidgets as qtw
from PySide2 import QtCore as qtc
import multiprocessing as mp
import io
import time
import datetime as dt
import numpy as np
import cv2
import sys
from os import path
import os
import gc
from utils import *

log = init_logging()
log.debug("os.environ: %s", os.environ)


R = redis.Redis(host='localhost', port=6379)
VIDEO_RECORD_TIMEOUT = 3600


class VideoPresentationWidget(qtw.QWidget):
    def __init__(self,
                 cam,
                 keys_to_process_per_tick=1,
                 refresh_threshold=100,
                 parent=None):
        super().__init__(parent)
        self.cam = cam
        self.camera_id = cam.camera_id
        self.image = qtg.QImage()
        self.timer = qtc.QBasicTimer()
        self.keys_to_process_per_tick = keys_to_process_per_tick
        self.input_queue = rqu.RECOGNIZER_QUEUE(self.camera_id)
        # Uncomment the following line to see the raw video
        # self.input_queue = rqu.CAMERA_QUEUE
        self.output_queue = rqu.HISTORY_QUEUE
        self.refresh_threshold = refresh_threshold

    def start_timer(self):
        self.timer.start(0, self)

    def stop_timer(self):
        self.timer.stop()

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        l = R.zcount(self.input_queue, "-inf", "+inf")
        log.debug("List Length for %s: %s", self.input_queue, l)
        if l > self.refresh_threshold:
            # Delete all keys without processing
            log.warning("Refresh Threshold Reached for %s", self.camera_id)
            skip_keys = R.zrange(
                self.input_queue, 0, l-1, withscores=True)
            for key, score in skip_keys:
                R.zrem(self.input_queue, key)
                fields = R.hkeys(key)
                R.hdel(key, *fields)

        key, score = rqu.get_next_key(self.input_queue)
        if key is None:
            return

        image_data = rqu.get_frame_image(key, name='processed_image')
        if image_data is None:
            image_data = rqu.get_frame_image(key, name='image')
        qs = self.size()
        image_data = cv2.resize(image_data, (qs.width(), qs.height()))
        self.image = get_qimage(image_data)
        # log.debug("R.zcount(self.output_queue, 0, 'inf'): %s", R.zcount(self.output_queue, 0, "inf"))
        R.zrem(self.input_queue, key)
        R.zadd(self.output_queue, {key: score})
        # log.debug("R.zcount(self.output_queue, 0, 'inf'): %s",
        #            R.zcount(self.output_queue, 0, "inf"))
        self.update()

    def paintEvent(self, event):
        painter = qtg.QPainter(self)
        painter.drawImage(0, 0, self.image)
        # del self.image
        # self.image = qtg.QImage()


class FaceRecognitionWidget(qtw.QWidget):
    def __init__(self, camera_controller, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.camera_controller = camera_controller
        self.video_device = camera_controller.device
        self.recognizer = fr6.FaceRecognizer_v6()
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.text_color = (255, 255, 0)
        self.rectangle_color = (255, 0, 255)

        self.image = qtg.QImage()
        self.timer = qtc.QBasicTimer()
        self.camera = cv2.VideoCapture(self.video_device)

    def start_timer(self):
        self.timer.start(0, self)

    def stop_timer(self):
        self.timer.stop()

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        before_read = time.time()
        read, image = self.camera.read()
        after_read = time.time()
        log.debug("camera.read() time: %s", after_read - before_read)
        log.debug("image.shape: %s", image.shape)
        log.debug("read: %s", read)
        if read:
            self.process_image(image)
        else:
            log.warning("Cannot Read From %s:", self.camera_controller)
        after_process = time.time()
        log.debug("after_process-after_read: %s", after_process - after_read)

    def add_frame_to_redis(self, image_data):
        t = time.time()
        key = 'frame:{}'.format(t)
        log.debug("key: %s", key)
        filename = video_recorder.get_video_filename(
            t, self.camera_controller.camera_id)
        # memfile = io.BytesIO()
        # we can use savez here if memory becomes an issue, I prefer speed
        # np.save(memfile, image_data)
        R.hmset(
            key,
            {
                'time': t,
                'camera_id': self.camera_controller.camera_id,
                # 'image_string': memfile.getvalue(),
                'image_data': image_data.tostring(),
                'image_shape_x': image_data.shape[0],
                'image_shape_y': image_data.shape[1],
                'image_shape_z': image_data.shape[2],
                'image_dtype': str(image_data.dtype),
                'filename': filename
            })
        R.rpush('framekeys', key)
        log.debug("R.llen(framekeys): %s", R.llen('framekeys'))
        return filename

    def process_image(self, image_data):
        video_filename = self.add_frame_to_redis(image_data)
        faces = self.recognizer.predict_faces(image_data)
        # image_data_processed = image_data.copy()
        log.debug("faces: %s", faces)
        for coords, person_id in faces:
            log.debug(coords)
            log.debug(person_id)
            (x, y, w, h) = coords
            db_result = []
            if person_id is not None:
                person_id = int(person_id)
                db_result = db.person_by_id(person_id)
            log.debug("db_result: %s", db_result)
            if db_result == []:
                name = "Unknown Unknown"
                title = "Unknown"
                notes = ""
            else:
                assert (len(db_result) == 1)
                (person_id_, name, title, notes) = db_result[0]
            log.debug("person_id: {} title: {} name: {}".format(
                person_id, title, name))
            face_image = image_data[x:(x + w), y:(y + h)]

            history_queue.enqueue(history_recorder.record, time.time(),
                                  self.camera_controller.camera_id, person_id,
                                  image_data, face_image, video_filename)

            # SYNC CALL FOR DEBUG history_recorder.record(time.time(),
            # self.camera_controller.camera_id, person_id, image_data,
            # face_image, video_filename)

            cv2.putText(image_data, name, (y, x - 5), self.font, 1.0,
                        self.text_color, 2)
            cv2.rectangle(image_data, (y, x), (y + h, x + w),
                          self.rectangle_color, 2)
        qs = self.size()
        image_data = cv2.resize(image_data, (qs.height(), qs.width()))
        self.image = get_qimage(image_data)
        # if self.image.size() != self.size():
        #     self.setFixedSize(self.image.size())

        self.update()

    def paintEvent(self, event):
        painter = qtg.QPainter(self)
        painter.drawImage(0, 0, self.image)
        del self.image
        self.image = qtg.QImage()


class HistoryTable(qtw.QTableWidget):
    def __init__(self, parent=0):
        super().__init__(parent)
        self.main_form = parent
        self.setColumnCount(4)
        self._face_image_w = 224
        self._face_image_h = 224
        self.unknown_face_image = qtg.QImage("default_profile_400x400.png")
        self.unknown_face_image = self.unknown_face_image.scaled(
            self._face_image_w, self._face_image_h)
        self._history_records = {}
        self._history_keys = {}
        self._dataset_images = {}
        self._camera_images = {}
        self._record_access_timestamps = {}

        self.MINIMUM_FRAME_COUNT_TO_SHOW = 3
        self.input_queue = rqu.HISTORY_RECORDING_QUEUE
        self.timerId = self.startTimer(1000)
        # self.init_timers()
        self.update_table()

    def stop_timer(self):
        self.timer.stop()

    def init_timers(self):
        self.timer = qtc.QTimer(self)
        self.timer.timeout.connect(self.check_history)
        self.timer.start(1000)

        log.debug("Initialized: self.timer: %s", self.timer)

    def add_key(self, key, score):
        current_data = rqu.get_frame(key)

    def update_table(self):

        self.setRowCount(len(self._history_records) * 3)
        self.verticalHeader().setDefaultSectionSize(self._face_image_h // 3)
        self.horizontalHeader().setDefaultSectionSize(self._face_image_w)

        log.debug("len(self._history_records): %s", len(self._history_records))
        log.debug("self._history_records: %s", self._history_records)

        for i, k in enumerate(self._history_records):
            r = self._history_records[k]
            r1 = i * 3
            r2 = i * 3 + 1
            r3 = i * 3 + 2
            log.debug("r.keys(): %s", r.keys())
            person_id = int(r['person_id'])
            log.debug("r['person_id']: %s", r['person_id'])
            log.debug("person_id: %s", person_id)
            if 'name' not in r:
                if person_id < 0:
                    title = ""
                    name = "Unknown Unknown"
                    notes = ""
                else:
                    person_rec = db.person_by_id(person_id)[0]
                    log.debug("person_rec: %s", person_rec)
                    (_, name, title, notes) = person_rec

                r['name'] = name
                r['title'] = title
                r['notes'] = notes

            log.debug("r['name']: %s", r['name'])
            if person_id >= 0:
                if k not in self._dataset_images:
                    log.debug("r['feature_id']: %s", r['feature_id'])
                    di = db.person_face_image_by_feature_id(
                        int(r['feature_id']))[0][2]
                    log.debug("di: %s", di)
                    self._dataset_images[k] = di
                    r['dataset_image'] = di
                else:
                    r['dataset_image'] = self._dataset_images[k]
            else:
                r['dataset_image'] = self.unknown_face_image
                log.debug("r['dataset_image']: %s", r['dataset_image'])

            if k in self._camera_images:
                ci = self._camera_images[k]
                r['camera_image'] = ci
            else:
                face_key = r['face_key'].decode("utf-8")
                log.debug("face_key: %s", face_key)
                face_i = int(r['face_i'])
                log.debug("face_i: %s", face_i)
                image_data = rqu.get_frame_image(
                    face_key, name=rqu.face_image_k(face_i))
                log.debug("image_data.shape: %s", image_data.shape)
                image_data = cv2.resize(
                    image_data, (self._face_image_w, self._face_image_h))
                log.debug("image_data.shape: %s", image_data.shape)
                ci = get_qimage(image_data)
                r['camera_image'] = ci
                self._camera_images[k] = ci

            log.debug("r['camera_image'].size(): %s", r['camera_image'].size())
            name_label = qtw.QLabel(r['name'])
            log.debug("name_label: %s", name_label)
            log.debug("r['camera_id']: %s", r['camera_id'])
            camera_label = qtw.QLabel(r['camera_id'].decode('utf-8'))
            log.debug("camera_label: %s", camera_label)
            ts_text = dt.datetime.fromtimestamp(
                float(r['timestamp'])).strftime("%F %T")
            ts_label = qtw.QLabel(ts_text)
            log.debug("ts_label: %s", ts_label)
            person_image_label = qtw.QLabel()
            log.debug("type(r['camera_image']): %s", type(r['camera_image']))
            person_image_label.setPixmap(qtg.QPixmap(r['camera_image']))
            log.debug("person_image_label: %s", person_image_label)
            db_image_label = qtw.QLabel()
            log.debug("type(r['dataset_image']): %s", type(r['dataset_image']))
            db_image_label.setPixmap(qtg.QPixmap(r['dataset_image']))
            log.debug("db_image_label: %s", db_image_label)
            log.debug("r.keys(): %s", r.keys())

            self._record_access_timestamps[k] = time.time()

            # buttons

            add_photo_button = qtw.QPushButton("Add Photo to Dataset", self)
            add_photo_button.setFlat(True)
            add_photo_button.clicked.connect(self.add_photo_button_callback)
            add_photo_button.record_key = k

            add_person_button = qtw.QPushButton("Add Person", self)
            add_person_button.setFlat(True)
            add_person_button.clicked.connect(self.add_person_button_callback)
            add_person_button.record_key = k
            if person_id > 0:
                add_person_button.setEnabled(False)

            details_button = qtw.QPushButton("Details", self)
            details_button.setFlat(True)
            details_button.clicked.connect(self.details_button_callback)
            details_button.record_key = k
            if person_id < 0:
                details_button.setEnabled(False)

            self.setCellWidget(r1, 0, person_image_label)
            self.setSpan(r1, 0, 3, 1)
            self.setCellWidget(r1, 2, db_image_label)
            self.setSpan(r1, 2, 3, 1)

            self.setCellWidget(r1, 1, name_label)
            self.setCellWidget(r2, 1, camera_label)
            self.setCellWidget(r3, 1, ts_label)

            self.setCellWidget(r1, 3, add_photo_button)
            self.setCellWidget(r2, 3, add_person_button)
            self.setCellWidget(r3, 3, details_button)

        self.update()

    def timerEvent(self, event):
        log.debug("event.timerId(): %s", event.timerId())
        log.debug("self.timerId: %s", self.timerId)
        if (event.timerId() != self.timerId):
            return

        log.debug("R.zcount(rqu.HISTORY_RECORDING_QUEUE): %s",
                  R.zcount(rqu.HISTORY_RECORDING_QUEUE, 0, "inf"))
        history_keys = R.zrevrange(rqu.HISTORY_RECORDING_QUEUE, 0, 1000, True)
        log.debug("history_keys: %s", history_keys)
        # changed_keys = {}
        # for k, s in history_keys:
        #     if k not in self._history_keys:
        #         changed_keys[k] = s
        #     elif self._history_keys[k] < s:
        #         changed_keys[k] = s

        # log.debug("changed_keys: %s", changed_keys)
        self._history_records = {}
        for k, s in history_keys:
            log.debug("k: %s", k)
            log.debug("s: %s", s)
            self._history_keys[k] = s
            f = rqu.get_frame(k)
            log.debug("f.keys(): %s", f.keys())
            if int(f['count']) >= self.MINIMUM_FRAME_COUNT_TO_SHOW:
                # if k in self._history_records:
                #     self._history_records[k].update(rqu.fix_keys(f))
                # else:
                fixed = rqu.fix_keys(f)
                log.debug("fixed: %s", fixed)
                self._history_records[k] = fixed

        self.update_table()

    def add_photo_button_callback(self):
        button = self.sender()
        record_key = button.record_key
        log.debug("record_key: %s", record_key)
        r = self._history_records[record_key]
        log.debug("r.keys(): %s", r.keys())
        person_id = int(r['person_id'])
        log.debug("person_id: %s", person_id)
        random_image_filename = '/tmp/facebin-img-{}.png'.format(
            random.randint(10000, 100000))
        log.debug("random_image_filename: %s", random_image_filename)
        camera_image = self._camera_images[record_key]
        camera_image.save(random_image_filename)
        log.debug("camera_image.size(): %s", camera_image.size())
        new_person_id = pd.PersonListDialog.SelectPerson(self)
        log.debug("person_id: %s", person_id)
        log.debug("new_person_id: %s", new_person_id)

        log.debug("random_image_filename: %s", random_image_filename)
        log.debug("self.main_form: %s", self.main_form)
        add_person_res = pd.ImageListDialog.AddPersonImage(
            new_person_id, random_image_filename, parent=self)
        log.debug("add_person_res: %s", add_person_res)
        if add_person_res:
            self.main_form.check_recognizer_processes(force_restart=True)

    def add_person_button_callback(self):
        button = self.sender()
        record_key = button.record_key
        r = self._history_records[record_key]
        log.debug("r.keys(): %s", r.keys())
        random_image_filename = '/tmp/facebin-img-{}.png'.format(
            random.randint(10000, 100000))
        log.debug("random_image_filename: %s", random_image_filename)
        camera_image = self._camera_images[record_key]
        camera_image.save(random_image_filename)

        new_person_id = pd.PersonDetailsDialog.AddPerson(self)
        log.debug("new_person_id: %s", new_person_id)
        if new_person_id is not None:
            add_person_res = pd.ImageListDialog.AddPersonImage(
                new_person_id, random_image_filename, parent=self)
            log.debug("add_person_res: %s", add_person_res)
            if add_person_res:
                self.main_form.check_recognizer_processes(force_restart=True)

    def details_button_callback(self):
        button = self.sender()
        record_key = button.record_key
        r = self._history_records[record_key]
        person_id = int(r['person_id'])
        pd.PersonDetailsDialog.ShowPerson(person_id, self)


class MainWidget(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.camera_controllers = cc.get_camera_controllers()
        self.camera_controllers = {
            k: c
            for k, c in self.camera_controllers.items() if c.device != ''
        }

        self.run_button = qtw.QPushButton('Start')
        self.camera_configuration_button = qtw.QPushButton('Camera Config')
        self.camera_configuration_button.clicked.connect(
            self.camera_config_dialog)
        self.history_button = qtw.QPushButton('History')
        self.history_button.clicked.connect(self.history_dialog)
        self.people_button = qtw.QPushButton('People')
        self.people_button.clicked.connect(self.people_dialog)
        self.close_button = qtw.QPushButton('Close')
        self.close_button.clicked.connect(self.close)

        self.history_table = HistoryTable(self)

        self.recognizers_per_camera = 1

        self.presenters = []
        log.debug("Cameras: {}".format(self.camera_controllers))
        for cam_id, cam in self.camera_controllers.items():
            cam.run_command()
            vpw = VideoPresentationWidget(
                cam,
                keys_to_process_per_tick=1,
                refresh_threshold=100,
                parent=self)
            self.presenters.append(vpw)
            self.run_button.clicked.connect(vpw.timer.start)

        # for cam_id, cam in self.camera_controllers.items():
        #     log.debug("Running for {}: {}".format(cam.name, cam.command))
        #     cam.run_command()
        #     frw = FaceRecognitionWidget(camera_controller=cam, parent=self)
        #     self.recognizers.append(frw)
        #     self.run_button.clicked.connect(frw.start_timer)

        # Connect the image data signal and slot together
        # image_data_slot = self.face_detection_widget.image_data_slot
        # self.record_video.image_data.connect(image_data_slot)
        # #
        # connect the run button to the start recording slot
        # self.run_button.clicked.connect(self.record_video.start_recording)

        # Create and set the layout

        log.debug("len(self.presenters): %s", len(self.presenters))

        if len(self.presenters) == 0:
            qtw.QMessageBox.warning(self, "No Cams", "No Cameras Are Present")
            self.camera_config_dialog()
        elif len(self.presenters) > 1:
            self.setLayout(self.layout_for_4_cameras())
        else:
            self.setLayout(self.layout_for_1_camera())

        for frw in self.presenters:
            frw.start_timer()

        self.init_worker_processes()

    def init_worker_processes(self):
        # Init Camera Readers

        R.flushall()

        self.camera_reader_processes = {}

        for k, cc in self.camera_controllers.items():
            self.camera_reader_processes[k] = mp.Process(
                target=camera_reader.reader_loop,
                args=(cc.camera_id, cc.device))
            self.camera_reader_processes[k].start()

        self.recognizer_processes = {}
        for k in range(
                len(self.camera_controllers) * self.recognizers_per_camera):
            self.recognizer_processes[k] = mp.Process(
                target=fr6.face_recognition_loop)
            self.recognizer_processes[k].start()
            log.debug("self.recognizer_processes:", self.recognizer_processes)

        self.history_processes = {}
        for k, cc in self.camera_controllers.items():
            self.history_processes[k] = mp.Process(
                target=history_recorder.record_loop)
            self.history_processes[k].start()

        self.heartbeat_timer = qtc.QBasicTimer()
        # we check the status of these processes every second
        self.heartbeat_timer.start(1000, self)

    def check_camera_processes(self, force_restart=False):
        for cpk, cp in self.camera_reader_processes.items():
            if not cp.is_alive():
                log.warning(
                    "Camera Process %s with PID %s is dead. Restarting.", cpk,
                    cp.pid)
                cc = self.camera_controllers[cpk]
                self.camera_reader_processes[cpk] = mp.Process(
                    target=camera_reader.reader_loop,
                    args=(cc.camera_id, cc.device))
                self.camera_reader_processes[cpk].start()
            elif force_restart:
                log.warning(
                    "Force Restarting Camera Process %s with PID %s.", cpk,
                    cp.pid)
                cp.terminate()
                cp.join()
                cc = self.camera_controllers[cpk]
                self.camera_reader_processes[cpk] = mp.Process(
                    target=camera_reader.reader_loop,
                    args=(cc.camera_id, cc.device))
                self.camera_reader_processes[cpk].start()
            else:
                log.debug("Camera Process %s with PID %s is alive", cpk,
                          cp.pid)

        log.debug("Camera Queue Length: %s",
                  rqu.queue_length(rqu.CAMERA_QUEUE))
        for cpk in self.camera_reader_processes:
            log.debug("Recognizer(%s) Queue Length: %s", cpk,
                      rqu.queue_length(rqu.RECOGNIZER_QUEUE(cpk)))

    def check_recognizer_processes(self, force_restart=False):

        for rk, rp in self.recognizer_processes.items():
            if not rp.is_alive():
                log.warning(
                    "Recognizer Process %s with PID %s is dead. Restarting.",
                    rk, rp.pid)
                self.recognizer_processes[rk] = mp.Process(
                    # target=fr.face_recognition_loop)
                    target=fr6.face_recognition_loop)
                self.recognizer_processes[rk].start()
            elif force_restart:
                log.warning(
                    "Force Restarting Recognizer Process %s with PID %s",
                    rk, rp.pid)
                rp.terminate()
                rp.join()
                self.recognizer_processes[rk] = mp.Process(
                    # target=fr.face_recognition_loop)
                    target=fr6.face_recognition_loop)
                self.recognizer_processes[rk].start()
            else:
                log.debug("Recognizer Process %s with PID %s is alive", rk,
                          rp.pid)

    def check_history_processes(self, force_restart=False):
        for hk, hp in self.history_processes.items():
            if not hp.is_alive():
                log.warning(
                    "History Process %s with PID %s is dead. Restarting.", hk,
                    hp.pid)
                self.history_processes[hk] = mp.Process(
                    target=history_recorder.record_loop)
                self.history_processes[hk].start()
            elif force_restart:
                log.warning(
                    "Force Restarting History Process %s with PID %s", hk,
                    hp.pid)
                hp.terminate()
                hp.join()
                self.history_processes[hk] = mp.Process(
                    target=history_recorder.record_loop)
                self.history_processes[hk].start()
            else:
                log.debug("History Process %s with PID %s is alive", hk,
                          hp.pid)

        log.debug("History Queue Length: %s",
                  rqu.queue_length(rqu.HISTORY_QUEUE))
        log.debug("History Recording Queue Length: %s",
                  rqu.queue_length(rqu.HISTORY_RECORDING_QUEUE))

    def timerEvent(self, event):
        if (event.timerId() != self.heartbeat_timer.timerId()):
            return

        self.check_camera_processes(force_restart=False)

        self.check_recognizer_processes(force_restart=False)

        self.check_history_processes(force_restart=False)

    def closeEvent(self, event):
        log.debug("camera_controllers: %s", self.camera_controllers)

        for cam_id, cam in self.camera_controllers.items():
            log.debug("cam_id: %s", cam_id)
            cam.kill_command()
            log.debug("cam.process: %s", cam.process)

        # for frw in self.presenters:
        #     frw.stop_timer()

        for cpk, cp in self.camera_reader_processes.items():
            cp.terminate()
            cp.join()

        for rk, rp in self.recognizer_processes.items():
            rp.terminate()
            rp.join()

        for hk, hp in self.history_processes.items():
            hp.terminate()
            hp.join()

        event.accept()

    def history_dialog(self):
        history_dialog = hd.HistoryDialog(self.helper_recognizer, self)
        history_dialog.exec_()

    def camera_config_dialog(self):
        dialog = cd.CameraConfigurationDialog(self)
        dialog.exec_()

    def people_dialog(self):
        self.person_dialog_process = mp.Process(target=pd.main)
        self.person_dialog_process.start()

    def layout_for_buttons(self):
        layout = qtw.QHBoxLayout()
        layout.addWidget(self.run_button, 1)
        layout.addSpacing(1)

        layout.addWidget(self.camera_configuration_button, 1)
        layout.addSpacing(1)

        # layout.addWidget(self.history_button, 1)
        # layout.addSpacing(1)

        layout.addWidget(self.people_button, 1)
        layout.addSpacing(1)

        layout.addWidget(self.close_button, 1)
        layout.addSpacing(1)

        return layout

    def layout_for_1_camera(self):
        layout = qtw.QVBoxLayout()
        history_camera_layout = qtw.QHBoxLayout()
        camera_layout = qtw.QHBoxLayout()
        camera_layout.addWidget(self.presenters[0])
        history_camera_layout = qtw.QHBoxLayout()
        history_camera_layout.addLayout(camera_layout, 75)
        history_camera_layout.addWidget(self.history_table, 25)
        layout.addLayout(history_camera_layout)
        layout.addLayout(self.layout_for_buttons())
        return layout

    def layout_for_4_cameras(self):
        layout = qtw.QVBoxLayout()
        camera_layout = qtw.QGridLayout()
        try:
            camera_layout.addWidget(self.presenters[0], 0, 0, 1, 1)
            camera_layout.addWidget(self.presenters[1], 0, 1, 1, 1)
            camera_layout.addWidget(self.presenters[2], 1, 0, 1, 1)
            camera_layout.addWidget(self.presenters[3], 1, 1, 1, 1)
        except IndexError:
            pass

        history_camera_layout = qtw.QHBoxLayout()
        history_camera_layout.addLayout(camera_layout, 75)
        history_camera_layout.addWidget(self.history_table, 25)
        layout.addLayout(history_camera_layout, 2)
        layout.addLayout(self.layout_for_buttons(), 0)
        return layout


def main():
    mp.set_start_method('spawn')
    app = qtw.QApplication(sys.argv)
    main_window = qtw.QMainWindow()
    main_widget = MainWidget()
    main_widget.setAttribute(qtc.Qt.WA_DeleteOnClose, True)
    main_window.setAttribute(qtc.Qt.WA_DeleteOnClose, True)
    main_window.setCentralWidget(main_widget)
    main_widget.close_button.clicked.connect(main_window.close)
    main_window.showFullScreen()
    app_return = app.exec_()
    log.debug("app_return: %s", app_return)
    return sys.exit(app_return)


def profile_this():
    import cProfile
    import datetime
    import pstats

    filename = datetime.datetime.now().strftime(
        "/tmp/facebin-profile-result-%F-%H-%M-%S-%f.prof")
    cProfile.run('main()', filename=filename)
    p = pstats.Stats(filename)
    p.sort_stats('cumulative').print_stats(100)


if __name__ == '__main__':
    main()
