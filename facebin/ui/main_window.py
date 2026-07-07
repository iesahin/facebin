"""Facebin desktop GUI.

Shows live (annotated) camera streams, the recent-appearances table, and
dialogs for managing cameras, people, and history.  Worker processes are
supervised by :class:`facebin.server.FacebinServer`; when the GUI is
started through `facebin run` it receives the running server instance and
can restart recognizers after dataset changes.

Start it with `facebin run` (server + GUI) or `facebin gui` (GUI only).
"""

import datetime as dt
import multiprocessing as mp
import os
import random
import sys
import time

import cv2
import numpy as np

import facebin.server.redis_queue_utils as rqu
import facebin.server.camera_controller as cc
import facebin.server.database_api as db
import facebin.ui.history_dialog as hd
import facebin.ui.person_dialog as pd
import facebin.ui.camera_dialog as cd
from facebin.config import load_config
from facebin.server import FacebinServer
from facebin.server.utils import init_logging
from facebin.ui.qt_compat import qtc, qtw, qtg, Signal, Slot, exec_app
from facebin.ui.qt_utils import get_qimage

log = init_logging()

R = rqu.R
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
        # pylint: disable=no-member
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


class HistoryTable(qtw.QTableWidget):
    def __init__(self, parent=0):
        super().__init__(parent)
        self.main_form = parent
        self.setColumnCount(4)
        self._face_image_w = 224
        self._face_image_h = 224
        self.unknown_face_image = qtg.QImage(
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "resources",
                "default_profile_400x400.png"))
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
    """Main window: camera views, appearance table, and control buttons.

    Args:
        config: The loaded :class:`facebin.config.Config`.
        server: The :class:`facebin.server.FacebinServer` supervising the
            worker processes, when this GUI owns it (``facebin run``).
            None when the GUI attaches to an externally managed server
            (``facebin gui``).
    """

    def __init__(self, config=None, server=None, parent=None):
        super().__init__(parent)

        self.config = config if config is not None else load_config()
        self.server = server
        self.helper_recognizer = None

        self.camera_controllers = cc.get_camera_controllers(self.config)
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

        if self.server is not None and not self.server._running:
            self.server.start()

    def check_recognizer_processes(self, force_restart=False):
        """Restart recognizers (e.g. after the dataset changed)."""
        if self.server is None:
            log.warning(
                "The GUI does not manage the worker processes; restart the "
                "recognizers from the `facebin server` side to pick up "
                "dataset changes.")
            return
        self.server.check_recognizer_processes(force_restart=force_restart)

    def closeEvent(self, event):
        for cam_id, cam in self.camera_controllers.items():
            cam.kill_command()

        if self.server is not None:
            self.server.stop()

        event.accept()

    def history_dialog(self):
        history_dialog = hd.HistoryDialog(self.helper_recognizer, self)
        history_dialog.exec()

    def camera_config_dialog(self):
        dialog = cd.CameraConfigurationDialog(self)
        dialog.exec()

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


def main(config=None, server=None):
    """Start the GUI; returns the Qt application exit code."""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if config is None:
        config = load_config()
    rqu.configure(config.redis)
    db.configure(config)
    app = qtw.QApplication(sys.argv)
    main_window = qtw.QMainWindow()
    main_widget = MainWidget(config=config, server=server)
    main_widget.setAttribute(qtc.Qt.WA_DeleteOnClose, True)
    main_window.setAttribute(qtc.Qt.WA_DeleteOnClose, True)
    main_window.setCentralWidget(main_widget)
    main_widget.close_button.clicked.connect(main_window.close)
    main_window.showFullScreen()
    app_return = exec_app(app)
    log.debug("app_return: %s", app_return)
    return app_return


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
    sys.exit(main())
