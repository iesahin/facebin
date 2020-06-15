import sys

from PySide2 import QtCore as qtc
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2.QtCore import Signal, Slot

import datetime as dt

import numpy as np
import database_api as db
import dataset_manager_v3 as dm3
import face_detection as fd
import face_recognition_v6 as fr6
import camera_controller as cc
import person_dialog as pd

import show_image_dialog as sid

from utils import *

log = init_logging()


class HistoryPersonDetailDialog(qtw.QDialog):

    ADD_NEW_PERSON = -1
    EDIT_STATE = 10
    SAVE_STATE = 20

    def __init__(self, history_record, face_recognizer, parent=None):
        super(HistoryPersonDetailDialog, self).__init__(parent)
        self.hr = history_record
        self.face_recognizer = face_recognizer
        if os.path.exists(self.hr.face_image_filename):
            self.face_label = qtw.QLabel()
            self.face_image = qtg.QImage(self.hr.face_image_filename)
            self.face_label.setPixmap(qtg.QPixmap.fromImage(self.face_image))
        else:
            self.face_label = qtw.QLabel("Cannot Load: {}".format(
                self.hr.face_image_filename))

        if os.path.exists(self.hr.camera_image_filename):
            self.camera_label = qtw.QLabel()
            self.camera_image = qtg.QImage(self.hr.camera_image_filename)
            self.camera_label.setPixmap(
                qtg.QPixmap.fromImage(self.camera_image))
        else:
            self.camera_label = qtw.QLabel("Cannot Load: {}".format(
                self.hr.camera_image_filename))

        self.image_layout = qtw.QHBoxLayout()
        self.image_layout.addWidget(self.face_label)
        self.image_layout.addWidget(self.camera_label)

        self.person_name_label = qtw.QLabel(self.hr.person_name)
        self.camera_name_label = qtw.QLabel("Camera {}".format(
            self.hr.camera_id))
        t = dt.datetime.utcfromtimestamp(self.hr.time).strftime("%F %H:%M:%S")
        self.timestamp_label = qtw.QLabel(t)

        self.info_layout = qtw.QHBoxLayout()
        self.info_layout.addWidget(self.person_name_label)
        self.info_layout.addWidget(self.camera_name_label)
        self.info_layout.addWidget(self.timestamp_label)

        self.edit_save_button = qtw.QPushButton("Edit")
        self.edit_save_button.edit_save_state = self.EDIT_STATE
        self.edit_save_button.clicked.connect(self.edit_save)
        self.person_cb = qtw.QComboBox()

        self.person_list = db.person_list()
        self.person_cb.addItem(
            "{} - {}".format(self.hr.person_title, self.hr.person_name),
            self.hr.person_id)
        for p in self.person_list:
            person_id = p[0]
            title = p[1]
            name = p[2]
            self.person_cb.addItem("{} - {}".format(title, name), person_id)
        self.person_cb.addItem("Add New...",
                               HistoryPersonDetailDialog.ADD_NEW_PERSON)
        self.person_cb.setEnabled(False)
        self.add_to_dataset_button = qtw.QPushButton("Add to Dataset")
        self.add_to_dataset_button.clicked.connect(self.add_to_dataset)
        self.close_button = qtw.QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        self.button_layout = qtw.QHBoxLayout()
        self.button_layout.addWidget(self.edit_save_button)
        self.button_layout.addWidget(self.person_cb)
        self.button_layout.addWidget(self.add_to_dataset_button)
        self.button_layout.addWidget(self.close_button)

        main_layout = qtw.QVBoxLayout()
        main_layout.addLayout(self.image_layout)
        main_layout.addLayout(self.info_layout)
        main_layout.addLayout(self.button_layout)

        self.setLayout(main_layout)

    def edit_save(self):

        if self.edit_save_button.edit_save_state == self.EDIT_STATE:
            self.person_cb.setEnabled(True)
            self.edit_save_button.setText("Save")
            self.edit_save_button.edit_save_state = self.SAVE_STATE
        elif self.edit_save_button.edit_save_state == self.SAVE_STATE:
            self.save_new_name()
            self.person_cb.setEnabled(False)
            self.edit_save_button.setText("Edit")
            self.edit_save_button.edit_save_state = self.EDIT_STATE

    def save_new_name(self):
        new_id = self.person_cb.currentData()
        if new_id != self.hr.person_id:
            if new_id == HistoryPersonDetailDialog.ADD_NEW_PERSON:
                new_id = pd.PersonDetailsDialog.AddPerson(self)
            # Note that new_id may be None if AddPerson dialog doesn't add this.
            # This runs regardless of new_id added via AddPerson or selected with the person_cb
            if new_id is not None:
                db.update_history_person(self.hr.row_id, self.hr.person_id,
                                         new_id)
                hr_list = db.history_by_id(self.hr.row_id)
                assert (len(hr_list) == 1)
                self.hr = hr_list[0]

    def add_to_dataset(self):
        if self.hr.person_id < 0:
            qtw.QMessageBox.warning(
                self, "Unknown Person",
                "You should record the name to add to dataset")
            return -1
        else:
            face_image = cv2.imread(self.hr.face_image_filename)
            self.face_recognizer.add_face_to_dataset(face_image,
                                                     self.hr.person_id)
            log.debug("self.face_recognizer.dataset.person_ids: %s",
                      self.face_recognizer.dataset.person_ids)

            qtw.QMessageBox.information(
                self, "Added",
                "Added the face to dataset for person_id: {} and person_name: {}"
                .format(self.hr.person_id, self.hr.person_name))


# # C_PERSON_NAME = 1
# C_PERSON_RECORD = 1
# C_TIME = 2
# C_DETAILS_BUTTON = 3

# C_CAMERA_IMAGE = 3
# C_FACE_IMAGE = 4
# C_VIDEO_FILE = 5


class HistoryDialog(qtw.QDialog):

    PERSON_ID_ALL = -2
    PERSON_ID_UNKNOWN = -1
    CAMERA_ID_ALL = -2
    MAX_ELEMENTS = 1000

    def __init__(self, face_recognizer, parent=None):
        super(HistoryDialog, self).__init__(parent)

        self.person_cache = {}
        self.camera_cache = {}
        self.face_recognizer = face_recognizer

        self.filter_person_cb = qtw.QComboBox()
        self.filter_camera_cb = qtw.QComboBox()
        self.filter_begin_dt = qtw.QDateTimeEdit()
        self.filter_end_dt = qtw.QDateTimeEdit()

        self.filter_person_cb.currentIndexChanged.connect(self.refresh_table)
        self.filter_camera_cb.currentIndexChanged.connect(self.refresh_table)
        self.filter_begin_dt.dateTimeChanged.connect(self.refresh_table)
        self.filter_end_dt.dateTimeChanged.connect(self.refresh_table)

        self.filter_layout = qtw.QHBoxLayout()
        self.filter_layout.addWidget(self.filter_person_cb)
        self.filter_layout.addWidget(self.filter_camera_cb)
        self.filter_layout.addWidget(self.filter_begin_dt)
        self.filter_layout.addWidget(self.filter_end_dt)

        self.history_table = qtw.QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(
            ['Camera', 'Name', 'Time', 'Images', 'Video'])
        self.history_table.clicked.connect(self.table_cell_clicked)
        # self.history_table.setColumnCount(6)
        # self.history_table.setHorizontalHeaderLabels(
        #     ['Camera', 'Name', 'Time', 'Camera Image', 'Face Image', 'Video'])

        self.table_layout = qtw.QHBoxLayout()
        self.table_layout.addWidget(self.history_table)

        self.populate_filter_items()
        self.refresh_table()

        # self.checkbox_list = []
        # for i, qi in enumerate(self.image_list):
        #     pi = qtg.QPixmap(qi)
        #     checkbox = qtw.QCheckBox("Selected")
        #     checkbox.setChecked(self.selected_by_default)
        #     self.checkbox_list.append(checkbox)
        #     label = qtw.QLabel()
        #     label.setPixmap(pi)
        #     self.images_table.setCellWidget(i, 0, checkbox)
        #     self.images_table.setCellWidget(i, 1, label)
        #     self.images_table.setRowHeight(i, max_height)

        cancel_button = qtw.QPushButton("Cancel")
        ok_button = qtw.QPushButton("OK")
        cancel_button.clicked.connect(self.reject)
        ok_button.clicked.connect(self.accept)
        self.button_layout = qtw.QHBoxLayout()
        self.button_layout.addWidget(cancel_button)
        self.button_layout.addWidget(ok_button)

        main_layout = qtw.QVBoxLayout()
        main_layout.addLayout(self.filter_layout)
        main_layout.addLayout(self.table_layout)
        main_layout.addLayout(self.button_layout)

        self.setLayout(main_layout)

    def reload_records(self):
        person_id = self.filter_person_cb.currentData()
        if person_id == HistoryDialog.PERSON_ID_ALL:
            person_id = None

        camera_id = self.filter_camera_cb.currentData()
        if camera_id == HistoryDialog.CAMERA_ID_ALL:
            camera_id = None

        date_begin = self.filter_begin_dt.dateTime().toTime_t()
        log.debug("date_begin: %s", date_begin)
        date_end = self.filter_end_dt.dateTime().toTime_t()
        log.debug("date_end: %s", date_end)

        self.history_records = db.history_query(
            person_id=person_id,
            camera_id=camera_id,
            datetime_begin=date_begin,
            datetime_end=date_end,
            max_elements=self.MAX_ELEMENTS)

    def refresh_table(self):
        self.reload_records()
        self.populate_history_items()

    def open_detail_form(self, index):
        record = self.history_records[index]
        detail_form = HistoryPersonDetailDialog(
            record, self.face_recognizer, parent=self)
        detail_form.exec_()
        self.history_records[index] = detail_form.hr
        self.populate_history_items()

    def table_cell_clicked(self, model_index):
        col = model_index.column()
        row = model_index.row()

        if col in [0, 1, 2, 3]:
            self.open_detail_form(row)
        elif col == 4:
            self.video_col_clicked(row)

    def populate_history_items(self):
        # ['Camera', 'Name', 'Time', 'Face Image', 'Camera Image', 'Video'])
        # history_list.append((camera_name, person_name, time, camera_image,
        #                          face_image, video_filename))

        self.history_table.setRowCount(len(self.history_records))

        for i, hr in enumerate(self.history_records):
            # log.debug("hr: %s", hr)

            # camera_button = qtw.QPushButton(hr.camera_id, self)
            # camera_button.setFlat(True)
            # camera_button.row_id = hr.row_id
            # camera_button.index = i
            # camera_button.clicked.connect(self.open_detail_form_callback)

            # person_button = qtw.QPushButton(hr.person_name, self)
            # person_button.setFlat(True)
            # person_button.row_id = hr.row_id
            # person_button.index = i
            # person_button.clicked.connect(self.open_detail_form_callback)

            # t = dt.datetime.utcfromtimestamp(hr.time).strftime("%F %H:%M:%S")
            # time_button = qtw.QPushButton(t, self)
            # time_button.setFlat(True)
            # time_button.row_id = hr.row_id
            # time_button.index = i
            # time_button.clicked.connect(self.open_detail_form_callback)

            # open_detail_button = qtw.QPushButton("Images", self)
            # open_detail_button.setFlat(True)
            # open_detail_button.row_id = hr.row_id
            # open_detail_button.index = i
            # open_detail_button.clicked.connect(self.open_detail_form_callback)

            # video_filename = os.path.split(hr.video_filename)[1]
            # video_button = qtw.QPushButton(video_filename, self)
            # video_button.setFlat(True)
            # video_button.row_id = hr.row_id
            # video_button.index = i
            # video_button.clicked.connect(self.video_button_clicked_callback)

            self.history_table.setItem(i, 0,
                                       qtw.QTableWidgetItem(hr.camera_id))
            self.history_table.setItem(i, 1,
                                       qtw.QTableWidgetItem(hr.person_name))
            t = dt.datetime.utcfromtimestamp(hr.time).strftime("%F %H:%M:%S")
            self.history_table.setItem(i, 2, qtw.QTableWidgetItem(t))
            self.history_table.setItem(i, 3, qtw.QTableWidgetItem("Images"))
            self.history_table.setItem(i, 4,
                                       qtw.QTableWidgetItem(hr.video_filename))
            # self.history_table.setCellWidget(i, 0, camera_button)
            # self.history_table.setCellWidget(i, 1, person_button)
            # self.history_table.setCellWidget(i, 2, time_button)
            # self.history_table.setCellWidget(i, 3, open_detail_button)
            # self.history_table.setCellWidget(i, 4, video_button)

            # camera_image_button = qtw.QPushButton("", self)
            # camera_image_button.setIcon(
            #     self._get_button_icon(hr[C_CAMERA_IMAGE]))
            # camera_image_button.setFlat(True)
            # camera_image_button.row_id = i
            # camera_image_button.clicked.connect(
            #     self.camera_image_button_clicked_callback)

            # face_image_button = qtw.QPushButton("", self)
            # face_image_button.setIcon(self._get_button_icon(hr[C_FACE_IMAGE]))
            # face_image_button.setFlat(True)
            # face_image_button.row_id = i
            # face_image_button.clicked.connect(
            #     self.face_image_button_clicked_callback)
        # self.history_table.setCellWidget(i, 3, camera_image_button)
        # self.history_table.setCellWidget(i, 4, face_image_button)
        # self.history_table.setCellWidget(i, 5, video_button)

    def _open_video_file(self, filename):
        sp.run(["/usr/bin/ffplay", filename])

    def video_col_clicked(self, index):
        video_filename = self.history_records[index].video_filename
        self._open_video_file(video_filename)

    # def camera_image_button_clicked_callback(self):
    #     button = self.sender()
    #     index = button.index
    #     camera_image = self.history_records[index].camera_image_filename
    #     image_dialog = sid.ShowImageDialog(camera_image)
    #     image_dialog.exec_()

    # def face_image_button_clicked_callback(self):
    #     button = self.sender()
    #     index = button.index
    #     face_image = self.history_records[index].face_image_filename
    #     image_dialog = sid.ShowImageDialog(face_image)
    #     image_dialog.exec_()

    def _get_button_icon(self, image):
        if image is None:
            qi = qtg.QImage()
        else:
            qi = get_qimage(image)
        pi = qtg.QPixmap(qi)
        return qtg.QIcon(pi)

    def populate_filter_items(self):
        persons = db.person_list()
        log.debug(persons)
        cameras = cc.get_camera_controllers()

        end_dt = qtc.QDate.fromString(dt.datetime.today().strftime("%F"),
                                      "yyyy-MM-dd")
        one_month_ago = dt.datetime.today() - dt.timedelta(days=30)
        begin_dt = qtc.QDate.fromString(
            one_month_ago.strftime("%F"), "yyyy-MM-dd")

        log.debug(end_dt)
        log.debug(begin_dt)

        self.filter_person_cb.addItem("All", HistoryDialog.PERSON_ID_ALL)
        self.filter_person_cb.addItem("Unknown",
                                      HistoryDialog.PERSON_ID_UNKNOWN)
        self.filter_person_cb.insertSeparator(2)

        for p in persons:
            person_id = p[0]
            title = p[1]
            name = p[2]
            self.filter_person_cb.addItem("{} - {}".format(title, name),
                                          person_id)

        for cam_id, cam in cameras.items():
            cam_id = cam.camera_id
            cam_name = cam.name
            self.filter_camera_cb.addItem("{}".format(cam.name), cam_id)

        self.filter_begin_dt.setDisplayFormat("dd MMM yy")
        self.filter_end_dt.setDisplayFormat("dd MMM yy")

        self.filter_begin_dt.setDate(begin_dt)
        self.filter_end_dt.setDate(end_dt)


def main():
    app = qtw.QApplication(sys.argv)
    recognizer = fr.FaceRecognizer_v6()
    history_dialog = HistoryDialog(recognizer)
    history_dialog.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
