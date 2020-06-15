import sys

from PySide2 import QtCore as qtc
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2.QtCore import Signal, Slot

import numpy as np
import database_api as db
import dataset_manager_v3 as dm3
import face_detection as fd
import face_recognition_v6 as fr6

from utils import *

log = init_logging()


class PersonDetailsDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super(PersonDetailsDialog, self).__init__(parent)

        self.title_edit = qtw.QLineEdit("")
        self.name_edit = qtw.QLineEdit("")
        self.notes_edit = qtw.QTextEdit("")
        form_layout = qtw.QFormLayout()
        form_layout.addRow(self.tr("Title"), self.title_edit)
        form_layout.addRow(self.tr("Name"), self.name_edit)
        form_layout.addRow(self.tr("Notes"), self.notes_edit)

        self.ok_button = qtw.QPushButton(self.tr("OK"))
        self.cancel_button = qtw.QPushButton(self.tr("Cancel"))

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout = qtw.QHBoxLayout()

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        main_layout = qtw.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    @staticmethod
    def AddPerson(parent=None):
        dialog = PersonDetailsDialog(parent)
        res = dialog.exec_()
        log.debug("res: %s", res)
        if res == qtw.QDialog.Accepted:
            person_id = db.insert_person(dialog.title_edit.text(),
                                         dialog.name_edit.text(),
                                         dialog.notes_edit.toPlainText())
            return person_id
        else:
            return None

    @staticmethod
    def ShowPerson(person_id, parent=None):
        dialog = PersonDetailsDialog(parent)
        person_info = db.person_by_id(person_id)
        print(person_info)
        assert len(person_info) == 1
        dialog.name_edit.setText(person_info[0][1])
        dialog.title_edit.setText(person_info[0][2])
        dialog.notes_edit.setText(person_info[0][3])
        res = dialog.exec_()
        if res == qtw.QDialog.Accepted:
            person_id = db.update_person(person_id, dialog.title_edit.text(),
                                         dialog.name_edit.text(),
                                         dialog.notes_edit.toPlainText())
            return person_id
        else:
            return None


class ImageListDialog(qtw.QDialog):
    def __init__(self, image_list, selected_by_default=True, parent=None):
        super(ImageListDialog, self).__init__(parent)
        log.debug("len(image_list): %s", len(image_list))
        self.image_list = image_list
        self.selected_by_default = selected_by_default
        log.debug("selected_by_default: %s", selected_by_default)

        self.images_table = qtw.QTableWidget()
        self.images_table.setColumnCount(2)
        self.images_table.setHorizontalHeaderLabels(['Selected', 'Image'])
        self.images_table.setRowCount(len(self.image_list))
        max_width = max([img.width() for img in self.image_list])
        max_height = max([img.height() for img in self.image_list])
        self.images_table.setColumnWidth(1, max_width)
        self.checkbox_list = []
        for i, qi in enumerate(self.image_list):
            pi = qtg.QPixmap(qi)
            checkbox = qtw.QCheckBox("Selected")
            checkbox.setChecked(self.selected_by_default)
            self.checkbox_list.append(checkbox)
            label = qtw.QLabel()
            label.setPixmap(pi)
            self.images_table.setCellWidget(i, 0, checkbox)
            self.images_table.setCellWidget(i, 1, label)
            self.images_table.setRowHeight(i, max_height)

        cancel_button = qtw.QPushButton("Cancel")
        ok_button = qtw.QPushButton("OK")
        cancel_button.clicked.connect(self.reject)
        ok_button.clicked.connect(self.accept)
        button_layout = qtw.QHBoxLayout()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)

        main_layout = qtw.QVBoxLayout()
        main_layout.addWidget(self.images_table)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    @staticmethod
    def ShowPersonImages(person_id, selected_by_default=True, parent=None):
        dataset = dm3.DatasetManager_v3()
        images = dataset.images_by_id(person_id)
        log.debug("images: %s", images)
        image_list = [qtg.QImage(r['path']) for i, r in images.items()]
        print(image_list)
        dialog = ImageListDialog(
            image_list, selected_by_default=True, parent=parent)
        dialog.exec_()

    @staticmethod
    def AddPersonImage(person_id, new_image_path, parent=None):
        log.debug("person_id: %s", person_id)
        log.debug("new_image_path: %s", new_image_path)
        dataset = dm3.DatasetManager_v3()
        images = dataset.images_by_id(person_id)
        new_image = qtg.QImage(new_image_path)
        image_list = [new_image
                      ] + [qtg.QImage(r['path']) for i, r in images.items()]
        log.debug("image_list: %s", image_list)
        dialog = ImageListDialog(
            image_list, selected_by_default=True, parent=parent)
        dialog_res = dialog.exec_()
        log.debug("dialog_res: %s", dialog_res)
        if dialog_res == qtw.QDialog.Accepted:
            log.debug("dialog.checkbox_list[0].isChecked(): %s",
                      dialog.checkbox_list[0].isChecked())
            if dialog.checkbox_list[0].isChecked():
                dataset.add_face_file_to_person(person_id, new_image_path)
                return True
            else:
                return False

    @staticmethod
    def SelectedImages(image_list, selected_by_default=True, parent=None):
        qimage_list = [get_qimage(fi) for fi in image_list]
        dialog = ImageListDialog(qimage_list, selected_by_default, parent)
        selected_image_list = []
        if dialog.exec_() == qtw.QDialog.Accepted:
            for i, cb in enumerate(dialog.checkbox_list):
                if cb.isChecked():
                    selected_image_list.append(image_list[i])
        return selected_image_list


class PersonListDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super(PersonListDialog, self).__init__(parent)

        self.person_table = qtw.QTableWidget()
        self._init_person_table()

        self.close_button = qtw.QPushButton(self.tr("Close"))
        self.select_button = qtw.QPushButton(self.tr("Select"))
        self.close_button.clicked.connect(self.reject)
        self.select_button.clicked.connect(self.accept)
        button_layout = qtw.QHBoxLayout()
        button_layout.addWidget(self.close_button)

        main_layout = qtw.QVBoxLayout()
        main_layout.addWidget(self.person_table)
        main_layout.addLayout(button_layout)

        self.selected_person_id = None

        self.setLayout(main_layout)

    def _init_person_table(self):

        people_data = db.person_list()

        self.person_table.setColumnCount(4)
        self.person_table.setRowCount(len(people_data))
        self.person_table.setHorizontalHeaderLabels(
            ['Title', 'Name', 'Notes', ''])

        for i, (person_id, name, title, notes) in enumerate(people_data):
            select_button = qtw.QPushButton("Select", self)
            select_button.person_id = person_id
            select_button.clicked.connect(self.select_button_callback)
            self.person_table.setItem(i, 0, qtw.QTableWidgetItem(title))
            self.person_table.setItem(i, 1, qtw.QTableWidgetItem(name))
            self.person_table.setItem(i, 2, qtw.QTableWidgetItem(notes))
            self.person_table.setCellWidget(i, 3, select_button)

    def select_button_callback(self):
        button = self.sender()
        self.selected_person_id = button.person_id
        self.accept()
        self.close()

    @staticmethod
    def SelectPerson(parent=None):
        log.debug("parent: %s", parent)
        dialog = PersonListDialog(parent)
        if dialog.exec_() == qtw.QDialog.Accepted:
            log.debug("dialog.selected_person_id: %s",
                      dialog.selected_person_id)
            return dialog.selected_person_id
        else:
            return None


class PersonDialog(qtw.QDialog):
    def __init__(self, recognizer, parent=None):
        super(PersonDialog, self).__init__(parent)

        ## self.dataset = ds.DatasetManager()
        self.recognizer = recognizer
        self.dataset = recognizer.dataset
        self.person_table = qtw.QTableWidget()
        self._init_person_table()

        self.close_button = qtw.QPushButton(self.tr("Close"))
        self.close_button.clicked.connect(self.close)
        self.add_button = qtw.QPushButton(self.tr("Add"))
        self.add_button.clicked.connect(self._add_person)
        button_layout = qtw.QHBoxLayout()
        button_layout.addWidget(self.close_button)
        button_layout.addWidget(self.add_button)

        main_layout = qtw.QVBoxLayout()
        main_layout.addWidget(self.person_table)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _get_person_icon(self, person_id):
        person_images = self.dataset.images_by_id(person_id)
        print(len(person_images))
        if len(person_images) > 0:
            k = list(person_images.keys())[0]
            first_img = person_images[k]
            print(first_img)
            qi = qtg.QImage(first_img['path'])
        else:
            qi = qtg.QImage("default_profile_400x400.png")
        pi = qtg.QPixmap(qi)
        return qtg.QIcon(pi)

    def _init_person_table(self):

        people_data = db.person_list()
        print(people_data)

        self.person_table.setColumnCount(6)
        self.person_table.setRowCount(len(people_data))
        self.person_table.setHorizontalHeaderLabels(
            ['Title', 'Name', 'Details', 'Images', '', ''])

        details_buttons = {}
        images_buttons = {}
        add_video_buttons = {}
        add_image_buttons = {}

        for i, (person_id, name, title, notes) in enumerate(people_data):
            print(i)
            details_buttons[i] = qtw.QPushButton("Details", self)
            details_buttons[i].setFlat(True)
            print(person_id)
            details_buttons[i].person_id = person_id
            details_buttons[i].clicked.connect(self.details_button_callback)
            images_buttons[i] = qtw.QPushButton("Images", self)
            images_buttons[i].setFlat(True)
            images_buttons[i].setIcon(self._get_person_icon(person_id))
            images_buttons[i].person_id = person_id
            images_buttons[i].clicked.connect(self.images_button_callback)

            add_video_buttons[i] = qtw.QPushButton("Add Video", self)
            add_video_buttons[i].setFlat(True)
            add_video_buttons[i].person_id = person_id
            add_video_buttons[i].clicked.connect(
                self.add_video_button_callback)

            add_image_buttons[i] = qtw.QPushButton("Add Image", self)
            add_image_buttons[i].setFlat(True)
            add_image_buttons[i].person_id = person_id
            add_image_buttons[i].clicked.connect(
                self.add_image_button_callback)

            self.person_table.setItem(i, 0, qtw.QTableWidgetItem(title))
            self.person_table.setItem(i, 1, qtw.QTableWidgetItem(name))
            self.person_table.setCellWidget(i, 2, details_buttons[i])
            self.person_table.setCellWidget(i, 3, images_buttons[i])
            self.person_table.setCellWidget(i, 4, add_video_buttons[i])
            self.person_table.setCellWidget(i, 5, add_image_buttons[i])

    def _add_person(self):
        person_id = PersonDetailsDialog.AddPerson(self)

        if person_id is not None:
            self.person_table.clear()
            self._init_person_table()

    def details_button_callback(self):
        button = self.sender()
        print("sender: {} person_id: {}".format(button, button.person_id))
        PersonDetailsDialog.ShowPerson(button.person_id, self)

    def images_button_callback(self):
        button = self.sender()
        log.debug("sender: {} person_id: {}".format(button, button.person_id))
        ImageListDialog.ShowPersonImages(button.person_id, self)

    def add_video_button_callback(self):
        button = self.sender()
        person_id = button.person_id
        filename = qtw.QFileDialog.getOpenFileName(
            self, "Add Video", '/tmp', 'Video Files (*.mpeg *.mp4 *.mjpg)',
            '*.mpeg', qtw.QFileDialog.Options())
        filename = filename[0]

        if filename != '':
            facelist = fd.faces_from_video_file(filename)
            selected_faces = ImageListDialog.SelectedImages(
                facelist, selected_by_default=False, parent=self)
            for img in selected_faces:
                self.recognizer.add_face_to_dataset(person_id, img)
            self.recognizer.train()

    def add_image_button_callback(self):
        button = self.sender()
        person_id = button.person_id
        filename = qtw.QFileDialog.getOpenFileName(
            self, "Add Image", '/tmp', 'Image Files (*.png *.jpeg *.jpg)',
            '*.jpeg', qtw.QFileDialog.Options())
        filename = filename[0]

        if filename != '':
            (face_imgs, coords) = fd.faces_from_image_file_v2(
                filename, resize=(224, 224))
            if face_imgs.shape[0] > 0:
                facelist = [face_imgs[i] for i in range(face_imgs.shape[0])]
                selected_faces = ImageListDialog.SelectedImages(
                    facelist, selected_by_default=False, parent=self)
                for img in selected_faces:
                    self.dataset.add_face_to_person(person_id, img)

            else:
                qtw.QMessageBox.information("No faces found on the image")

            self.recognizer.train()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    app = qtw.QApplication(sys.argv)
    recognizer = fr6.FaceRecognizer_v6()
    pd = PersonDialog(recognizer)
    pd.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
