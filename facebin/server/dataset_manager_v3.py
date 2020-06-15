import numpy as np
import numpy.random as nr
import cv2
import datetime as dt
import os
import re
import random
import glob
from . import database_api as db
from . import face_detection as fd
import pandas as pd
import keras
from .utils import *
import logging
import hashlib

import configparser as cp
from . import face_recognition_v6 as fr6

log = init_logging()

FEATURE_SIZE = 4096


class DatasetManager_v3:
    """Dataset manager to retrieve images, feature vectors, person ids and metadata
about the image dataset on the disk. It's mainly used to supply images to
ImageDataGenerator of Keras.

    # >>> import shutil
    # >>> dm_test_dir = '/tmp/{}/'.format(np.random.randint(100000))
    # >>> test_artifacts_dir = os.path.expandvars('$HOME/facebin-artifacts/dm3-test/')
    # >>> _ = shutil.copytree(test_artifacts_dir, dm_test_dir)
    >>> dm3 = DatasetManager_v3()

    """

    def __init__(self, user_root=None, detector=None, encoder=None):
        if user_root is None:
            config = get_configuration()
            self.USER_ROOT = os.path.expandvars(
                config['general']['dataset-dir'])
        else:
            self.USER_ROOT = os.path.expandvars(user_root)

        self.IMAGE_SIZE = (224, 224)
        self.FEATURE_FILE = os.path.join(self.USER_ROOT,
                                         "facebin.feature.gz.npz")
        self.FEATURE_SIZE = FEATURE_SIZE
        if detector is None:
            self.detector = fd.FaceDetectorTensorflow()
        else:
            self.detector = detector

        self.encoder = encoder

        if not os.path.exists(self.USER_ROOT):
            log.warning("USER_ROOT does not exist: %s",
                        os.path.abspath(self.USER_ROOT))
            os.makedirs(self.USER_ROOT)

        if os.path.exists(self.FEATURE_FILE):
            log.debug("Loading From: %s", self.FEATURE_FILE)
            self.feature_array = np.load(self.FEATURE_FILE)['feature_array']
        else:
            log.warning("Cannot Find %s. Recreating Dataset",
                        self.FEATURE_FILE)
            self.feature_array = np.empty((0, self.FEATURE_SIZE))

    def add_user_image(self, person_id, image, add_face_also=False):
        """Add an image for the person given by person_id"""
        img_dir = self._user_image_dir(person_id)
        if not os.path.exists(img_dir):
            log.info("Creating %s", img_dir)
            os.makedirs(img_dir)
        filename = "img-{}.png".format(
            hashlib.md5(image.tostring()).hexdigest())
        image_path = os.path.join(img_dir, filename)
        cv2.imwrite(image_path, image)
        img_h, img_w, img_c = image.shape
        super_image_id = db.insert_image(
            person_id=person_id,
            path=image_path,
            is_face=False,
            width=img_w,
            height=img_h,
            feature_id=None,
            super_image_id=None)
        self.save_dataset()
        if add_face_also:
            self.add_face_from_image(
                person_id, image_path, super_image_id=super_image_id)

    def add_face_from_image(self,
                            person_id,
                            image_file,
                            super_image_id,
                            feature=None):
        """Extracts face image and stores it as a record"""
        log.debug("image_file: %s", image_file)
        (face_imgs, coords) = fd.faces_from_image_file_v2(
            image_file,
            largest_only=False,
            resize=(224, 224),
            detector=self.detector)
        if face_imgs.shape[0] == 0:
            log.warning("No face found: %s", image_file)
        elif face_imgs.shape[0] > 1:
            log.warning("Multiple faces found: %s", image_file)
        else:
            self.add_face_to_person(person_id, face_imgs[0], super_image_id)

    def add_face_to_person(self,
                           person_id,
                           face_image,
                           super_image_id=None,
                           feature=None):
        face_file = self._user_face_image_file(person_id, face_image)
        log.debug("face_file: %s", face_file)
        cv2.imwrite(face_file, face_image)
        last_index = None
        log.debug("face_image.shape: %s", face_image.shape)
        log.debug("feature: %s", feature)
        img_h, img_w, img_c = face_image.shape
        assert (img_h, img_w) == self.IMAGE_SIZE
        if (feature is not None):
            self.feature_array = np.vstack([self.feature_array, feature])
            last_index = self.feature_array.shape[0] - 1
        if (feature is None) and (self.encoder is not None):
            feature = self.encoder.encode(face_image)
            self.feature_array = np.vstack([self.feature_array, feature])
            last_index = self.feature_array.shape[0] - 1
        elif feature is None and (self.encoder is None):
            log.warning("Adding face without feature to the dataset!")
            last_index = None

        log.debug("last_index: %s", last_index)
        log.debug("self.feature_array.shape: %s", self.feature_array.shape)
        face_image_id = db.insert_image(
            person_id=person_id,
            path=face_file,
            is_face=True,
            width=img_w,
            height=img_h,
            feature_id=last_index,
            super_image_id=super_image_id)
        self.save_dataset()

    def add_face_file_to_person(self,
                                person_id,
                                face_image_path,
                                super_image_id=None,
                                feature=None):
        face_image = cv2.imread(face_image_path)
        log.debug("face_image.shape: %s", face_image.shape)
        assert face_image.shape == (self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3)
        self.add_face_to_person(person_id, face_image, super_image_id, feature)

    def add_missing_face_features(self):
        assert self.encoder is not None

        faces_without_features = db.person_face_images_without_features()
        log.debug("len(faces_without_features): %s",
                  len(faces_without_features))

        if len(faces_without_features) > 0:
            for image_id, person_id, path, is_face, feature_id in faces_without_features:
                log.debug("image_id: %s", image_id)
                face_image = cv2.imread(path)
                log.debug("path: %s", path)
                feature = self.encoder.encode(face_image)
                self.feature_array = np.vstack([self.feature_array, feature])
                last_index = self.feature_array.shape[0] - 1
                log.debug("last_index: %s", last_index)
                db.update_face_image_feature_id(image_id, last_index)

                log.debug("saving dataset")
                self.save_dataset()

    def create_missing_face_images(self):
        """Creates the missing images in validation and user sets.

        >>> dm = DatasetManager_v3(user_root="dataset-images/user-2", validation_root="dataset-images/validation")
        >>> len(dm.list_missing_validation_face_images()) > 0
        True
        >>> dm.create_missing_face_images()
        >>> len(dm.list_missing_validation_face_images()) > 0
        False
        """
        all_files = self.list_missing_user_face_images()
        for f in all_files:
            (face_imgs, coords) = fd.faces_from_image_file_v2(
                f,
                largest_only=False,
                resize=(224, 224),
                detector=self.detector)
            if face_imgs is None:
                log.warning("Cannot read Image: %s", f)
                d, b = os.path.split(f)
                os.rename(f, os.path.join(d, "noimage-" + b))
            elif face_imgs.shape[0] == 0:
                log.warning("No face found: %s", f)
                d, b = os.path.split(f)
                os.rename(f, os.path.join(d, "noface-" + b))
            elif face_imgs.shape[0] > 1:
                log.warning("Multiple faces found: %s", f)
                d, b = os.path.split(f)
                os.rename(f, os.path.join(d, "multiface-" + b))
            else:
                face_file = self._user_face_image_file(person_id, face_imgs[0])
                cv2.imwrite(face_file, face_imgs[0])
                self.insert_record(person_id, face_file, True, None, img_h,
                                   img_w)

    def get_user_dataframe(self):
        training_df = self.dataframe.sample(frac=1).reset_index(drop=True)
        return training_df

    def _extract_class(self, dirname):
        """Extracts the class portion from a dirname. Used in dataset dirs like 01-Emre 02-Ayse ...

        >>> dm = DatasetManager_v2()
        >>> dm._extract_class("001-Emre")
        1
        """
        dd = os.path.basename(dirname)
        rr = re.match(r'^([0-9]+).*', dd)
        if rr:
            return int(rr.group(1))
        else:
            return None

    def _list_user_files(self):
        """Lists all user files in USER_ROOT

        >>> dm = DatasetManager_v2()
        >>> user_files = dm._list_user_files()
        >>> len(user_files)
        122
        """
        paths = {}
        for d in os.scandir(self.USER_ROOT):
            if d.is_dir():
                paths[d.name] = {}
                for f in os.scandir(d.path):
                    paths[d.name][f.name] = f.path
        return paths

    def _read_user_dataset(self):
        paths = {}
        for d in os.scandir(self.USER_ROOT):
            if d.is_dir():
                paths[d.name] = {}
                for f in os.scandir(d.path):
                    if self._is_face_file(f.name):
                        paths[d.name][f.name] = os.path.abspath(f.path)
        return paths

    def _user_image_dir(self, person_id):
        candidates = glob.glob(self.USER_ROOT + '/{:05d}--*'.format(person_id))
        if len(candidates) > 1:
            raise Exception("Multiple user image directories with same id: {}".
                            format(person_id))
        elif len(candidates) == 0:
            user_dir = os.path.join(self.USER_ROOT,
                                    "{:05d}--".format(person_id))
            log.warning("User image dir does not exist with person id %s",
                        user_dir)
            os.makedirs(user_dir)
            return user_dir
        else:
            return candidates[0]

    def _user_face_image_file(self, person_id, image):
        "Returns the face image path of an image file."
        filename = "face-{}.png".format(
            hashlib.md5(image.tostring()).hexdigest())
        d = self._user_image_dir(person_id)
        return os.path.join(d, filename)

    def _user_orig_image_file(self, image_path):
        d, f = os.path.split(image_path)
        if f.startswith('face-'):
            return os.path.join(d, f[5:])
        else:
            return image_path

    def _is_face_file(self, image_path):
        d, f = os.path.split(image_path)
        return f.startswith("face-")

    def size(self):
        return self.feature_array.shape[0]

    def save_dataset(self):
        np.savez(self.FEATURE_FILE, feature_array=self.feature_array)

    def add_feature_to_image(self, image_file, feature):
        assert False
        self.save_dataset()

    def images_by_id(self, person_id):
        image_records = db.person_face_images_by_person_id(person_id)
        results = {}
        for r in image_records:
            _id = r[0]
            results[_id] = {}
            results[_id]["id"] = r[0]
            results[_id]["person_id"] = r[1]
            results[_id]["path"] = r[2]
            results[_id]["is_face"] = True if r[3] == 1 else False
            results[_id]["super_image_id"] = r[4]
            results[_id]["width"] = r[5]
            results[_id]["height"] = r[6]
            results[_id]["feature_id"] = r[7]

        return results


if __name__ == "__main__":
    import doctest
    doctest.testmod()
