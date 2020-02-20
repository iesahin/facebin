import numpy as np
import numpy.random as nr
import PySide2.QtGui as qtg
import cv2
import datetime as dt
import os
import re
import random
import glob
import database_api as db
import face_detection as fd
import pandas as pd
import keras
from utils import *
import logging
import hashlib

log = init_logging()


class DatasetManager_v2:
    """Dataset manager to retrieve images, feature vectors, person ids and metadata
about the image dataset on the disk. It's mainly used to supply images to
ImageDataGenerator of Keras.

    """

    def __init__(self, user_root="dataset-images/user", validation_root=None):
        self.USER_ROOT = os.path.expandvars(user_root)
        self.VALIDATION_ROOT = os.path.expandvars(
            validation_root) if validation_root is not None else None
        self.IMAGE_SIZE = (224, 224)
        self.detector = fd.FaceDetectorTensorflow()
        if not os.path.exists(self.USER_ROOT):
            log.warning("USER_ROOT does not exist: %s",
                        os.path.abspath(self.USER_ROOT))
            os.makedirs(self.USER_ROOT)

        if self.VALIDATION_ROOT is not None and (not os.path.exists(
                self.VALIDATION_ROOT)):
            log.warning("VALIDATION_ROOT does not exist: %s",
                        os.path.abspath(self.VALIDATION_ROOT))
            os.makedirs(self.VALIDATION_ROOT)

    def n_classes(self):
        n_user = len(self._read_user_dataset())
        n_validation = len(self._read_validation_dataset())
        return n_user + n_validation

    def add_user_image(self, person_id, image, add_face_also=False):
        """Add an image for the person given by person_id

        >>> dm = DatasetManager_v2()
        """
        img_dir = self._user_image_dir(person_id)
        if not os.path.exists(img_dir):
            log.info("Creating %s", img_dir)
            os.makedirs(img_dir)
        filename = "ds-{}.png".format(
            hashlib.md5(image.tostring()).hexdigest())
        image_path = os.path.join(img_dir, filename)
        cv2.imwrite(image_path, image)
        if add_face_also:
            self.add_user_face_image(person_id, image_path)

    def add_user_face_image(self, person_id, image_file):
        """

        """
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
            face_file = self._user_face_image_file(image_file)
            cv2.imwrite(face_file, face_imgs[0])

    def add_validation_face_image(self, image_file):
        if self.VALIDATION_ROOT is None:
            return None

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
            face_file = self._user_face_image_file(image_file)
            cv2.imwrite(face_file, face_imgs[0])

    def list_missing_user_face_images(self):
        all_files = self._list_user_files()
        missing_files = []
        for d in all_files:
            for n in all_files[d]:
                if not self._is_face_file(n):
                    n_face = self._user_face_image_file(n)
                    if n_face not in all_files[d]:
                        missing_files.append(all_files[d][n])
        return missing_files

    def list_missing_validation_face_images(self):
        """Lists the images where we don't have an accompanying face.png image

        >>> dm = DatasetManager_v2(user_root="dataset-images/user-2", validation_root="dataset-images/validation")
        """
        all_files = self._list_validation_files()
        missing_files = []
        if self.VALIDATION_ROOT is None:
            return []
        for d in all_files:
            for n in all_files[d]:
                if not self._is_face_file(n):
                    n_face = self._user_face_image_file(n)
                    if n_face not in all_files[d]:
                        missing_files.append(all_files[d][n])
        return missing_files

    def create_missing_face_images(self,
                                   user_files=True,
                                   validation_files=True):
        """Creates the missing images in validation and user sets.

        >>> dm = DatasetManager_v2(user_root="dataset-images/user-2", validation_root="dataset-images/validation")
        >>> len(dm.list_missing_validation_face_images()) > 0
        True
        >>> dm.create_missing_face_images()
        >>> len(dm.list_missing_validation_face_images()) > 0
        False
        """
        all_files = []
        if user_files:
            all_files += self.list_missing_user_face_images()

        if validation_files:
            all_files += self.list_missing_validation_face_images()

        for f in all_files:
            (face_imgs, coords) = fd.faces_from_image_file_v2(
                f,
                largest_only=False,
                resize=(224, 224),
                detector=self.detector)
            if face_imgs is None:
                log.warning("Cannot read Image: %s", f)
                base, ext = os.path.splitext(f)
                os.rename(f, base + '.noimage' + ext)
            elif face_imgs.shape[0] == 0:
                log.warning("No face found: %s", f)
                base, ext = os.path.splitext(f)
                os.rename(f, base + '.noface' + ext)
            elif face_imgs.shape[0] > 1:
                log.warning("Multiple faces found: %s", f)
                base, ext = os.path.splitext(f)
                os.rename(f, base + '.multiface' + ext)
            else:
                face_file = self._user_face_image_file(f)
                cv2.imwrite(face_file, face_imgs[0])

    def get_user_dataframe(self, max_files_per_class=None):
        user_set = self._read_user_dataset()

        training_x = []
        training_y = []
        if max_files_per_class is None:
            max_files_per_class = 10000
        for uc, uf in user_set.items():
            cc = self._extract_class(uc)
            if cc is not None:
                uf_items = list(uf.items())
                nr.shuffle(uf_items)
                for ufn, ufp in uf_items[:max_files_per_class]:
                    training_x.append(ufp)
                    training_y.append(cc)
        training_df = pd.DataFrame({
            "filename": training_x,
            "class": [str(c) for c in training_y]
        })

        training_df = training_df.sample(frac=1).reset_index(drop=True)
        return training_df

    def get_dataframes(self,
                       n_validation_classes=None,
                       n_validation_samples=10):
        user_set = self._read_user_dataset()
        validation_set = self._read_validation_dataset()
        log.debug("validation_set.keys(): %s", validation_set.keys())
        validation_classes = list(validation_set.keys())
        nr.shuffle(validation_classes)
        validation_classes = validation_classes[:n_validation_classes]
        vc_in_training = {}
        vc_in_validation = {}
        for vc in validation_classes:
            all_files = list(validation_set[vc])
            nr.shuffle(all_files)
            tr_subset = all_files[:n_validation_samples]
            va_subset = all_files[n_validation_samples:(
                n_validation_samples * 2)]
            vc_in_training[vc] = {k: validation_set[vc][k] for k in tr_subset}
            vc_in_validation[vc] = {
                k: validation_set[vc][k]
                for k in va_subset
            }

        training_x = []
        training_y = []
        validation_x = []
        validation_y = []

        for uc, uf in user_set.items():
            cc = self._extract_class(uc)
            if cc is not None:
                for ufn, ufp in uf.items():
                    training_x.append(ufp)
                    training_y.append(cc)

        max_user_class = max(training_y)
        validation_cc = {
            k: (max_user_class + c + 1)
            for c, k in enumerate(validation_classes)
        }

        for vc, vf in vc_in_training.items():
            cc = validation_cc[vc]
            for vfn, vfp in vf.items():
                training_x.append(vfp)
                training_y.append(cc)

        for vc, vf in vc_in_validation.items():
            cc = validation_cc[vc]
            for vfn, vfp in vf.items():
                validation_x.append(vfp)
                validation_y.append(cc)

        training_df = pd.DataFrame({
            "filename": training_x,
            "class": [str(c) for c in training_y]
        })
        training_df = training_df.sample(frac=1).reset_index(drop=True)
        validation_df = pd.DataFrame({
            "filename": validation_x,
            "class": [str(c) for c in validation_y]
        })
        validation_df = validation_df.sample(frac=1).reset_index(drop=True)
        if log.getEffectiveLevel() == logging.DEBUG:
            with open("/tmp/training.csv", "w") as ft:
                ft.write(training_df.to_csv())
            with open("/tmp/validation.csv", "w") as fv:
                fv.write(validation_df.to_csv())

        return (training_df, validation_df)

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

    def _list_validation_files(self):
        """Lists all validation files in USER_ROOT

        >>> dm = DatasetManager_v2()
        >>> val_files = dm._list_validation_files()
        >>> len(val_files)
        16
        """
        if self.VALIDATION_ROOT is None:
            return {}
        paths = {}
        for d in os.scandir(self.VALIDATION_ROOT):
            if d.is_dir():
                paths[d.name] = {}
                for f in os.scandir(d.path):
                    paths[d.name][f.name] = f.path
        return paths

    def _read_validation_dataset(self):
        if self.VALIDATION_ROOT is None:
            return {}
        paths = {}
        for d in os.scandir(self.VALIDATION_ROOT):
            if d.is_dir():
                paths[d.name] = {}
                for f in os.scandir(d.path):
                    if self._is_face_file(f.name):
                        paths[d.name][f.name] = os.path.abspath(f.path)
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

    def _user_face_image_file(self, image_path):
        "Returns the face image path of an image file. Basically it appends .face.png to the original path."
        return image_path + '.face.png'

    def _user_orig_image_file(self, image_path):
        if image_path.endswith('.face.png'):
            return image_path[:-9]
        else:
            return image_path

    def _is_face_file(self, image_path):
        return image_path.endswith(".face.png")

    def size(self):
        return self.person_ids.shape[0]

    def images_by_id(self, person_id):
        filter = self.person_ids == person_id
        filtered_images = np.copy(self.images[filter[:, 0]])
        return filtered_images

    def add_item(self, image: np.ndarray, person_id: int, save=True):
        assert (image.shape == (self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3))
        self.images = np.vstack([self.images, np.expand_dims(image, 0)])
        self.person_ids = np.vstack([self.person_ids, np.array([person_id])])
        if save:
            self.save_dataset()

    def _get_metadata(self, key):
        r = self.metadata[self.metadata['key'] == key]
        if r.shape == (1, ):
            return r[0][1]
        elif r.shape == (0, 2):
            return None
        else:
            raise Exception('Multiple Values for Key: ' + k)

    def add_metadata(self, key, value):
        r = self.metadata[self.metadata['key'] == key]
        if r.shape == (1, ):
            r['value'][0] = value
        elif r.shape == (0, ):
            self.metadata = np.vstack(
                [self.metadata, np.array([(key, value)])])

    def save_dataset(self):
        np.savez(
            self.DATASET_PATH,
            person_ids=self.person_ids,
            images=self.images,
            metadata=self.metadata)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
