from utils import *
log = init_logging()
import tensorflow as tf
import keras.backend.tensorflow_backend
keras.backend.tensorflow_backend.set_session(
    tf.Session(config=tensorflow_config()))
log.debug(
    "mem_frac: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    per_process_gpu_memory_fraction)
log.debug(
    "allow_growth: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    allow_growth)

import numpy as np
import sklearn.neighbors as skn
import sklearn.svm as svm
import sklearn.ensemble as ske
import sklearn.linear_model as sklm
import sklearn.neural_network as sknn
import sklearn.gaussian_process as skgp
log.debug(
    "mem_frac: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    per_process_gpu_memory_fraction)
log.debug(
    "mem_frac: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    allow_growth)

from sklearn.calibration import CalibratedClassifierCV
import cv2
import os
import time
import pickle

import database_api as db
import dataset_api as ds
import face_detection as fd
import redis
import redis_queue_utils as rqu
log.debug(
    "mem_frac: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    per_process_gpu_memory_fraction)
log.debug(
    "allow_growth: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    allow_growth)

import vgg_face_encoder

log.debug(
    "mem_frac: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    per_process_gpu_memory_fraction)
log.debug(
    "allow_growth: %s",
    keras.backend.tensorflow_backend.get_session()._config.gpu_options.
    allow_growth)


class FaceRecognizer_v1:
    def __init__(self, prediction_threshold=0.50, retrain=False):
        self.prediction_threshold = prediction_threshold
        log.debug("prediction_threshold")
        log.debug(
            "mem_frac: %s",
            keras.backend.tensorflow_backend.get_session()._config.gpu_options.
            per_process_gpu_memory_fraction)
        log.debug(
            "allow_growth: %s",
            keras.backend.tensorflow_backend.get_session()._config.gpu_options.
            allow_growth)

        self.encoder = vgg_face_encoder.VGGEncoder()
        log.debug(
            "mem_frac: %s",
            keras.backend.tensorflow_backend.get_session()._config.gpu_options.
            per_process_gpu_memory_fraction)
        log.debug(
            "allow_growth: %s",
            keras.backend.tensorflow_backend.get_session()._config.gpu_options.
            allow_growth)

        log.debug("encoder")
        self.detector = fd.FaceDetectorTensorflow()
        log.debug("detector")
        # self.detector_tf = fd.FaceDetectorTensorflow()
        # self.detector_dlib = fd.FaceDetectorDlib()
        self.dataset = ds.DatasetManager()
        log.debug("dataset")
        ## self.classifier = svm.SVC(kernel='rbf', gamma=2, C=1, probability=True)
        # random_forest = ske.RandomForestClassifier(
        #     n_estimators=1000, max_depth=10)
        # log.debug("random_forest")
        # self.classifier = CalibratedClassifierCV(random_forest)

        log.debug("classifier")
        self.model_filename = "facebin_model_v1.pickle"
        if self.dataset.size() > 0:
            self.load_or_train(retrain)
        else:
            log.warning("Dataset is EMPTY.")
        log.debug("load_or_train")

    def _random_forest(self):
        random_forest = ske.RandomForestClassifier(
            n_estimators=1000, max_depth=20, class_weight='balanced')
        return random_forest

    def _asgd(self):
        asgd = sklm.SGDClassifier(
            average=True,
            tol=0.00001,
            max_iter=100,
            loss='modified_huber',
            fit_intercept=True,
            shuffle=True,
            verbose=1,
            n_jobs=-1,
            learning_rate='optimal',
            class_weight='balanced')
        return asgd

    def _mlp(self):
        mlp = sknn.MLPClassifier(
            hidden_layer_sizes=(1000, 500, 500, 500, 500),
            activation='relu',
            solver='adam')
        return mlp

    def _knn(self):
        knn = skn.KNeighborsClassifier(
            n_neighbors=9, weights='uniform', metric='euclidean', n_jobs=-1)
        return knn

    def _gaussian_process(self):
        kernel = 1.0 * skgp.kernels.RBF(1.0)
        gpc = skgp.GaussianProcessClassifier(
            kernel=kernel, multi_class='one_vs_rest')
        return gpc

    def train(self):
        log.debug(self.dataset.person_ids.shape)
        log.debug(self.dataset.feature_vectors.shape)
        self.classifier = CalibratedClassifierCV(self._mlp())
        y = np.ravel(self.dataset.person_ids)
        log.debug(y.shape)
        log.debug(y)
        self.classifier.fit(self.dataset.feature_vectors, y)

    def load_or_train(self, retrain=False):
        if retrain:
            log.warning("Retraining")
            self.train()
            self.save_model()
        elif not os.path.exists(self.model_filename):
            log.warning("Cannot Find Stored Model File, retraining")
            self.train()
            self.save_model()
        elif os.path.getmtime(self.model_filename) < os.path.getmtime(
                self.dataset.DATASET_PATH):
            log.warning("Dataset file is newer that the model. Retraining.")
            self.train()
            self.save_model()
        else:
            load_f = open(self.model_filename, 'rb')
            self.classifier = pickle.load(load_f)

    def save_model(self):
        save_f = open(self.model_filename, 'wb')
        pickle.dump(self.classifier, save_f)

    def add_image_to_dataset(self, image, person_id):
        (faces, coords) = fd.faces_from_image(
            image, largest_only=True, detector=self.detector)
        log.debug("Face Coords: ", coords)
        if len(faces) != 1:
            log.warning("There should be 1 (and only 1) face on the image: {}",
                        len(faces))
        else:
            self.add_face_to_dataset(faces[0], person_id)
            # resized, feature_vector = self._preprocess_img(faces[0])
            # self.dataset.add_item(resized, feature_vector, person_id)

    def add_face_to_dataset(self, face_image, person_id):
        resized, feature_vector = self._preprocess_img(face_image)
        log.debug(resized.shape)
        self.dataset.add_item(resized, feature_vector, person_id)

    def predict_faces(self, image):
        # r = np.random.rand()
        # if r > 0.5:
        #     self.detector = self.detector_tf
        #     log.debug("Face Detector TensorFlow")
        # else:
        #     self.detector = self.detector_dlib
        #     log.debug("Face Detector DLib")

        fd_start = time.time()
        (faces, coords) = fd.faces_from_image(
            image, largest_only=False, detector=self.detector)
        log.debug("Face Detection Time: %s", time.time() - fd_start)
        log.debug("Faces #: %s", len(faces))
        results = []
        for k in faces:
            (resized, feature_array) = self._preprocess_img(faces[k])
            print(feature_array.shape)
            prediction = self.classifier.predict_proba(feature_array)
            log.debug(prediction)
            sorted_i = np.argsort(prediction[-1])
            first_p = prediction[0][sorted_i[-1]]
            second_p = prediction[0][sorted_i[-2]]
            log.debug("first_p: %.2f", first_p)
            log.debug("second_p: %.2f", second_p)
            # TODO: Decide on a persistent criteria by testing
            if (first_p - second_p) > second_p:
                #   if first_p > 0.5:
                # if (first_p - second_p) > (second_p / 10):
                pred = self.classifier.classes_[sorted_i[-1]]
            else:
                pred = None
            results.append((coords[k], pred))
        return results

    def _save_dataset(self):
        self.dataset.save_dataset()

    def _preprocess_img(self, img):
        resized = cv2.resize(
            img,
            dsize=self.dataset.IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST)
        feature_arr = self.encoder.encode(resized)
        return (resized, feature_arr)


def add_processed_frame(R, key, image_data):
    res = R.hmset(
        key, {
            'processed_image_data': image_data.tostring(),
            'processed_image_shape_x': image_data.shape[0],
            'processed_image_shape_y': image_data.shape[1],
            'processed_image_shape_z': image_data.shape[2],
            'processed_image_dtype': str(image_data.dtype),
        })

    log.debug("res: %s", res)
    return res


def add_face(R, key, face_i, face_image, person_id, coords):
    x, y, w, h = coords
    R.hmset(
        key, {
            "face_{}_x".format(face_i): int(x),
            "face_{}_y".format(face_i): int(y),
            "face_{}_w".format(face_i): int(w),
            "face_{}_h".format(face_i): int(h),
            "face_{}_person_id".format(face_i): person_id,
            'face_image_{}_data'.format(face_i): face_image.tostring(),
            'face_image_{}_shape_x'.format(face_i): int(face_image.shape[0]),
            'face_image_{}_shape_y'.format(face_i): int(face_image.shape[1]),
            'face_image_{}_shape_z'.format(face_i): int(face_image.shape[2]),
            'face_image_{}_dtype'.format(face_i): str(face_image.dtype),
        })


def face_recognition_loop():
    log.debug("Initializing Face Recognizer")
    recognizer = FaceRecognizer_v1()
    log.debug("Face Recognizer Initialized")
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text_color = (0, 255, 255)
    known_rectangle_color = (0, 255, 0)
    unknown_rectangle_color = (0, 0, 255)
    skip_threshold = 1000
    prev_faces = []
    prev_faces_used = 0
    R = rqu.R
    input_queue = rqu.CAMERA_QUEUE
    while True:
        l = R.zcount(input_queue, 0, "inf")
        key, score = rqu.get_next_key(input_queue, delete=True)
        if key is None:
            continue
        image_data = rqu.get_frame_image(key)
        log.debug("image_data.shape: %s", image_data.shape)
        camera_id = rqu.get_frame(key, fields=['camera_id'])['camera_id']
        log.debug("camera_id: %s", camera_id)
        output_queue = rqu.RECOGNIZER_QUEUE(camera_id)
        if image_data is None:
            continue
        if l > skip_threshold:
            log.debug("Skipping because of l=%s", l)
            # Use previous face data
            faces = prev_faces
            prev_faces_used += 1
            if prev_faces_used >= skip_threshold:
                prev_faces = []
        else:
            recognizer_begin = time.time()
            faces = recognizer.predict_faces(image_data)
            prev_faces = faces
            prev_faces_used = 0
            recognizer_end = time.time()
            log.debug("Recognizer Time: %s",
                      (recognizer_end - recognizer_begin))

        for face_i, face_data in enumerate(faces):
            coords, person_id = face_data
            processing_begin = time.time()
            (x, y, w, h) = coords
            log.debug("coords: %s", coords)
            db_result = []
            if person_id is not None:
                person_id = int(person_id)
                db_result = db.person_by_id(person_id)
            else:
                person_id = int(-1 * time.time())

            log.debug("db_result: %s", db_result)
            if db_result == []:
                name = "Unknown Unknown"
                title = "Unknown"
                notes = ""
                cv2.rectangle(image_data, (y, x), (y + h, x + w),
                              unknown_rectangle_color, 2)
            else:
                assert (len(db_result) == 1)
                (person_id_, name, title, notes) = db_result[0]
                cv2.rectangle(image_data, (y, x), (y + h, x + w),
                              known_rectangle_color, 2)
            log.debug("person_id: {} title: {} name: {}".format(
                person_id, title, name))
            cv2.putText(image_data, name, (y, x - 5), font, 1.0, text_color, 2)
            processing_end = time.time()
            if prev_faces_used == 0:
                face_image = image_data[x:(x + w), y:(y + h)]
                add_face(R, key, face_i, face_image, person_id, coords)
                log.debug("Face Processing Time: %s",
                          (processing_end - processing_begin))

        log.debug("Adding Processed Frame: %s", image_data.shape)
        add_processed_frame(R, key, image_data)
        R.zadd(output_queue, {key: score})


if __name__ == '__main__':
    face_recognition_loop()
