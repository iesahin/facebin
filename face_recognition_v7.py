from utils import *
log = init_logging()
import numpy as np
import cv2
import os
import time
import pickle
import sys
import shutil

import database_api as db
import dataset_manager_v3 as ds3
import face_detection as fd
import redis
import redis_queue_utils as rqu

from keras.engine import Model
from keras.layers import Input, Dense, Flatten, concatenate
from keras.preprocessing.image import load_img, save_img, img_to_array, ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras_vggface.vggface import VGGFace

from tensorflow.contrib.losses import metric_learning
import keras_vggface as kvgg


class FaceRecognizer_v7:
    def __init__(self,
                 retrain=False,
                 classifier_type="knn",
                 user_root=None,
                 use_gpu=True):
        log.debug("encoder created")
        self.detector = fd.FaceDetectorTensorflow()
        log.debug("detector")
        self.dataset = ds3.DatasetManager_v3(
            detector=self.detector, encoder=self, user_root=user_root)
        self.classifier_type = classifier_type
        log.debug("classifier")
        self.model_filename = "facebin_{}_v7.pickle".format(classifier_type)
        if self.dataset.size() > 0:
            log.debug("Loaded Dataset from: %s", self.dataset.FEATURE_FILE)
        else:
            log.warning("Dataset is EMPTY.")
        log.debug("load_or_train")

        self.create_model()
        self.dataset.add_missing_face_features()

    def create_model(self):
        vgg_model = VGGFace(
            model='vgg16',
            trainable=False,
            pooling=None,
            include_top=True,
            input_shape=(224, 224, 3))
        vgg_encoder_layer = vgg_model.get_layer('fc7').output
        self.encoder = Model(vgg_model.input, vgg_encoder_layer)

        # self.classifier = vgg_model
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.1)
        self.encoder.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            # optimizer=sgd,
            metrics=['accuracy'])
        log.info(self.encoder.summary())
        return self.encoder

    def train_triplet_layer(self):
        self.encoder.summary()
        base_encoding_input = Input(
            shape=(4096,),
            name="base_encoding")
        base_labels_input = Input(shape=(16,), name="base_labels", dtype="int32")
        print("base_labels_input: %s", base_labels_input) 
        print("base_encoding_input: %s", base_encoding_input) 
        triplet_encoder_layer = Dense(
            4096, name='triplet_dense1')(base_encoding_input)

        self.triplet_encoder = Model(inputs=[base_labels_input, base_encoding_input],
                                     outputs=[base_labels_input, triplet_encoder_layer])

        self.triplet_encoder.summary()

        self.triplet_encoder.compile(
            loss=metric_learning.triplet_semihard_loss,
            optimizer='adam',
            metrics=['accuracy'])

        feature_array = self.dataset.feature_array
        db_labels = db.person_feature_id_list()
        assert feature_array.shape[0] == len(db_labels)

        labels = np.zeros(dtype=np.int32, shape=(1, feature_array.shape[0]))

        for fi, pi in db_labels:
            labels[fi] = int(pi)

        self.triplet_encoder.fit(x=[labels, feature_array], epochs=10, batch_size=16)

    def encode(self, img):
        assert img.shape == (224, 224, 3)
        x = img_to_array(img)
        log.debug("x: %s", x)
        x = np.expand_dims(x, axis=0)
        log.debug("x: %s", x)
        x = kvgg.utils.preprocess_input(x, version=1)
        log.debug("x: %s", x)
        preds = self.encoder.predict(x)
        log.debug("preds: %s", preds)
        return preds

    def _predict_by_euclidean_distance(self, encoding):
        log.debug("self.dataset.feature_array.shape: %s",
                  self.dataset.feature_array.shape)
        dists = np.linalg.norm(self.dataset.feature_array - encoding, axis=1)
        print(dists)
        assert dists.shape[0] == self.dataset.feature_array.shape[0]
        candidate = np.argmin(dists)
        return (candidate, dists[candidate])

    def _predict_by_triplet_distance(self, encoding):
        triplet_encoding = self.triplet_encoder.predict(encoding)
        dists = np.linalg.norm(
            self.dataset.feature_array - triplet_encoding, axis=1)
        print(dists)
        assert dists.shape[0] == self.dataset.feature_array.shape[0]
        candidate = np.argmin(dists)
        return (candidate, dists[candidate])

    def predict_faces(self, image):
        fd_start = time.time()
        (faces, coords) = fd.faces_from_image_v2(
            image,
            resize=(224, 224),
            largest_only=False,
            detector=self.detector)
        log.debug("Face Detection Time: %s", time.time() - fd_start)
        log.debug("Faces #: %s", len(faces))
        results = np.empty(shape=(faces.shape[0], ), dtype=np.int)
        dists = np.empty(shape=(faces.shape[0], ), dtype=np.float32)
        encodings = np.empty(
            shape=(faces.shape[0], self.dataset.FEATURE_SIZE),
            dtype=np.float32)
        for j in range(faces.shape[0]):
            log.debug("faces[0].shape: %s", faces[0].shape)
            encoded = self.encode(faces[j])
            log.debug("encoded.shape: %s", encoded.shape)
            c, d = self._predict_by_triplet_distance(encoded)
            results[j] = c
            dists[j] = d
            encodings[j] = encoded

        return (coords, results, dists, encodings)

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
            rqu.processed_image_data_k(): image_data.tostring(),
            rqu.processed_image_shape_x_k(): image_data.shape[0],
            rqu.processed_image_shape_y_k(): image_data.shape[1],
            rqu.processed_image_shape_z_k(): image_data.shape[2],
            rqu.processed_image_dtype_k(): str(image_data.dtype),
        })

    log.debug("res: %s", res)
    return res


def add_face(R, key, face_i, face_image, person_id, coords, encoding,
             feature_id):
    log.debug("encoding.shape: %s", encoding.shape)
    log.debug("encoding.dtype: %s", encoding.dtype)
    x, y, w, h = coords[0], coords[1], coords[2], coords[3]
    R.hmset(
        key, {
            rqu.face_x_k(face_i): int(x),
            rqu.face_y_k(face_i): int(y),
            rqu.face_w_k(face_i): int(w),
            rqu.face_h_k(face_i): int(h),
            rqu.face_person_id_k(face_i): person_id,
            rqu.face_encoding_k(face_i): encoding.tostring(),
            rqu.face_encoding_dtype_k(face_i): str(encoding.dtype),
            rqu.face_image_data_k(face_i): face_image.tostring(),
            rqu.face_image_shape_x_k(face_i): int(face_image.shape[0]),
            rqu.face_image_shape_y_k(face_i): int(face_image.shape[1]),
            rqu.face_image_shape_z_k(face_i): int(face_image.shape[2]),
            rqu.face_image_dtype_k(face_i): str(face_image.dtype),
            rqu.face_feature_id_k(face_i): feature_id,
        })


def face_recognition_loop():
    outfile = open(
        "/tmp/facebin-face-recognition-out-pid-{}.txt".format(os.getpid()),
        "a")
    sys.stdout = outfile
    sys.stderr = outfile
    log.debug("Initializing Face Recognizer")
    recognizer = FaceRecognizer_v7()
    log.debug("Face Recognizer Initialized")
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text_color = (0, 255, 255)
    known_rectangle_color = (0, 255, 0)
    unknown_rectangle_color = (0, 0, 255)
    skip_threshold = 1000
    prev_faces = (np.array([]), np.array([]), np.array([]), np.array([]))
    prev_faces_used = 100
    R = rqu.R
    input_queue = rqu.CAMERA_QUEUE
    RECOGNITION_THRESHOLD = 500
    while True:
        l = R.zcount(input_queue, 0, "inf")
        key, score = rqu.get_next_key(input_queue, delete=True)
        log.debug("key: %s", key)
        log.debug("score: %s", score)
        if key is None:
            continue
        image_data = rqu.get_frame_image(key)
        if image_data is None:
            continue
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
                prev_faces = (np.array([]), np.array([]), np.array([]),
                              np.array([]))
                # delete waiting keys
                R.zremrangebyrank(input_queue, 0, skip_threshold)
        else:
            recognizer_begin = time.time()
            faces = recognizer.predict_faces(image_data)
            prev_faces = faces
            prev_faces_used = 0
            recognizer_end = time.time()
            log.debug("Recognizer Time: %s",
                      (recognizer_end - recognizer_begin))

        coords, results, dists, encodings = faces
        for face_i in range(coords.shape[0]):
            processing_begin = time.time()
            (x, y, w, h) = coords[face_i]
            log.debug("coords: %s", coords)
            candidate = int(results[face_i])
            log.debug("candidate: %s", candidate)
            dist = dists[face_i]
            if dist < RECOGNITION_THRESHOLD:
                person = db.person_by_feature_id(candidate)
                log.debug("person: %s", person)
                if len(person) > 0:
                    person = person[0]
                    (person_id, name, title, notes) = person
                else:
                    (person_id, name, title, notes) = (-1, "DB Error",
                                                       "Unknown", "")
                cv2.rectangle(image_data, (y, x), (y + h, x + w),
                              known_rectangle_color, 2)
            else:
                name = "Unknown Unknown"
                title = "Unknown"
                notes = ""
                cv2.rectangle(image_data, (y, x), (y + h, x + w),
                              unknown_rectangle_color, 2)
                person_id = int(-1 * time.time())

            # log.debug("person_id: {} title: {} name: {}".format(person_id, title, name))
            label = "{} - {} - {:.2f}".format(name, candidate, dist)
            cv2.putText(image_data, label, (y, x - 5), font, 1.0, text_color,
                        2)
            processing_end = time.time()
            if prev_faces_used == 0:
                face_image = image_data[x:(x + w), y:(y + h)]
                add_face(R, key, face_i, face_image, person_id, coords[face_i],
                         encodings[face_i], candidate)
                log.debug("Face Processing Time: %s",
                          (processing_end - processing_begin))

        log.debug("Adding Processed Frame: %s", image_data.shape)
        add_processed_frame(R, key, image_data)
        R.zadd(output_queue, {key: score})


def test_classification(training_dir, testing_dir):
    output_dir = "/tmp/fr-test-{}".format(time.time())
    fr = FaceRecognizer_v7(user_root=training_dir)
    fr.train_triplet_layer()
    for d in os.scandir(testing_dir):
        res_dir = os.path.join(output_dir, d.name)
        os.makedirs(res_dir)
        for f in os.scandir(d.path):
            if f.name.endswith('gif'):
                continue
            print(f.path)
            img = cv2.imread(f.path)
            coords, results, dists, encodings = fr.predict_faces(img)
            if results.shape[0] == 1:
                person = db.person_by_feature_id(int(results[0]))
                if len(person) > 0:
                    person = person[0]
                    (person_id, name, title, notes) = person
                else:
                    (person_id, name, title, notes) = (-1, "DB-Error",
                                                       "Unknown", "")
                target_dir = os.path.join(res_dir, name)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                target_file = "{}/{:.2f}-{}".format(target_dir, dists[0],
                                                    f.name)
                shutil.copy(f, target_file)


if __name__ == '__main__':
    test_classification(
        testing_dir="dataset-images/vgg-test-train/train",
        training_dir="dataset-images/user-test-v4/")
