from utils import *
log = init_logging()

import tensorflow as tf
# import keras.backend.tensorflow_backend
# keras.backend.tensorflow_backend.set_session(
#     tf.Session(config=tensorflow_config()))
# log.debug(
#     "mem_frac: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     per_process_gpu_memory_fraction)
# log.debug(
#     "allow_growth: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     allow_growth)
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, save_img, img_to_array, ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils

import numpy as np
import cv2
import os
import time
import pickle
import subprocess as sp
import glob

import database_api as db
import dataset_manager_v2 as dm2
import face_detection as fd
import redis
import redis_queue_utils as rqu

# import vgg_face_encoder


class FaceRecognizer_v3:
    def __init__(self, user_root="dataset-images/user-test", retrain=False):

        self.detector = fd.FaceDetectorTensorflow()
        log.debug("detector OK")
        self.vgg_model_filename = 'facebin-artifacts/vgg_face_weights.h5'
        self.model_filename = "facebin-artifacts/facebin_model_v3.h5"
        self.dataset = dm2.DatasetManager_v2(
            user_root=user_root, validation_root=None)
        self.load_or_train(retrain)

    def create_model(self):
        input_img = Input(shape=(224, 224, 3))
        TRAINVGG = False

        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(input_img)
        vgg = Convolution2D(
            64, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            64, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = MaxPooling2D((2, 2), strides=(2, 2), trainable=TRAINVGG)(vgg)

        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            128, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            128, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = MaxPooling2D((2, 2), strides=(2, 2), trainable=TRAINVGG)(vgg)

        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            256, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            256, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            256, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = MaxPooling2D((2, 2), strides=(2, 2), trainable=TRAINVGG)(vgg)

        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            512, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            512, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            512, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = MaxPooling2D((2, 2), strides=(2, 2), trainable=TRAINVGG)(vgg)

        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            512, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            512, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = ZeroPadding2D((1, 1), trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            512, (3, 3), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = MaxPooling2D((2, 2), strides=(2, 2), trainable=TRAINVGG)(vgg)

        vgg = Convolution2D(
            4096, (7, 7), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = Dropout(rate=0.5, trainable=TRAINVGG)(vgg)
        vgg = Convolution2D(
            4096, (1, 1), activation='relu', trainable=TRAINVGG)(vgg)
        vgg = Dropout(rate=0.5, trainable=TRAINVGG)(vgg)
        vgg_descriptor = Convolution2D(2622, (1, 1), trainable=TRAINVGG)(vgg)
        # 1
        vgg_flatten = Flatten(trainable=TRAINVGG)(vgg_descriptor)
        vgg_softmax = Activation('softmax', trainable=TRAINVGG)(vgg_flatten)

        # 2
        vgg_relu = Activation('relu')(vgg_descriptor)

        self.vgg_softmax = Model(input_img, vgg_softmax)
        self.vgg_softmax.load_weights(self.vgg_model_filename)

        # When vgg encoding is needed
        self.vgg_descriptor = vgg_descriptor

        # trainable layers
        facebin = Dense(1024, activation='relu')(vgg_flatten)
        facebin = Dense(512, activation='relu')(facebin)
        facebin = Dense(512, activation='relu')(facebin)
        facebin = Dense(
            self.dataset.n_classes(), activation='softmax')(facebin)

        self.classifier = Model(input_img, facebin)
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.1)
        self.classifier.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        log.info(self.classifier.summary())
        return self.classifier

    def train(self):

        training_df = self.dataset.get_user_dataframe()
        log.debug("training_df: %s", training_df)
        # log.debug("validation_df: %s", validation_df)

        generator_params = {
            "vertical_flip": True,
            "rescale": 1.0 / 255,
            "validation_split": 0.1,
            "rotation_range": 20,
            # "width_shift_range": 0.2,
            # "height_shift_range": 0.2
        }

        flow_parameters = {
            "directory":
            os.path.dirname(os.path.abspath(os.path.realpath(__file__))) + '/',
            "target_size": (224, 224),
            "x_col":
            "filename",
            "y_col":
            "class"
        }

        log.debug("flow_parameters: %s", flow_parameters)

        idg = ImageDataGenerator(**generator_params)
        training_generator = idg.flow_from_dataframe(
            training_df, **flow_parameters, subset='training')
        validation_generator = idg.flow_from_dataframe(
            training_df, **flow_parameters, subset='validation')
        batch_size = 32
        nb_epoch = 100
        steps_per_epoch = 1000

        stop_early = EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, mode='auto')

        tensor_board = TensorBoard(
            log_dir='/tmp/face_recognition_v3/tf-{}'.format(time.time()),
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True)

        self.classifier.fit_generator(
            generator=training_generator,
            use_multiprocessing=True,
            validation_data=validation_generator,
            validation_steps=100,
            epochs=nb_epoch,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            callbacks=[stop_early, tensor_board])

    def generate_training_face_directory(self):
        def get_paths_in_dict(path):
            dd = {d.name: d.path for d in os.scandir(path) if d.is_dir()}
            ff = {
                k: {u.name: u.path
                    for u in os.scandir(v)}
                for k, v in dd.items()
            }
            return ff

        user_files = get_paths_in_dict(self.user_image_dir)
        validation_test_files = get_paths_in_dict(self.validation_test_dir)
        validation_training_files = get_paths_in_dict(
            self.validation_training_dir)

        for cls, d in user_files.items():
            for n, p in d.items():
                if not n.endswith(".face.png"):
                    face_file = n + '.face.png'
                    face_path = p + '.face.png'
                    if not face_file in d:
                        face_imgs = fd.faces_from_image_file_v2(
                            p, largest_only=False, detector=self.detector)
                        if face_imgs.shape[0] == 0:
                            log.warning("No face found: %s", p)
                        elif face_imgs.shape[0] > 1:
                            log.warning("Multiple faces found: %s", p)
                        else:
                            cv2.imwrite(face_path, face_imgs[0])
                    else:
                        log.debug("Face file found already for %s", face_path)
        # we get again because of the changes
        user_files = get_paths_in_dict(self.user_image_dir)

        # Copy all user_files to training_root

        self.training_set_root = os.path.expandvars(
            "$HOME/facebin-data/training-{}".format(time.time()))

        self.validation_set_root = os.path.expandvars(
            "$HOME/facebin-data/validation-{}".format(time.time()))

        os.makedirs(self.training_set_root)
        os.makedirs(self.validation_set_root)

        for uc, uimgs in user_files.items():
            cls_dir = os.path.join(self.training_set_root, uc)

    def load_or_train(self, retrain=False):
        def refresh():
            self.create_model()
            self.train()
            self.save_model()

        if retrain:
            log.warning("Retraining")
            refresh()
        elif not os.path.exists(self.model_filename):
            log.warning("Cannot Find Stored Model File, retraining")
            refresh()
        elif os.path.getmtime(self.model_filename) < os.path.getmtime(
                self.dataset.USER_ROOT):
            log.warning("Dataset file is newer than the model. Retraining.")
            refresh()
        else:
            self.classifier = load_model(self.model_filename)
            log.info("LOADED:\n" + self.model_filename)

    def save_model(self):
        self.classifier.save(self.model_filename)

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
        (faces, coords) = fd.faces_from_image_v2(
            image,
            resize=(224, 224),
            largest_only=False,
            detector=self.detector)
        log.debug("Face Detection Time: %s", time.time() - fd_start)
        log.debug("Faces #: %s", len(faces))

        if len(faces) > 0:
            faces = faces / 255.0
            prediction = self.classifier.predict(faces)
            if prediction.size > 0:
                log.debug("prediction: %s", prediction)
                results = np.argmax(prediction, axis=1)
                probs = np.max(prediction, axis=1)
                log.debug("results: %s", results)
                log.debug("probs: %s", probs)
            else:
                results = None
                probs = None
        else:
            results = None
            probs = None
        return (coords, results, probs)

    def _save_dataset(self):
        self.dataset.save_dataset()

    def _preprocess_img(self, img):
        resized = cv2.resize(
            img,
            dsize=self.dataset.IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST)
        feature_arr = self._vgg_encode(resized)
        return (resized, feature_arr)

    def _vgg_encode(self, img):
        arr = img_to_array(img)
        arr = np.expand_dims(img, axis=0)
        arr = preprocess_input(arr)
        return self.vgg_descriptor.predict(arr)


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
    recognizer = FaceRecognizer_v3()
    log.debug("Face Recognizer Initialized")
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text_color = (0, 255, 255)
    known_rectangle_color = (0, 255, 0)
    unknown_rectangle_color = (0, 0, 255)
    skip_threshold = 1000
    prev_faces = None
    prev_faces_used = 0
    R = rqu.R
    input_queue = rqu.CAMERA_QUEUE
    prediction = None
    coords = None
    probs = None
    prediction_threshold = 0.8
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
                prev_faces = None
        else:
            recognizer_begin = time.time()
            coords, prediction, probs = recognizer.predict_faces(image_data)
            print(prediction)
            prev_faces = (coords, prediction, probs)
            prev_faces_used = 0
            recognizer_end = time.time()
            log.debug("Recognizer Time: %s",
                      (recognizer_end - recognizer_begin))
        if prediction is None:
            log.debug("prediction is None")
        else:
            for i in range(prediction.shape[0]):
                x, y, w, h = (coords[i, 0], coords[i, 1], coords[i, 2],
                              coords[i, 3])
                person_id = prediction[i]
                person_prob = probs[i]
                processing_begin = time.time()
                log.debug("coords: %s", coords)
                db_result = []
                if person_prob > prediction_threshold:
                    person_id = int(person_id)
                    db_result = db.person_by_id(person_id)
                else:
                    person_id = int(-1 * time.time() * i)

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
                text_line = "{} (%{})".format(name, int(person_prob * 100))
                cv2.putText(image_data, text_line, (y, x - 5), font, 1.0,
                            text_color, 2)
                processing_end = time.time()
                if prev_faces_used == 0:
                    face_image = image_data[x:(x + w), y:(y + h)]
                    add_face(R, key, i, face_image, person_id, (x, y, w, h))
                    log.debug("Face Processing Time: %s",
                              (processing_end - processing_begin))

        log.debug("Adding Processed Frame: %s", image_data.shape)
        add_processed_frame(R, key, image_data)
        R.zadd(output_queue, key, score)


if __name__ == '__main__':
    import traceback
    rr = FaceRecognizer_v3(retrain=True)
    for d in glob.glob(
            os.path.expandvars(
                "$HOME/facebin-artifacts/dataset-images/user-test/*")):
        print(d)
        for f in glob.glob(d + '/*'):
            ii = cv2.imread(f)
            print("Dir: {}".format(d))
            pp = rr.predict_faces(ii)
            print("Prediction: {}".format(pp))
            try:
                print(pp[0], pp[1], pp[2])
                person_id = pp[1][0]
                person = db.person_by_id(str(person_id))
                print("Person: {}".format(person))
            except Exception:
                traceback.print_exc()
