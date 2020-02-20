from utils import *
log = init_logging()
# import tensorflow as tf
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

import numpy as np
import sklearn.neighbors as skn
import sklearn.svm as svm
import sklearn.ensemble as ske
import sklearn.linear_model as sklm
import sklearn.neural_network as sknn
import sklearn.gaussian_process as skgp
# log.debug(
#     "mem_frac: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     per_process_gpu_memory_fraction)
# log.debug(
#     "mem_frac: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     allow_growth)

from sklearn.calibration import CalibratedClassifierCV
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
# log.debug(
#     "mem_frac: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     per_process_gpu_memory_fraction)
# log.debug(
#     "allow_growth: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     allow_growth)

# import vgg_face_encoder

from keras.engine import Model
from keras.layers import Input
from keras.preprocessing.image import load_img, save_img, img_to_array, ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras_vggface.vggface import VGGFace
import keras_vggface as kvgg

# from libKMCUDA import kmeans_cuda, knn_cuda

# log.debug(
#     "mem_frac: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     per_process_gpu_memory_fraction)
# log.debug(
#     "allow_growth: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     allow_growth)


class FaceRecognizer_v6:
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
        ## self.classifier = svm.SVC(kernel='rbf', gamma=2, C=1, probability=True)
        # random_forest = ske.RandomForestClassifier(
        #     n_estimators=1000, max_depth=10)
        # log.debug("random_forest")
        # self.classifier = CalibratedClassifierCV(random_forest)

        self.classifier_type = classifier_type
        log.debug("classifier")
        self.model_filename = "facebin_{}_v6.pickle".format(classifier_type)
        if self.dataset.size() > 0:
            log.debug("Loaded Dataset from: %s", self.dataset.FEATURE_FILE)
        else:
            log.warning("Dataset is EMPTY.")
        log.debug("load_or_train")

        self.create_model()
        self.dataset.add_missing_face_features()

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

    def create_model(self):
        vgg_model = VGGFace(
            model='vgg16',
            trainable=False,
            pooling=None,
            include_top=True,
            input_shape=(224, 224, 3))
        encoder_layer = vgg_model.get_layer('fc7').output
        # out = Dense(
        #     self.dataset.n_classes(), activation='softmax',
        #     name='fc8')(last_layer)
        # x = Flatten(name='flatten')(last_layer)
        # x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        # x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        # out = Dense(
        #     self.dataset.n_classes(), activation='softmax', name='fc8')(x)
        self.encoder = Model(vgg_model.input, encoder_layer)
        # self.classifier = vgg_model
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.1)
        self.encoder.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            # optimizer=sgd,
            metrics=['accuracy'])
        log.info(self.encoder.summary())
        return self.encoder

    def train_vgg(self):

        training_df = self.dataset.get_user_dataframe(max_files_per_class=1000)
        # log.debug("training_df: %s", training_df)
        # log.debug("validation_df: %s", validation_df)
        flow_output_dir = "/tmp/facebin-flow-output-{}".format(time.time())
        os.makedirs(flow_output_dir)

        generator_params = {
            # "horizontal_flip": True,
            # "vertical_flip": True,
            # "preprocessing_function": preprocess_for_vgg,
            "validation_split": 0.1,
            # "rotation_range": 10,
            # "width_shift_range": 0.1,
            # "height_shift_range": 0.2
        }

        flow_parameters = {
            "directory":
            os.path.dirname(os.path.abspath(os.path.realpath(__file__))) + '/',
            "target_size": (224, 224),
            "x_col":
            "filename",
            "y_col":
            "class",
            "save_to_dir":
            flow_output_dir
        }

        log.debug("flow_parameters: %s", flow_parameters)

        idg = ImageDataGenerator(**generator_params)
        # log.debug(training_df.to_csv())
        training_generator = idg.flow_from_dataframe(
            training_df, **flow_parameters, subset='training')
        validation_generator = idg.flow_from_dataframe(
            training_df, **flow_parameters, subset='validation')
        batch_size = 64
        nb_epoch = 100
        steps_per_epoch = 200

        stop_early = EarlyStopping(
            monitor='val_loss', patience=20, verbose=1, mode='auto')

        tensor_board = TensorBoard(
            log_dir='/tmp/face_recognition_v5/tf-{}'.format(time.time()),
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True)

        ntfy_progress = NtfyProgress()
        model_checkpoint = ModelCheckpoint(
            "/tmp/facebin-model-v5-val_acc:{val_acc:.2f}.h5",
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            period=1)
        self.classifier.fit_generator(
            generator=training_generator,
            use_multiprocessing=True,
            validation_data=validation_generator,
            validation_steps=100,
            epochs=nb_epoch,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            callbacks=[
                stop_early, tensor_board, ntfy_progress, model_checkpoint
            ])

    def train(self):
        "This does NOTHING as we just use Norm of encodings for prediction"
        pass

    # def train(self):
    #     if self.classifier_type == "knn":
    #         cc = self._knn()
    #     elif self.classifier_type == "random_forest":
    #         cc = self._random_forest()
    #     elif self.classifier_type == "asgd":
    #         cc = self._asgd()
    #     elif self.classifier_type == "mlp":
    #         cc = self._mlp()
    #     elif self.classifier_type == "gaussian_process":
    #         cc = self._gaussian_process()
    #     else:
    #         log.warning("Unrecognized Classifier Type: %s Default to KNN",
    #                     self.classifier_type)
    #         cc = self._knn()

    #     self.classifier = CalibratedClassifierCV(cc)
    #     df = self.dataset.get_user_dataframe()
    #     face_rows = df[df.loc["is_face_image"] == True]
    #     feature = face_rows['feature'].values
    #     person_id = face_rows['person_id'].values

    #     y = np.ravel(self.dataset.person_ids())
    #     log.debug(y.shape)
    #     log.debug(y)
    #     self.classifier.fit(self.dataset.feature_vectors(), y)

    def load_or_train(self, retrain=False):
        def refresh():
            self.create_model()
            self.train()
            self.save_model()

        log.debug("Adding missing face features:") 
        self.dataset.add_missing_face_features()

        if retrain:
            log.warning("Retraining")
            refresh()
        elif not os.path.exists(self.model_filename):
            log.warning("Cannot Find Stored Model File, retraining")
            refresh()
        elif os.path.getmtime(self.model_filename) < os.path.getmtime(
                self.dataset.FEATURE_FILE):
            log.warning("Dataset file is newer than the model. Retraining.")
            refresh()
        else:
            load_f = open(self.model_filename, 'rb')
            self.classifier = pickle.load(load_f)

    def save_model(self):
        save_f = open(self.model_filename, 'wb')
        pickle.dump(self.classifier, save_f)

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
        results = np.empty(shape=(faces.shape[0], ), dtype=np.int)
        dists = np.empty(shape=(faces.shape[0], ), dtype=np.float32)
        encodings = np.empty(
            shape=(faces.shape[0], self.dataset.FEATURE_SIZE),
            dtype=np.float32)
        for j in range(faces.shape[0]):
            log.debug("faces[0].shape: %s", faces[0].shape)
            encoded = self.encode(faces[j])
            log.debug("encoded.shape: %s", encoded.shape)
            c, d = self._predict_by_euclidean_distance(encoded)
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
    recognizer = FaceRecognizer_v6()
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
    fr = FaceRecognizer_v6(user_root=training_dir)
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
