import cv2
import dlib
import numpy as np
import utils
import os
import time

import sys

import tensorflow as tf
import label_map_util

import redis

import redis_queue_utils as rqu

import logging

log = utils.init_logging()


class FaceDetectorTensorflow:
    def __init__(self):
        self.DETECTION_THRESHOLD = 0.6
        self.PATH_TO_MODEL = "./frozen_inference_graph_face.pb"
        self.PATH_TO_LABELS = './face_label_map.pbtxt'
        self.NUM_CLASSES = 2
        self.TARGET_WIDTH = 480
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map,
            max_num_classes=self.NUM_CLASSES,
            use_display_name=True)
        self.category_index = label_map_util.create_category_index(
            self.categories)

        # Read the model file and build the model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            self.sess = tf.Session(
                graph=self.detection_graph, config=utils.tensorflow_config())
            self.windowNotSet = True

    def export_tflite(self):
        assert self.detection_graph
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        # We need a definite shape for input

        image_tensor.set_shape((640, 480, 100, 3))
        input_tensors = [image_tensor]
        output_tensors = [boxes, scores, classes, num_detections]
        output_node_names = ['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections']
        for t in input_tensors + output_tensors:
            print(t.shape)

        with tf.Session(graph=self.detection_graph) as sess:
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess, sess.graph_def, output_node_names)

        tflite_model = tf.lite.toco_convert(frozen_graph_def, input_tensors, output_tensors)

        open("facebin_detect_v1.tflite", "wb").write(tflite_model)


    def detect_faces(self, image: np.ndarray):
        # We reduce the size of the image
        larger_side = max(image.shape[0], image.shape[1])
        resize_ratio = self.TARGET_WIDTH / larger_side
        image_resized = cv2.resize(
            image, fx=resize_ratio, fy=resize_ratio, dsize=None)
        if log.getEffectiveLevel() == logging.DEBUG:
            cv2.imwrite("/tmp/face-detection-resized.png", image_resized)
        log.debug("resize_ratio: %s", resize_ratio)
        log.debug("image.shape: %s", image.shape)
        log.debug("image_resized.shape: %s", image_resized.shape)

        image_np = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')
        # start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # elapsed_time = time.time() - start_time
        # print('face detection time cost: {}'.format(elapsed_time))
        # log.debug("Boxes: %s ", boxes)
        # log.debug("Boxes.shape %s", boxes.shape)
        # log.debug("Scores: %s ", scores)
        # log.debug("Scores shape: %s ", scores.shape)
        # log.debug("max score: %s", scores.max())
        # log.debug("scores > 0.70: %s", scores[scores > 0.7])
        # log.debug("Classes shape: %s", classes.shape)
        # log.debug("Num Detections: %s", num_detections)
        # log.debug("classes where scores > 0.7: %s", classes[scores > 0.7])

        result_boxes_raw = boxes[scores > self.DETECTION_THRESHOLD]
        log.debug("result_boxes_raw.shape: %s", result_boxes_raw.shape)
        log.debug("result_boxes_raw: %s", result_boxes_raw)

        result_boxes = np.zeros(shape=result_boxes_raw.shape, dtype=np.int)
        # be careful about x, y!
        xs, ys, cs = image.shape
        xmin, ymin, xmax, ymax = (result_boxes_raw[:, 0],
                                  result_boxes_raw[:, 1],
                                  result_boxes_raw[:, 2],
                                  result_boxes_raw[:, 3])

        orig_w = (xmax - xmin)
        orig_h = (ymax - ymin)
        enlarge_x = orig_w * 0.20
        enlarge_y = orig_h * 0.20

        w = (xmax - xmin + 2 * enlarge_x)
        h = (ymax - ymin + 2 * enlarge_y)
        x = (xmin - enlarge_x)
        y = (ymin - enlarge_y)
        w[w > 1] = 1
        h[h > 1] = 1
        x[x < 0] = 0
        y[y < 0] = 0

        result_boxes = np.empty(shape=result_boxes.shape, dtype=np.int)
        result_boxes[:, 0] = np.floor(x * xs)
        result_boxes[:, 1] = np.floor(y * ys)
        result_boxes[:, 2] = np.floor(w * xs)
        result_boxes[:, 3] = np.floor(h * ys)
        log.debug("result_boxes: %s", result_boxes)
        log.debug("result_boxes.shape: %s", result_boxes.shape)
        log.debug("result_boxes.dtype: %s", result_boxes.dtype)
        return result_boxes


class FaceDetectorHaar:
    def __init__(self, haar_cascade_filepath=None):
        if haar_cascade_filepath is None:
            self._file_path = "haarcascade_frontalface_default.xml"
        else:
            self._file_path = haar_cascade_filepath

        self._min_size = (30, 30)
        self.classifier = cv2.CascadeClassifier(self._file_path)

    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        log.debug("Face Detector image shape: {}".format(image.shape))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        faces = self.classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.3,
            minNeighbors=4,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self._min_size)
        print(faces)
        return faces


class FaceDetectorDlib:
    def __init__(self):
        self.max_size_for_dlib = 360
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image):
        log.debug(type(image))
        resize_ratio = max(image.shape[0] // self.max_size_for_dlib,
                           image.shape[1] // self.max_size_for_dlib)

        if resize_ratio == 0:
            resize_ratio = 1

        scale_ratio = 1 / resize_ratio
        img = cv2.resize(image, dsize=None, fx=scale_ratio, fy=scale_ratio)

        dets = self.detector(img, 1)
        faces = []
        for d in dets:
            x = d.left() * resize_ratio
            y = d.top() * resize_ratio
            ## x and y can be negative for partial faces, so we check it here
            ## https://github.com/davisking/dlib/issues/767
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            w = d.right() * resize_ratio - x - 1
            h = d.bottom() * resize_ratio - y - 1
            faces.append((x, y, w, h))
        return np.floor(np.array(faces)).astype(np.int)


class FaceDetectorDlibCNN:
    """This one uses dlib's CNN face detector but it's much slower than
`frontal_face_detector` we used above in CPU setting. (Didn't test CUDA yet.)
From the initial tests, it doesn't bring much to the project.

    """

    def __init__(self):
        self.max_size_for_dlib = 360
        dlib_cnn_model = os.path.expandvars("mmod_human_face_detector.dat")
        self.detector = dlib.cnn_face_detection_model_v1(dlib_cnn_model)

    def detect_faces(self, image):
        log.debug(type(image))
        resize_ratio = max(image.shape[0] // self.max_size_for_dlib,
                           image.shape[1] // self.max_size_for_dlib)

        if resize_ratio == 0:
            resize_ratio = 1

        scale_ratio = 1 / resize_ratio
        img = cv2.resize(image, dsize=None, fx=scale_ratio, fy=scale_ratio)

        dets = self.detector(img, 1)
        faces = []
        for d in dets:
            x = d.rect.left() * resize_ratio
            y = d.rect.top() * resize_ratio
            ## x and y can be negative for partial faces, so we check it here
            ## https://github.com/davisking/dlib/issues/767
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            w = d.rect.right() * resize_ratio - x - 1
            h = d.rect.bottom() * resize_ratio - y - 1
            faces.append((x, y, w, h))
        return np.floor(np.array(faces)).astype(np.int)


def faces_from_image_v2(image: np.ndarray,
                        resize=(224, 224),
                        largest_only=False,
                        detector=None):
    if detector is None:
        detector = FaceDetectorTensorflow()
    coords = detector.detect_faces(image)
    if largest_only and coords.shape[0] > 0:
        sizes = coords[:, 2] * coords[:, 3]
        largest_i = np.argmax(sizes)
        coords = np.expand_dims(coords[largest_i], 0)

    n_coords = coords.shape[0]

    face_imgs = np.zeros(
        shape=(n_coords, resize[0], resize[1], 3), dtype=image.dtype)

    for i in range(n_coords):
        x, y, w, h = coords[i]
        log.debug("x, y, w, h: %s, %s, %s, %s", x, y, w, h)
        window = image[x:(x + w), y:(y + h)]
        log.debug("window.shape: %s", window.shape)
        face_imgs[i] = cv2.resize(
            window, dsize=resize, interpolation=cv2.INTER_AREA)

    return (face_imgs, coords)


def faces_from_image(image: np.ndarray,
                     largest_only=False,
                     detector=None,
                     resize_to=None):
    """Finds faces on the image and return images and coordinates

    >>> fn = os.path.expandvars("$HOME/Repository/facebin/test-input/face-image-1.jpg")
    >>> img = cv2.imread(fn)
    >>> faces, coords = faces_from_image(img)
    >>> assert(len(faces) == len(coords))
    >>> assert(len(faces) > 0)
    >>> print(faces[0].shape)
    (320, 320, 3)

    """
    if detector is None:
        detector = FaceDetectorDlib()
    faces = detector.detect_faces(image)
    log.debug(faces)
    if largest_only and faces.shape[0] > 0:
        sizes = faces[:, 2] * faces[:, 3]
        largest_i = np.argmax(sizes)
        faces = np.expand_dims(faces[largest_i], 0)
    face_imgs = {}
    face_coords = {}
    for i in range(faces.shape[0]):
        y = faces[i, 0]
        x = faces[i, 1]
        h = faces[i, 2]
        w = faces[i, 3]
        face_imgs[i] = np.copy(image[x:(x + w), y:(y + h)])
        log.debug(face_imgs[i].shape)
        assert (np.max(face_imgs[i]) <= 255)
        face_coords[i] = (x, y, w, h)
    return (face_imgs, face_coords)


def faces_from_image_file_v2(image_filename: str,
                             largest_only=False,
                             detector=None,
                             resize=None):
    image = cv2.imread(image_filename)
    if image is None:
        return (None, None)
    else:
        return faces_from_image_v2(
            image, largest_only=largest_only, detector=detector, resize=resize)


def faces_from_image_file(image_filename: str,
                          largest_only=False,
                          detector=None):
    image = cv2.imread(image_filename)
    return faces_from_image(image, largest_only, detector)


def faces_from_video_file(video_filename: str):
    """ get the list of face images from video keyframes

    @return list of ndarrays
    >>> video_filename = os.path.expandvars("$HOME/Repository/facebin/test-input/"
        "camera-2018-10-23-11-44-19-470021.mpeg")
    >>> faces = faces_from_video_file(video_filename)
    >>> len(faces)
    95
    """
    frame_filenames = utils.extract_keyframes(video_filename)
    detector = FaceDetectorDlib()
    result = []
    for f in sorted(frame_filenames):
        log.debug(f)
        img = cv2.imread(f)
        assert (img is not None)
        face_imgs, face_coords = faces_from_image(img, False, detector)

        for k, v in face_imgs.items():
            log.debug(v.shape)
            assert (np.max(v) <= 255)
            result.append(v)
        os.remove(f)

    return result

# def face_detection_loop():
#     R = redis.Redis(host='localhost', port=6379)
#     detector = FaceDetectorDlib()

#     while True:
#         l = R.llen('framekeys')
#         key = R.lpop('framekeys')
#         if key is None:
#             continue
#         log.debug("Face Detection len: %s, key: %s", l, key)
#         frame = rqu.get_framekey(R, key, True)
#         if frame is None:
#             continue

#         face_imgs, face_coords = faces_from_image(
#             frame['image'], detector=detector)
#         log.debug("face_coords: %s", face_coords)
#         for face_i in face_coords:
#             face_img = face_imgs[face_i]
#             face_xywh = face_coords[face_i]
#             log.debug("face_i: %s", face_i)
#             log.debug("face_img.shape: %s", face_img.shape)
#             log.debug("face_img.tostring(): %s", face_img.tostring())
#             log.debug("face_xywh: %s", face_xywh)
#             facekey = 'face:{}:{}'.format(key, face_i)
#             R.hmset(
#                 facekey, {
#                     'framekey': key,
#                     'x': int(face_xywh[0]),
#                     'y': int(face_xywh[1]),
#                     'w': int(face_xywh[2]),
#                     'h': int(face_xywh[3]),
#                     'shape_x': face_img.shape[0],
#                     'shape_y': face_img.shape[1],
#                     'shape_z': face_img.shape[2],
#                     'dtype': str(face_img.dtype),
#                     'face_img': face_img.tostring()
#                 })
#             R.rpush('facekeys', facekey)
#             log.debug("R.llen(facekeys): %s", R.llen('facekeys'))

if __name__ == "__main__":
     # face_detection_loop()
     fd = FaceDetectorTensorflow()
     fd.export_tflite()
