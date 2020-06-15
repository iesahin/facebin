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

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
# from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

# log.debug(
#     "mem_frac: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     per_process_gpu_memory_fraction)
# log.debug(
#     "allow_growth: %s",
#     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
#     allow_growth)

# import matplotlib.pyplot as plt


class VGGEncoder:
    def __init__(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        model.load_weights('vgg_face_weights.h5')

        self.model = model

        self.vgg_face_descriptor = Model(
            inputs=self.model.layers[0].input,
            outputs=self.model.layers[-2].output)

        self.cosine_epsilon = 0.40
        # log.debug(
        #     "mem_frac: %s",
        #     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
        #     per_process_gpu_memory_fraction)
        # log.debug(
        #     "allow_growth: %s",
        #     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
        #     allow_growth)

    # def preprocess_image(self, image_path):
    #     img = load_img(image_path, target_size=(224, 224))
    #     img = img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     img = preprocess_input(img)
    #     return img

    def encode(self, img):
        "img: a 224x224 image"
        # log.debug(
        #     "mem_frac: %s",
        #     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
        #     per_process_gpu_memory_fraction)
        # log.debug(
        #     "
        #     allow_growth: %s",
        #     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
        # allow_growth)
        # arr = img_to_array(img)
        arr = np.expand_dims(img, axis=0)
        arr = preprocess_input(arr)
        # log.debug(
        #     "mem_frac: %s",
        #     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
        #     per_process_gpu_memory_fraction)
        # log.debug(
        #     "allow_growth: %s",
        #     keras.backend.tensorflow_backend.get_session()._config.gpu_options.
        #     allow_growth)

        return self.vgg_face_descriptor.predict(arr)

    def findCosineSimilarity(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def findEuclideanDistance(self, source_representation,
                              test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(
            np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    # def verifyFace(img1, img2):
    #     img1_representation = vgg_face_descriptor.predict(preprocess_image('C:/Users/IS96273/Desktop/trainset/%s' % (img1)))[0,:]
    #     img2_representation = vgg_face_descriptor.predict(preprocess_image('C:/Users/IS96273/Desktop/trainset/%s' % (img2)))[0,:]
    #     cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    #     euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    #     print("Cosine similarity: ",cosine_similarity)
    #     print("Euclidean distance: ",euclidean_distance)
    #     if(cosine_similarity < epsilon):
    #         print("verified... they are same person")
    #     else:
    #         print("unverified! they are not same person!")
    #     f = plt.figure()
    #     f.add_subplot(1,2, 1)
    #     plt.imshow(image.load_img('C:/Users/IS96273/Desktop/trainset/%s' % (img1)))
    #     plt.xticks([]); plt.yticks([])
    #     f.add_subplot(1,2, 2)
    #     plt.imshow(image.load_img('C:/Users/IS96273/Desktop/trainset/%s' % (img2)))
    #     plt.xticks([]); plt.yticks([])
    #     plt.show(block=True)
    #     print("-----------------------------------------")
