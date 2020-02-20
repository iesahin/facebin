import numpy as np
import PySide2.QtGui as qtg
import cv2
import datetime
import os
import random
from utils import *

log = init_logging()


class DatasetManager:
    """Dataset manager to retrieve images, feature vectors, person ids and metadata
about the image dataset.

    >>> ds_fn = "test-input/facebin_dataset-{}.npz".format(datetime.datetime.now().strftime('%s'))
    >>> ds = DatasetManager(ds_fn)
    >>> assert(ds.DATASET_PATH == ds_fn)
    >>> ds.IMAGE_SIZE
    (224, 224)
    >>> image1 = cv2.imread("test-input/face-image-1.jpg")
    >>> resized = cv2.resize(image1, dsize=ds.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    >>> feature_vector = np.random.rand(1, ds.FEATURE_VECTOR_SIZE)
    >>> person_id = 999
    >>> ds.add_item(resized, feature_vector, person_id)
    >>> ds.add_metadata("name.{}".format(person_id), "John Doe")
    >>> ds.get_metadata("name.{}".format(person_id))
    "John Doe"
    >>> ds2 = DatasetManager(ds_fn)
    >>> image2 = ds2.image_by_person_id(person_id)[0]
    >>> assert(resized == image2)
    >>> feature_vector2 = ds2.feature_vector_by_person_id(person_id)[0]
    >>> assert(feature_vector == feature_vector2)
    >>> os.remove(ds_fn)
    """

    def __init__(self, dataset_path="facebin_dataset.npz"):
        self.DATASET_PATH = dataset_path
        self.IMAGE_SIZE = (224, 224)
        self.FEATURE_VECTOR_SIZE = 2622
        if os.path.exists(self.DATASET_PATH):
            self.files = np.load(self.DATASET_PATH)
            self.images = self.files['images']
            self.feature_vectors = self.files['feature_vectors']
            self.person_ids = self.files['person_ids']
            self.metadata = self.files['metadata']
        else:
            self.images = np.zeros(
                (0, self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3), dtype=np.uint8)
            self.feature_vectors = np.zeros((0, self.FEATURE_VECTOR_SIZE),
                                            dtype=np.float)
            self.person_ids = np.zeros((0, 1), dtype=np.int)
            self.metadata = np.zeros(
                1,
                dtype={
                    'names': ('key', 'value'),
                    'formats': ('U50', 'U50')
                })
            self.metadata['key'][0] = 'dataset-initialization'
            self.metadata['value'][0] = datetime.datetime.now().strftime(
                '%x %X')
            self.save_dataset()

    def size(self):
        return self.person_ids.shape[0]

    def images_by_id(self, person_id):
        filter = self.person_ids == person_id
        filtered_images = np.copy(self.images[filter[:, 0]])
        return filtered_images

    def add_item(self, image: np.ndarray, feature_vector: np.ndarray,
                 person_id: int):
        assert (image.shape == (self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], 3))
        assert (feature_vector.shape == (1, self.FEATURE_VECTOR_SIZE))
        self.images = np.vstack([self.images, np.expand_dims(image, 0)])
        self.feature_vectors = np.vstack(
            [self.feature_vectors, feature_vector])
        self.person_ids = np.vstack([self.person_ids, np.array([person_id])])
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
            feature_vectors=self.feature_vectors,
            person_ids=self.person_ids,
            images=self.images,
            metadata=self.metadata)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
