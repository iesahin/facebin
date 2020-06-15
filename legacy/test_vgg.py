import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest
import os
import database_api as db
import face_recognition_v6 as fr6
import cv2
# import libKMCUDA as km
from utils import *
import sklearn.neighbors as skn
import sklearn.svm as svm
import sklearn.ensemble as ske
import sklearn.linear_model as sklm
import sklearn.neural_network as sknn
import sklearn.gaussian_process as skgp
from sklearn.model_selection import train_test_split, cross_val_score
import time

IMAGEDIR = os.path.expandvars(
    "$HOME/facebin-artifacts/dataset-images/user-test")


def testVGG16():
    keras.backend.set_image_dim_ordering('tf')
    model = VGGFace(model='vgg16')
    for d in os.scandir(IMAGEDIR):
        print("d: ", d.name)
        person_id = int(d.name[:-2])
        print(person_id)
        person = db.person_by_id(person_id)[0]
        for f in os.scandir(d.path):
            if f.name.endswith('.face.png'):
                print("f: ", f.name)
                img = image.load_img(f.path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=1)
                preds = model.predict(x)
                print("person: {} prediction: {}".format(
                    person[1],
                    utils.decode_predictions(preds)[0][0][0]))
                # print(preds)
                # print(np.argmax(preds))
                # self.assertIn('A.J._Buckley', utils.decode_predictions(preds)[0][0][0])
                # self.assertAlmostEqual(
                #     utils.decode_predictions(preds)[0][0][1], 0.9790116, places=3)


def prepare_dataset():
    recognizer = fr6.FaceRecognizer_v6()
    feature_size = 4096
    features = np.empty((0, feature_size), dtype=np.float32)
    person_ids = np.empty((0, ), dtype=np.int32)
    n_dirs = 0
    for d in os.scandir(IMAGEDIR):
        if d.is_dir():
            n_dirs += 1
            print("d: ", d.name)
            person_id = int(d.name[:-2])
            files = {
                f.name: f.path
                for f in os.scandir(d.path) if f.name.startswith('face-')
            }
            current_person_ids = np.ones(
                shape=(len(files), ), dtype=np.int32) * person_id
            current_features = np.empty(shape=(len(files), feature_size))
            for i, fn in enumerate(files):
                print("f: ", fn)
                img = cv2.imread(files[fn])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                current_features[i] = recognizer.encode(img)

            person_ids = np.hstack([person_ids, current_person_ids])
            features = np.vstack([features, current_features])

    return (features, person_ids)


def get_dataset():
    filename = 'facebin-test-data.npz'
    if os.path.exists(filename):
        ff = np.load(filename)
        features = ff['features']
        person_ids = ff['person_ids']
        return (features, person_ids)
    else:
        features, person_ids = prepare_dataset()
        np.savez(filename, features=features, person_ids=person_ids)
        return (features, person_ids)


def kmeans(features, person_ids):
    n_centers = len(np.unique(person_ids))
    centers = np.squeeze(np.array(features, dtype=np.float32))
    classes = np.array(person_ids, dtype=np.uint32)
    centroids, assignments, average_distance = km.kmeans_cuda(
        samples=centers, clusters=n_centers, average_distance=True)

    results['kmeans'] = {}
    results['kmeans']['centers'] = centers
    results['kmeans']['classes'] = classes
    results['kmeans']['centroids'] = centroids
    results['kmeans']['assignments'] = assignments
    results['kmeans']['average_distance'] = average_distance

    return results


def knn(neighbors=9, weights='uniform', metric='euclidean'):
    c = skn.KNeighborsClassifier(
        n_neighbors=neighbors, weights=weights, metric=metric, n_jobs=-1)
    return c


def random_forest(n_estimators=1000, max_depth=20, class_weight='balanced'):
    c = ske.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight)
    return c


def asgd(average=True,
         tol=0.00001,
         max_iter=100,
         loss='modified_huber',
         learning_rate='optimal',
         class_weight='balanced'):
    c = sklm.SGDClassifier(
        average=average,
        tol=tol,
        max_iter=max_iter,
        loss=loss,
        fit_intercept=True,
        shuffle=True,
        n_jobs=-1,
        learning_rate=learning_rate,
        class_weight=class_weight)
    return c


def mlp(
        hidden_layer_sizes=(1000, 500, 500, 500, 500),
        activation='relu',
        solver='adam'):
    c = sknn.MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver)
    return c


def gaussian_process(kernel=1.0 * skgp.kernels.RBF(1.0),
                     multi_class='one_vs_rest'):
    c = skgp.GaussianProcessClassifier(
        kernel=kernel, multi_class='one_vs_rest')
    return c


def measure_performances():
    """
<function knn at 0x7f33c7eae268>({'neighbors': 9, 'weights': 'uniform', 'metric': 'euclidean'})
Training Time:  7.070426940917969
Scores:  0.8108484005563282
Score Time:  104.75363612174988
<function knn at 0x7f33c7eae268>({'neighbors': 3, 'weights': 'uniform', 'metric': 'euclidean'})
Training Time:  6.9550323486328125
Scores:  0.7983310152990264
Score Time:  101.5170738697052
<function knn at 0x7f33c7eae268>({'neighbors': 1, 'weights': 'uniform', 'metric': 'euclidean'})
Training Time:  6.929074764251709
Scores:  0.8018080667593881
Score Time:  96.40605211257935
<function knn at 0x7f33c7eae268>({'neighbors': 20, 'weights': 'uniform', 'metric': 'euclidean'})
Training Time:  7.257203102111816
Scores:  0.8063282336578581
Score Time:  103.44514155387878
<function knn at 0x7f33c7eae268>({'neighbors': 100, 'weights': 'uniform', 'metric': 'euclidean'})
Training Time:  6.977663040161133
Scores:  0.7885952712100139
Score Time:  102.37434434890747
<function random_forest at 0x7f33c7eae2f0>({'n_estimators': 1000, 'max_depth': 20, 'class_weight': 'balanced'})
Training Time:  12022.396698236465
Scores:  0.8042420027816412
Score Time:  3.578932523727417
<function random_forest at 0x7f33c7eae2f0>({'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'})
Training Time:  634.4614391326904
Scores:  0.6310848400556328
Score Time:  0.3594486713409424
<function knn at 0x7f24717f11e0>({'neighbors': 9, 'weights': 'uniform', 'metric': 'minkowski'})
Training Time:  6.91987681388855
Scores:  0.8226703755215578
Score Time:  105.8181140422821
<function knn at 0x7f24717f11e0>({'neighbors': 9, 'weights': 'uniform', 'metric': 'manhattan'})
Training Time:  7.124099016189575
Scores:  0.8223226703755215
Score Time:  106.43413496017456
<function knn at 0x7f24717f11e0>({'neighbors': 9, 'weights': 'uniform', 'metric': 'chebyshev'})
Training Time:  7.241780042648315
Scores:  0.8122392211404729
Score Time:  73.98568725585938
<function knn at 0x7fa491038268>({'neighbors': 10, 'weights': 'distance', 'metric': 'euclidean'})
Training Time:  6.428001642227173
Scores:  0.8223226703755215
Score Time:  109.37942147254944
<function knn at 0x7fa491038268>({'neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'})
Training Time:  7.086570739746094
Scores:  0.8209318497913769
Score Time:  107.39100098609924
<function random_forest at 0x7fa4910382f0>({'n_estimators': 500, 'max_depth': 200, 'class_weight': 'balanced'})
Training Time:  19350.764025449753
Scores:  0.8372739916550765
Score Time:  2.6782658100128174
<function mlp at 0x7fa491038400>({'hidden_layer_sizes': (1000, 500, 500, 500, 500), 'activation': 'relu', 'solver': 'adam'})
Training Time:  758.5193777084351
Scores:  0.7461752433936022
Score Time:  0.23350954055786133
<function mlp at 0x7fa491038400>({'hidden_layer_sizes': (5000, 1000, 1000, 1000, 1000), 'activation': 'relu', 'solver': 'adam'})
Training Time:  10137.319849014282
Scores:  0.7016689847009736
Score Time:  0.951458215713501
<function mlp at 0x7fa491038400>({'hidden_layer_sizes': (1000, 500, 500), 'activation': 'relu', 'solver': 'adam'})
Training Time:  670.6056108474731
Scores:  0.741307371349096
Score Time:  0.19451093673706055
<function mlp at 0x7fa491038400>({'hidden_layer_sizes': (1000, 500, 500, 500, 500), 'activation': 'tanh', 'solver': 'adam'})
Training Time:  1500.5716524124146
Scores:  0.34735744089012516
Score Time:  0.4100217819213867
<function knn at 0x7fe3f75c0268>({'neighbors': 1, 'weights': 'uniform', 'metric': 'euclidean'})
Training Time:  6.492412805557251
Scores:  0.8063282336578581
Score Time:  104.40398144721985
<function mlp at 0x7fe3f75c0400>({'hidden_layer_sizes': (300, 300, 300), 'activation': 'relu', 'solver': 'adam'})
Training Time:  373.97151017189026
Scores:  0.7409596662030598
Score Time:  0.10064077377319336
<function mlp at 0x7fe3f75c0400>({'hidden_layer_sizes': (500, 500), 'activation': 'tanh', 'solver': 'adam'})
Training Time:  790.0764918327332
Scores:  0.6578581363004172
Score Time:  0.1674649715423584
<function asgd at 0x7fe3f75c0378>({'average': True, 'tol': 1e-05, 'max_iter': 100, 'loss': 'modified_huber', 'learning_rate': 'optimal', 'class_weight': 'balanced'})
Training Time:  632.382080078125
Scores:  0.8320584144645341
Score Time:  0.09457135200500488

    """

    function_params = [
        (knn, {
            'neighbors': 1,
            'weights': 'uniform',
            'metric': 'euclidean'
        }),
        (mlp, {
            "hidden_layer_sizes": (300, 300, 300),
            "activation": 'relu',
            "solver": "adam"
        }),
        (mlp, {
            "hidden_layer_sizes": (500, 500),
            "activation": 'tanh',
            "solver": "adam"
        }),
        # (gaussian_process, {
        #     'kernel': 1.0 * skgp.kernels.RBF(1.0)
        # }),
        (asgd, {
            'average': True,
            'tol': 0.00001,
            'max_iter': 100,
            'loss': 'modified_huber',
            'learning_rate': 'optimal',
            'class_weight': 'balanced'
        }),
    ]

    features, person_ids = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        features, person_ids, test_size=0.1)

    for func, params in function_params:
        print("{}({})".format(func, params))
        classifier = func(**params)
        t_begin = time.time()
        classifier.fit(X_train, y_train)
        t_end = time.time()
        print("Training Time: ", (t_end - t_begin))
        t_begin = time.time()
        scores = classifier.score(X_test, y_test)
        print("Scores: ", scores)
        t_end = time.time()
        print("Score Time: ", (t_end - t_begin))


if __name__ == '__main__':
    measure_performances()
