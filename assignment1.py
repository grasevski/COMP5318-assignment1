#!/usr/bin/env python3
"""Image classifier tuning and evaluation, performed on a 2018 Macbook Pro."""
import csv
from datetime import datetime
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple
from skimage import exposure
from skimage.feature import hog
from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data into numpy arrays."""
    with h5py.File('Input/train/images_training.h5', 'r') as H:
        X_train = np.copy(H['datatrain'])
    with h5py.File('Input/train/labels_training.h5', 'r') as H:
        y_train = np.copy(H['labeltrain'])
    with h5py.File('Input/test/images_testing.h5', 'r') as H:
        X_test = np.copy(H['datatest'])
    with h5py.File('Input/test/labels_testing_2000.h5', 'r') as H:
        y_test_2000 = np.copy(H['labeltest'])
    return X_train, y_train, X_test, y_test_2000


def to_image(x: np.ndarray) -> np.ndarray:
    """Convert input vector to 2D image."""
    return x.reshape(28, 28)


def generate_hog_image(X: np.ndarray) -> None:
    """Create comparison image of original and HOG transform."""
    plt.close()
    plt.figure()
    for i, x in enumerate(X):
        plt.subplot(len(X), 2, 2 * i + 1)
        image = to_image(x)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
        if i == 0:
            plt.title('Original')
        plt.subplot(len(X), 2, 2 * i + 2)
        _, hog_image = hog(image, **HOG_PARAMS, visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image,
                                                        in_range=(0, 10))
        plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
        if i == 0:
            plt.title('HOG')
    plt.savefig('Output/hog.png')


def preprocess(X: np.ndarray) -> np.ndarray:
    """Clean data and extract features."""
    return np.array([hog(to_image(x), **HOG_PARAMS) for x in X])


# Preprocessing hyperparameters.
HOG_PARAMS = {
    'orientations': 8,
    'pixels_per_cell': (4, 4),
    'block_norm': 'L1-sqrt',
    'transform_sqrt': True,
}

# Define all the ML algorithms to compare. This can be edited to
# remove combinations, to speed up the script.
ALGORITHMS = {
    'Naive Bayes': (GaussianNB(), {}),
    'SVM': (SVC(random_state=0), {
        'kernel': ['rbf'],
        'C': [10]
    }),
    # 'Logistic Regression':
    # (LogisticRegression(solver='saga', n_jobs=-1, random_state=0), {
    #     'penalty': ['l1', 'l2'],
    #     'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # }),
    # 'SVM': (SVC(random_state=0), {
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #     'C': [1, 10, 100, 1000]
    # }),
}

# Load the data.
print(json.dumps({'ts': str(datetime.now()), 'msg': 'load_data'}), flush=True)
X_train, y_train, X_test, y_test_2000 = load_data()

# Generate example preprocessed image and save in Output directory.
print(json.dumps({
    'ts': str(datetime.now()),
    'msg': 'generate_hog_image'
}),
      flush=True)
generate_hog_image(X_train[:3])

# Preprocessing.
print(json.dumps({'ts': str(datetime.now()), 'msg': 'preprocess'}), flush=True)
t0 = time.time()
X_train, X_test = preprocess(X_train), preprocess(X_test)
pca = PCA(n_components=128, random_state=0).fit(X_train)
X_train, X_test = pca.transform(X_train), pca.transform(X_test)
time_seconds = time.time() - t0
print(json.dumps({
    'ts': str(datetime.now()),
    'msg': 'preprocess_done',
    'time_seconds': time_seconds
}),
      flush=True)

# Evaluate each respective ML algorithm.
print(json.dumps({'ts': str(datetime.now()), 'msg': 'tuning'}), flush=True)
fieldnames = [
    'ts', 'msg', 'name', 'test_acc', 'cv_acc', 'fit_time_seconds',
    'inf_time_seconds', 'ix'
]
best, top = None, -np.inf
with open('Output/results.csv', 'w') as f:
    wtr = csv.DictWriter(f, fieldnames=fieldnames)
    wtr.writeheader()
    for name, (algorithm, param_grid) in ALGORITHMS.items():
        model = GridSearchCV(algorithm,
                             param_grid,
                             scoring='accuracy',
                             n_jobs=-1,
                             cv=10)
        model.fit(X_train, y_train)
        t0 = time.time()
        y_test_pred = model.predict(X_test)
        inf_time_seconds = time.time() - t0
        acc = accuracy_score(y_test_2000, y_test_pred[:2000])
        if acc > top:
            best, top = y_test_pred, acc
        results = {
            'ts': str(datetime.now()),
            'msg': 'gridsearch',
            'name': name,
            'test_acc': acc,
            'cv_acc': model.best_score_,
            'fit_time_seconds': model.refit_time_,
            'inf_time_seconds': inf_time_seconds,
            'ix': int(model.best_index_),
        }
        wtr.writerow(results)
        results['params'] = model.best_params_
        print(json.dumps(results), flush=True)
        df = pd.DataFrame(model.cv_results_)
        df.drop(columns='params', inplace=True)
        df.to_csv(f'Output/{name}.csv')

# Write test predictions to file.
with h5py.File('Output/predicted_labels.h5', 'w') as H:
    H.create_dataset('Output', data=best)
