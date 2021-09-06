#!/usr/bin/env python3
"""Image classifier tuning and evaluation, performed on a 2018 Macbook Pro."""
from datetime import datetime
import h5py
import json
import numpy as np
import pandas as pd
from typing import Tuple
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data into numpy arrays."""
    with h5py.File('Algorithm/Input/train/images_training.h5', 'r') as H:
        X_train = np.copy(H['datatrain'])
    with h5py.File('Algorithm/Input/train/labels_training.h5', 'r') as H:
        y_train = np.copy(H['labeltrain'])
    with h5py.File('Algorithm/Input/test/images_testing.h5', 'r') as H:
        X_test = np.copy(H['datatest'])
    with h5py.File('Algorithm/Input/test/labels_testing_2000.h5', 'r') as H:
        y_test_2000 = np.copy(H['labeltest'])
    return X_train, y_train, X_test, y_test_2000


def preprocess(X: np.ndarray) -> np.ndarray:
    """Do data cleaning, feature extraction etc."""
    features = [hog(x.reshape(28, 28)) for x in X]
    return np.array(features)
    features = [
        hog(x.reshape(28, 28),
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(3, 3),
            block_norm='L1-sqrt') for x in X
    ]
    return np.concatenate((normalize(X), np.array(features)), axis=1)


# Define all the ML algorithms to compare.
ALGORITHMS = {
    'Naive Bayes': (GaussianNB(), {}),
    'SGD': (SGDClassifier(penalty='elasticnet', n_jobs=-1, random_state=0), {
        'loss': ['hinge', 'log'],
        'l1_ratio': [0, 0.15, 0.5, 1],
        'class_weight': [None, 'balanced']
    }),
    'Decision Tree': (DecisionTreeClassifier(random_state=0), {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'ccp_alpha': [0, 0.001, 0.01, 0.1]
    }),
    'Nearest Neighbor': (KNeighborsClassifier(n_jobs=-1), {
        'n_neighbors': [1, 2, 3, 4],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }),
    'Bagging': (BaggingClassifier(random_state=0, n_jobs=-1), {
        'n_estimators': [1, 2, 4, 8, 16, 32, 64, 128],
        'max_samples': [0.25, 0.5]
    }),
    'Ada Boost': (AdaBoostClassifier(random_state=0), {
        'n_estimators': [1, 2, 4, 8, 16, 32, 64, 128],
        'learning_rate': [0.1, 1]
    }),
}

# Load and preprocess the data.
X_train, y_train, X_test, y_test_2000 = load_data()
X_train, X_test, best_model = preprocess(X_train), preprocess(X_test), None

# Run hyperparam tuning on each respective ML algorithm.
for name, (algorithm, param_grid) in ALGORITHMS.items():
    model = GridSearchCV(algorithm,
                         param_grid,
                         scoring='accuracy',
                         n_jobs=-1,
                         cv=10)
    model.fit(X_train, y_train)
    print(json.dumps({
        'ts': str(datetime.now()),
        'msg': 'gridsearch',
        'name': name,
        'score': model.best_score_,
        'time_seconds': model.refit_time_,
        'ix': int(model.best_index_),
        'params': model.best_params_
    }),
          flush=True)
    pd.DataFrame(model.cv_results_).to_csv(f'Algorithm/Output/{name}.csv')
    if best_model is None or model.best_score_ > best_model.best_score_:
        best_model = model

# Make test predictions using the best model and evaluate accuracy.
y_test_pred = best_model.predict(X_test)
score = accuracy_score(y_test_2000, y_test_pred[:2000])
print(json.dumps({
    'ts': str(datetime.now()),
    'msg': 'accuracy',
    'accuracy': score
}),
      flush=True)

# Write test predictions to file.
with h5py.File('Algorithm/Output/predicted_labels.h5', 'w') as H:
    H.create_dataset('Output', data=y_test_pred)
