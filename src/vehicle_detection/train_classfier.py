from time import time
import pickle
import os
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from src.vehicle_detection import feature_extraction
from src.vehicle_detection.heatmap_funcs import *
from config import *
from sklearn.feature_selection import RFECV


def feature_preparation():
    train_vehicle_features = feature_extraction.extract_features_dir(train_vehicle_paths)
    test_vehicle_features = feature_extraction.extract_features_dir(test_vehicle_paths)
    train_non_vehicle_features = feature_extraction.extract_features_dir(train_non_vehicle_paths)
    test_non_vehicle_features = feature_extraction.extract_features_dir(test_non_vehicle_paths)

    X_train = np.vstack((train_vehicle_features, train_non_vehicle_features)).astype(np.float64)

    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    y_train = np.hstack((np.ones(len(train_vehicle_features)), np.zeros(len(train_non_vehicle_features))))
    X_train, y_train = shuffle(X_train, y_train, random_state=43)

    X_test = np.vstack((test_vehicle_features, test_non_vehicle_features)).astype(np.float64)
    X_test = X_scaler.transform(X_test)
    y_test = np.hstack((np.ones(len(test_vehicle_features)), np.zeros(len(test_non_vehicle_features))))
    X_test, y_test = shuffle(X_test, y_test, random_state=43)
    return X_train, y_train, X_test, y_test, X_scaler


def train_classfier(X_train, y_train, X_test, y_test):
    svc = LinearSVC()
    clf = RFECV(svc, step=0.1, cv=7, n_jobs=-1)
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    t = time.time()
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    print('time takes: ', time.time() - t)
    return clf

def get_classifier_n_scaler(path):
    if not os.path.isfile(path):
        X_train, y_train, X_test, y_test, X_scaler = feature_preparation()
        clf = train_classfier(X_train, y_train, X_test, y_test)
        model = {'model': clf, 'X_scaler': X_scaler}
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    else:
        pkl_content = pickle.load(open(path, 'rb'))
        clf = pkl_content['model']
        X_scaler = pkl_content['X_scaler']
    return clf, X_scaler
