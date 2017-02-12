import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from src.vehicle_detection.search_classifier import *
from src.vehicle_detection.heatmap_funcs import *
from sklearn.feature_selection import RFECV


def feature_preparation():
    train_vehicle_paths = ['./media/vehicle_img/vehicles/KITTI*/*.png']
    test_vehicle_paths = ['./media/vehicle_img/vehicles/GTI*/*.png']
    train_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/Extras/*.png']
    test_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/GTI/*.png']

    train_vehicle_features = extract_features_dir(train_vehicle_paths)
    test_vehicle_features = extract_features_dir(test_vehicle_paths)
    train_non_vehicle_features = extract_features_dir(train_non_vehicle_paths)
    test_non_vehicle_features = extract_features_dir(test_non_vehicle_paths)

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


def detect_vehicle_by_clf(image, clf, X_scaler, frame_tracker):
    hot_windows = []
    near_y_start_stop = [500, None]

    # search close cars
    near_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=near_y_start_stop,
                                xy_window=(220, 200), xy_overlap=(0.8, 0.8))
    hot_windows.extend(search_windows(image, near_windows, clf, X_scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat))

    # search middle close cars
    middle_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 600],
                                  xy_window=(125, 100), xy_overlap=(0.8, 0.8))
    hot_windows.extend(search_windows(image, middle_windows, clf, X_scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat))

    # search far Vehicle
    far_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 550],
                               xy_window=(62, 50), xy_overlap=(0.67, 0.67))
    hot_windows.extend(search_windows(image, far_windows, clf, X_scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat))

    draw_img = mark_vehicles(image, hot_windows, frame_tracker, 30)

    return draw_img


