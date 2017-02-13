from src.vehicle_detection import feature_extraction
from config import *
from src import sliding_windows
import cv2
import numpy as np


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = feature_extraction.single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def detect_vehicle_by_clf(image, clf, X_scaler, frame_tracker):
    hot_windows = []


    # search close cars
    near_windows = sliding_windows.slide_window(image, x_start_stop=[None, None], y_start_stop=near_y_start_stop,
                                                xy_window=near_xy_window_size, xy_overlap=near_xy_overlap)
    hot_windows.extend(search_windows(image, near_windows, clf, X_scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat))

    # search middle close cars
    middle_windows = sliding_windows.slide_window(image, x_start_stop=[None, None], y_start_stop=middle_y_start_stop,
                                                  xy_window=middle_xy_window_size, xy_overlap=middle_xy_overlap)
    hot_windows.extend(search_windows(image, middle_windows, clf, X_scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat))

    # search far Vehicle
    far_windows = sliding_windows.slide_window(image, x_start_stop=[None, None], y_start_stop=far_y_start_stop,
                                               xy_window=far_xy_window_size, xy_overlap=far_xy_overlap)
    hot_windows.extend(search_windows(image, far_windows, clf, X_scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat))

    return hot_windows






