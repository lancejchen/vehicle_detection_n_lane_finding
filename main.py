from moviepy.editor import VideoFileClip
from vehicle_tracking import *
from src.VehicleDetectionHeatMap import VehicleDetectionHeatMap
from config import *
from time import time

import pickle
import os

if not os.path.isfile('model.pkl'):
    X_train, y_train, X_test, y_test, X_scaler = feature_preparation()
    clf = train_classfier(X_train, y_train, X_test, y_test)
    model = {'model': clf, 'X_scaler': X_scaler}
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    pkl_content = pickle.load(open('model.pkl', 'rb'))
    clf = pkl_content['model']
    X_scaler = pkl_content['X_scaler']

frame_tracker = VehicleDetectionHeatMap()


def detect_vehicle(image):
    return detect_vehicle_by_clf(image, clf, X_scaler, frame_tracker)

white_output = './media/videos/vehicle_yeah_full.mp4'
clip1 = VideoFileClip("./media/videos/test_video.mp4")
white_clip = clip1.fl_image(detect_vehicle)
t = time()
white_clip.write_videofile(white_output, audio=False)
print('It takes: ', time()-t, ' secs')
