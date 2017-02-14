from time import time

from moviepy.editor import VideoFileClip

import track_vehicle_single_frame
from config import *
from src.vehicle_detection import heatmap_funcs
from src.vehicle_detection import train_classfier
from src.vehicle_detection.VehicleDetectionHeatMap import VehicleDetectionHeatMap
import detect_lane_line
from src.Line import Line

left_lane = Line('left')
right_lane = Line('right')
classifier, X_scaler = train_classfier.get_classifier_n_scaler(model_path)
frame_tracker = VehicleDetectionHeatMap()


def detect_vehicle(image):
    image, lanes = detect_lane_line.detect_lane_line_pipeline(image, left_lane, right_lane)
    # get a heatmap from a independent frame
    single_heatmap = track_vehicle_single_frame.detect_vehicle_by_clf(image, classifier, X_scaler)
    # combine the heatmap into frame heat map tracker.
    vehicle_img = heatmap_funcs.mark_vehicles(image, single_heatmap, frame_tracker,
                                              heat_threshold=combined_frame_threshold)
    vehicle_img = cv2.addWeighted(vehicle_img, 1, lanes, 0.3, 0)
    return vehicle_img

clip1 = VideoFileClip(input_video)
white_clip = clip1.fl_image(detect_vehicle)
t = time()
white_clip.write_videofile(output_video, audio=False)
print('It takes: ', time()-t, ' secs')
