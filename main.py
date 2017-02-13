from moviepy.editor import VideoFileClip
from config import *
from src.VehicleDetectionHeatMap import VehicleDetectionHeatMap
from time import time
from src.vehicle_detection import train_classfier
import track_vehicle_single_frame
from src.vehicle_detection import heatmap_funcs


classifier, X_scaler = train_classfier.get_classifier_n_scaler(model_path)
frame_tracker = VehicleDetectionHeatMap()


def detect_vehicle(image):
    # get a heatmap from a independent frame
    single_heatmap = track_vehicle_single_frame.detect_vehicle_by_clf(image, classifier, X_scaler, frame_tracker)
    # combine the heatmap into frame heat map tracker.
    vehicle_img = heatmap_funcs.mark_vehicles(image, single_heatmap, frame_tracker,
                                              heat_threshold=combined_frame_threshold)
    return vehicle_img

clip1 = VideoFileClip(input_video)
white_clip = clip1.fl_image(detect_vehicle)
t = time()
white_clip.write_videofile(output_video, audio=False)
print('It takes: ', time()-t, ' secs')



