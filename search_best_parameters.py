from time import time

from moviepy.editor import VideoFileClip

import track_vehicle_single_frame
from config import *
from src.vehicle_detection import heatmap_funcs
from src.vehicle_detection import train_classfier
from src.vehicle_detection.VehicleDetectionHeatMap import VehicleDetectionHeatMap

classifier, X_scaler = train_classfier.get_classifier_n_scaler(model_path)

nb_frame_n_threshold = ((10, 15), (5, 20), (5, 30), (10, 20), (10, 30), (10, 40), (10, 50), (15, 40))
nb_frame_n_threshold = ((5, 5), (5, 8))
video_input_list = ('test_3.mp4', )
# video_input_list = ("./media/videos/test_3.mp4", '')
# video_input_list = ("./media/videos/test_1.mp4", '')

for video_num, input_video in enumerate(video_input_list):
    full_input_video = './media/videos/' + input_video
    for nb_recent_heatmap, combined_frame_threshold in nb_frame_n_threshold:
        nb_recent_heatmap = nb_recent_heatmap
        combined_frame_threshold = combined_frame_threshold
        frame_tracker = VehicleDetectionHeatMap()

        def detect_vehicle(image):
            # get a heatmap from a independent frame
            single_heatmap = track_vehicle_single_frame.detect_vehicle_by_clf(image, classifier, X_scaler)
            # combine the heatmap into frame heat map tracker.
            vehicle_img = heatmap_funcs.mark_vehicles(image, single_heatmap, frame_tracker,
                                                      heat_threshold=combined_frame_threshold)
            return vehicle_img

        clip1 = VideoFileClip(full_input_video)
        white_clip = clip1.fl_image(detect_vehicle)
        t = time()
        # print('str(video_num + 1)', str(video_num + 1))
        input_video = input_video.split('.')[0]
        output_video = './media/videos/' + input_video + '_' + str(nb_recent_heatmap) + '_' + str(
            combined_frame_threshold) + '_simpler_output.mp4'
        white_clip.write_videofile(output_video, audio=False)
        print('It takes: ', time()-t, ' secs')
