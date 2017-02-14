# vehicle traffic configuration.
import numpy as np
import cv2

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off



# train and test image path
train_vehicle_paths = ['./media/vehicle_img/vehicles/KITTI*/*.png']
train_vehicle_paths = ['./media/vehicle_img/vehicles/**/*.png']

test_vehicle_paths = ['./media/vehicle_img/vehicles/GTI*/*.png']

train_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/Extras/*.png']
train_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/**/*.png']
test_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/GTI/*.png']

# train test split ratio for classifier
test_size = 0.25

nb_recent_heatmap = 10
# For combined frame values, how many are remained.
combined_frame_threshold = 55


# classifier model file.
model_path = 'model.pkl'


input_video = "./media/videos/project_video.mp4"
output_video = './media/videos/project_video_output.mp4'
input_video = "./media/videos/test_3.mp4"
output_video = './media/videos/test_3_output.mp4'

# vehicle search window configuration
## near windows
near_y_start_stop = [500, None]
near_xy_window_size = (220, 200)
near_xy_overlap = (0.8, 0.8)

## middle distance windows
middle_y_start_stop = (400, 600)
middle_xy_window_size = (125, 100)
middle_xy_overlap = (0.8, 0.8)

## far away distance windows
far_y_start_stop = (400, 550)
far_xy_window_size = (62, 50)
far_xy_overlap = (0.8, 0.8)


# for lane line detection
# parameters for camera calibration
corner_x = 9 # how many corners in x direction of the chess board
corner_y = 6 # how many corners in y direction of the chess board

# define some normal variables
ksize = 15 # kernel size for image gradient thresholds

x = 1280 # image width
y = 720 # image height

# sliding windows constants
width = 120
height = 120
stride = 10
search_range = 300

# perspective transformation
offset_top = 590 # top left & right offset in original image
offset_btm = 200 # bottom left & right offset in original image
offset_dst = 300 # offset in bird view image

# meters per pixel (Convert from picture to real world distance)
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# detected lane checking params
curvature_diff_tolerance = 2000 # difference tolerance between current image with average of near previous ones.
distance_tolerance = 1000 # how many pixels are allowed between left lane bottom and right lane bottom in current image.
continue_miss_threshold = 30 # how many continuous undetermined images are allowed before trash previous imgs

# get perspective transform M and inverse perspective transform Minv
src = np.float32([(offset_top, 450), (x-offset_top, 450), (x-offset_btm, y), (offset_btm, y)])
dst = np.float32([(offset_dst, 0), (x-offset_dst, 0), (x-offset_dst, y), (offset_dst, y)])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
