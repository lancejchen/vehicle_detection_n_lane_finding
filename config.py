# vehicle traffic configuration.

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

# For combined frame values, how many are remained.
combined_frame_threshold = 40

# train and test image path
train_vehicle_paths = ['./media/vehicle_img/vehicles/KITTI*/*.png']
train_vehicle_paths = ['./media/vehicle_img/vehicles/**/*.png']

test_vehicle_paths = ['./media/vehicle_img/vehicles/GTI*/*.png']

train_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/Extras/*.png']
train_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/**/*.png']
test_non_vehicle_paths = ['./media/vehicle_img/non-vehicles/GTI/*.png']

# classifier model file.
model_path = 'model.pkl'


input_video = "./media/videos/test_video.mp4"
output_video = './media/videos/vehicle_yeah_full.mp4'

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
far_xy_overlap = (0.67, 0.67)
