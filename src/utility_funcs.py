import glob
from matplotlib import pyplot as plt

from .img_gradient_threshold_funcs import *


def get_camera_mtx_dist(src, corner_x, corner_y, x, y):
    objp = np.zeros((corner_y * corner_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(src)
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (x, y), None, None)
    return mtx, dist


def plot_orig_and_processed_img(img, processed, orig_img_name, processed_img_name):
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title(orig_img_name, fontsize=30)
    ax2.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    ax2.set_title(processed_img_name, fontsize=30)


def get_highest_in_x(start_point, histogram, search_range, width, stride):
    dist = int((search_range - width) / 2)
    start = start_point - dist
    end = start_point + dist

    highest_range = []
    highest_range_arg = []

    for pos in range(start, end + 1, stride):
        leftx = pos
        rightx = leftx + width
        highest_range.append(np.sum(histogram[leftx:rightx]))
        highest_range_arg.append((leftx, rightx))
    # print('highest left range is: ', highest_left_range)
    index = np.argmax(highest_range)  # it's a range
    lane_pixel_pos = highest_range_arg[index][0]
    return lane_pixel_pos


def get_bottom_lane_marks(img, width, search_range, stride):
    histogram = np.sum(img[2 * int(img.shape[0] / 3):, :], axis=0)
    middle = int(histogram.shape[0] / 2)
    left_high = np.argmax(histogram[:middle])
    right_high = np.argmax(histogram[middle:]) + middle

    left_start_point = int(left_high - width / 2)
    right_start_point = int(right_high - width / 2)

    left_lane_pixel_pos = get_highest_in_x(left_start_point, histogram, search_range, width, stride)
    right_lane_pixel_pos = get_highest_in_x(right_start_point, histogram, search_range, width, stride)
    return left_lane_pixel_pos, right_lane_pixel_pos


def find_lane_patches(img, last_left_x, last_right_x, current_y, height, search_range, width, stride):
    """ Given previous patch location, find best sliding window above it
        param: img, previous location width, height
        return: left start point of the maximum sliding window
    """
    histogram = np.sum(img[current_y: current_y + height, :], axis=0)
    left_lane_pixel_pos = get_highest_in_x(last_left_x, histogram, search_range, width, stride)
    right_lane_pixel_pos = get_highest_in_x(last_right_x, histogram, search_range, width, stride)

    return left_lane_pixel_pos, right_lane_pixel_pos


def get_real_world_curvature(roi_x, roi_y, y, ym_per_pix, xm_per_pix, side='left'):
    fit_cr = np.polyfit(roi_y * ym_per_pix, roi_x * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_cr[0] * y * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curverad


def get_sliding_window_combined(lane_patches, width, height):
    roi_x = []
    roi_y = []

    for i, patch in enumerate(lane_patches):
        roi_x.extend((patch[0], patch[0] + width, patch[0] + width, patch[0]))
        roi_y.extend((patch[1], patch[1], patch[1] + height, patch[1] + height))

    roi_x = np.array(roi_x)
    roi_y = np.array(roi_y)
    return roi_x, roi_y


def get_sliding_window(warped, y, height, width, search_range, stride):
    left_lane_patches = []
    right_lane_patches = []

    bottom_left, bottom_right = get_bottom_lane_marks(warped, width, search_range, stride)
    left_lane_patches.append((bottom_left, y - height))
    right_lane_patches.append((bottom_right, y - height))

    for i in range(int(y / height) - 1):
        current_y = left_lane_patches[-1][1] - height
        left, right = find_lane_patches(warped, left_lane_patches[-1][0], right_lane_patches[-1][0], current_y,
                                            height, search_range, width, stride)
        left_lane_patches.append((left, current_y))
        right_lane_patches.append((right, current_y))

    return left_lane_patches, right_lane_patches


def get_roi_poly_bird_view(undistort, warped, left_fitx, left_roi_y, right_fitx, right_roi_y):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, left_roi_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_roi_y])))])
    pts = np.hstack((pts_left, pts_right))
    # print('pts', pts)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    return color_warp


def get_polyfit(roi_y, roi_x):
    fit = np.polyfit(roi_y, roi_x, 2)
    roi_y = np.unique(roi_y)
    fitx = fit[0] * roi_y ** 2 + fit[1] * roi_y + fit[2]
    return fitx, roi_y, fit


def cal_offset_from_lane_center(left, right, x, xm_per_pix):
    lane_center = (left + right) / 2
    expected_center = x / 2
    offset = np.abs(lane_center - expected_center) * xm_per_pix
    return offset