# import libraries
import numpy as np
import cv2
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from src.utility_funcs import *
from src.lane_line_validation import *
from src.Line import Line
from config import *


mtx, dist = get_camera_mtx_dist('media/camera_cal/calibra*.jpg', corner_x, corner_y, x, y)


def detect_lane_line_pipeline(img, left_lane, right_lane):
    # global left_lane
    # global right_lane, z
    # global continue_miss_threshold

    undistort = cv2.undistort(img, mtx, dist, None, mtx)

    threshold_img = get_threshold_img(undistort, ksize)

    warped = cv2.warpPerspective(threshold_img, M, (x, y))

    # left_lane_patches is (window_start_x, window_low_y)
    left_lane_patches, right_lane_patches = get_sliding_window(warped, y, height, width, search_range, stride)

    # return a list of rectangles, 4 points of the it.
    left_roi_x, left_roi_y = get_sliding_window_combined(left_lane_patches, width, height)
    left_real_world_curvature = get_real_world_curvature(left_roi_x, left_roi_y, y, ym_per_pix, xm_per_pix, side='left')
    left_fitx, left_roi_y, left_coeff = get_polyfit(left_roi_y, left_roi_x)

    right_roi_x, right_roi_y = get_sliding_window_combined(right_lane_patches, width, height)
    right_real_world_curvature = get_real_world_curvature(right_roi_x, right_roi_y, y, ym_per_pix, xm_per_pix,
                                                          side='right')
    right_fitx, right_roi_y, right_coeff = get_polyfit(right_roi_y, right_roi_x)

    # check whether the detected lines are real line or not, use previous real lane line data.
    tf_real_lanes = are_lanes_detected(left_lane, left_real_world_curvature, left_fitx, \
                                       right_lane, right_real_world_curvature, right_fitx, curvature_diff_tolerance,
                                       distance_tolerance)

    # if lane lines are correctly detected, update them. If not, use the old ones
    if left_lane.continuous_missed_frames > continue_miss_threshold:
        left_lane.reset('left')
        right_lane.reset('right')
    elif tf_real_lanes:
        offset_from_lane_center = cal_offset_from_lane_center(left_fitx[-1], right_fitx[-1], x, xm_per_pix)
        # update parameters in lanes
        left_lane.update_params(left_fitx, left_roi_y, tf_real_lanes, left_real_world_curvature,
                                offset_from_lane_center, left_coeff)
        right_lane.update_params(right_fitx, right_roi_y, tf_real_lanes, right_real_world_curvature,
                                 offset_from_lane_center, right_coeff)
    else:
        left_lane.continuous_missed_frames += 1
    # average between the last n and draw them
    if left_lane.bestx is None or right_lane.bestx is None:
        result = undistort
    else:
        # color_warp = get_roi_poly_bird_view(undistort, warped, left_fitx, left_roi_y, right_fitx, right_roi_y)
        color_warp = get_roi_poly_bird_view(undistort, warped, left_lane.bestx, left_roi_y, \
                                            right_lane.bestx, right_roi_y)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)

    return result

def detect_lane_line_pipeline(img, left_lane, right_lane):
    # global left_lane
    # global right_lane, z
    # global continue_miss_threshold

    undistort = cv2.undistort(img, mtx, dist, None, mtx)

    threshold_img = get_threshold_img(undistort, ksize)

    warped = cv2.warpPerspective(threshold_img, M, (x, y))

    # left_lane_patches is (window_start_x, window_low_y)
    left_lane_patches, right_lane_patches = get_sliding_window(warped, y, height, width, search_range, stride)

    # return a list of rectangles, 4 points of the it.
    left_roi_x, left_roi_y = get_sliding_window_combined(left_lane_patches, width, height)
    left_real_world_curvature = get_real_world_curvature(left_roi_x, left_roi_y, y, ym_per_pix, xm_per_pix, side='left')
    left_fitx, left_roi_y, left_coeff = get_polyfit(left_roi_y, left_roi_x)

    right_roi_x, right_roi_y = get_sliding_window_combined(right_lane_patches, width, height)
    right_real_world_curvature = get_real_world_curvature(right_roi_x, right_roi_y, y, ym_per_pix, xm_per_pix,
                                                          side='right')
    right_fitx, right_roi_y, right_coeff = get_polyfit(right_roi_y, right_roi_x)

    # check whether the detected lines are real line or not, use previous real lane line data.
    tf_real_lanes = are_lanes_detected(left_lane, left_real_world_curvature, left_fitx, \
                                       right_lane, right_real_world_curvature, right_fitx, curvature_diff_tolerance,
                                       distance_tolerance)

    # if lane lines are correctly detected, update them. If not, use the old ones
    if left_lane.continuous_missed_frames > continue_miss_threshold:
        left_lane.reset('left')
        right_lane.reset('right')
    elif tf_real_lanes:
        offset_from_lane_center = cal_offset_from_lane_center(left_fitx[-1], right_fitx[-1], x, xm_per_pix)
        # update parameters in lanes
        left_lane.update_params(left_fitx, left_roi_y, tf_real_lanes, left_real_world_curvature,
                                offset_from_lane_center, left_coeff)
        right_lane.update_params(right_fitx, right_roi_y, tf_real_lanes, right_real_world_curvature,
                                 offset_from_lane_center, right_coeff)
    else:
        left_lane.continuous_missed_frames += 1
    # average between the last n and draw them
    if left_lane.bestx is None or right_lane.bestx is None:
        result = undistort
    else:
        # color_warp = get_roi_poly_bird_view(undistort, warped, left_fitx, left_roi_y, right_fitx, right_roi_y)
        color_warp = get_roi_poly_bird_view(undistort, warped, left_lane.bestx, left_roi_y, \
                                            right_lane.bestx, right_roi_y)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)

    return undistort, newwarp