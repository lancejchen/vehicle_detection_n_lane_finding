import numpy as np


def are_lane_parallel(left_curvature, right_curvature, curvature_diff_tolerance):
    diff = np.abs(left_curvature - right_curvature)
    if diff < curvature_diff_tolerance:
        return True
    return False


def are_right_distance_h(left_bottom, right_bottom, distance_tolerance):
    diff = np.abs(left_bottom - right_bottom)
    if diff < distance_tolerance:
        return True
    return False


# check if the real line was detected.
def are_lanes_detected(left_lane, left_curvature, left_fitx, right_lane, right_curvature, right_fitx,
                       curvature_diff_tolerance, distance_tolerance):
    # have similar curvature
    if len(left_lane.curvatures) != 0 and len(right_lane.curvatures) != 0:
        if not left_lane.is_lane_detected(left_curvature):
            return False
        if not right_lane.is_lane_detected(right_curvature):
            return False

    # roughly parallel
    if not are_lane_parallel(left_curvature, right_curvature, curvature_diff_tolerance):
        return False

    # separated by approximately the right distance horizontally
    if not are_right_distance_h(left_fitx[-1], right_fitx[-1], distance_tolerance):
        return False
    return True
