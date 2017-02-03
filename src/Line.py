from collections import deque
import numpy as np


class Line:
    def __init__(self, name):
        self.name = name

        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=5)
        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # recent curvatures of the last n frames
        self.curvatures = deque(maxlen=5)
        self.radius_of_curvature = None

        self.continuous_missed_frames = 0
        self.line_base_pos = 0

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # polynomial coefficients for the most recent fit
        self.current_fit = deque(maxlen=5)

    def is_lane_detected(self, curvature):
        # have similar curvature
        self.radius_of_curvature = curvature
        average_curv = np.mean(self.curvatures)

        if np.abs(average_curv - self.radius_of_curvature) <= 1000:
            return True
        return False

    def update_params(self, x, y, tf_real, curvature, offset_from_lane_center, coeff):
        self.detected = tf_real
        self.continuous_missed_frames = 0
        self.allx = x
        self.ally = y
        self.radius_of_curvature = curvature

        self.recent_xfitted.append(self.allx)
        self.bestx = np.mean(self.recent_xfitted, axis=0)

        self.curvatures.append(self.radius_of_curvature)
        # self.line_base_pos = self.get_line_base_pos()
        self.continuous_missed_frames = 0
        self.line_base_pos = offset_from_lane_center

    def reset(self, name):
        self.__init__(name)
