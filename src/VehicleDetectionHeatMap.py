from collections import deque
import numpy as np

class VehicleDetectionHeatMap:
    def __init__(self):
        self.recent_heatmaps = deque(maxlen=10)
        self.combined_heatmap = None

    def combine_recent_heatmap(self, new_heatmap):
        self.recent_heatmaps.append(new_heatmap)
        self.combined_heatmap = np.zeros_like(new_heatmap)
        # insert new heap map to deque
        for array in self.recent_heatmaps:
            self.combined_heatmap += array
        return self.combined_heatmap