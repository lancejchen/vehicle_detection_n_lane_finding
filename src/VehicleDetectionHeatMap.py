from collections import deque


class VehicleDetectionHeatmap:
    def __init__(self):
        recent_heatmaps = deque(maxlen=5)
        combined_heatmap