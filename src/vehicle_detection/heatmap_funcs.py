import numpy as np
import cv2
from scipy.ndimage.measurements import label


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def mark_vehicles(image, hot_windows, near_heatmaps, heat_threshold):
    blank_heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = add_heat(blank_heat, hot_windows)

    # here is the heapmap need to be added into the collections
    heatmap = near_heatmaps.combine_recent_heatmap(heatmap)

    heatmap = apply_threshold(heatmap, heat_threshold)
    final_map = np.clip(heatmap, 0, 255)

    labels = label(final_map)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img
