import numpy as np
from map import calculate_map

# Define the ground truth boxes
gt_class_boxes = np.array([
    [0, 10, 10, 20, 1, 0],
    [0, 30, 30, 40, 1, 0],
    [0, 50, 50, 60, 1, 0],
    [0, 70, 70, 80, 1, 0],
])

# Define the predicted boxes
pred_class_boxes = np.array([
    [0.9, 10, 10, 20, 20],
    [0.8, 35, 35, 45, 45],
    [0.8, 50, 50, 60, 60],
    [0.7, 80, 80, 90, 90],
])


map = calculate_map(gt_class_boxes, pred_class_boxes)
print(map)
