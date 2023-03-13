import numpy as np


def calculate_iou(bbox1:np.array, bbox2:np.array):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculates the area of each boxes
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Calculates the coordinates of the intersection boxes
    x_intersect = max(0, min(x1+w1, x2 + w2) - max(x1, x2))
    y_intersect = max(0, min(y1+h1, y2 + h2) - max(y1, y2))
    area_intersect = x_intersect * y_intersect
    
    # Calculate of IoU
    iou = area_intersect / (area1 + area2 - area_intersect)
    return iou

def calculate_ap(precision, recall):
    """Calculates Average Precision (AP) given precision and recall values."""
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return ap

def calculate_map(gt_boxes, pred_boxes, iou_threshold=np.linspace(0.4, 0.95, 1)):
    """Calculates mean Average Precision (mAP) given ground truth and predicted boxes."""
    num_classes = max(gt_boxes[:, 4].max(), pred_boxes[:, 4].max()) + 1
    average_predictions = np.zeros(num_classes)
    aps = []

    for class_id in range(num_classes):
        gt_class_boxes = gt_boxes[gt_boxes[:, class_id+4] == class_id][:, :4]
        pred_class_boxes = pred_boxes[pred_boxes[:, class_id+4] == class_id][:, :4]

        num_gt_boxes = len(gt_class_boxes)
        num_pred_boxes = len(pred_class_boxes)
        
        # If there is no ground truth object and predictions objects, then average precision = 0
        if num_gt_boxes == 0 or num_pred_boxes == 0:
            average_predictions[class_id] = 0
            continue
        
        iou = np.zeros((num_pred_boxes, num_gt_boxes))

        for i, pred_box in enumerate(pred_class_boxes):
            for j, gt_box in enumerate(gt_class_boxes):
                iou[i, j] = calculate_iou(pred_box[:4], gt_box)
                
        # Sort the predictions based on the highest confidence score
        sorted_indices = np.argsort(-pred_class_boxes[:, 4])
        pred_class_boxes = pred_class_boxes[sorted_indices]
        
        # Initiate true positive array and false positive array 
        tp = np.zeros(num_pred_boxes)
        fp = np.zeros(num_pred_boxes)
        
        # Check eaach objects predictions
        for i in range(num_pred_boxes):
            gt_indices = np.where(iou[i] >= iou_threshold)[0]
            if len(gt_indices) == 0:
                fp[i] = 1
            else:
                max_iou_index = np.argmax(iou[i])
                if tp[max_iou_index] == 0:
                    tp[max_iou_index] = 1
                else:
                    fp[i] = 1
                    
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / num_gt_boxes
        precision = tp / (tp + fp + 1e-8)

        average_predictions = calculate_ap(precision, recall)
        aps.append(average_predictions)
    
    # Compute mAP over all classes
    mAP = np.mean(aps)
    return mAP
