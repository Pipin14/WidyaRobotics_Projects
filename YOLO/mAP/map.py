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

def calculate_map(y_true, y_pred, iou_threshold = np.linspace(0.5, 0.95, 1)):
    n_classes = y_true.shape[1] - 4
    ap_list = np.zeros(n_classes)
    
    for c in range(n_classes):
        pred_class_boxes = y_pred[y_true[:, c+4] == 1]
        y_true_c = y_true[y_true[:, c+4] == 1]
        
        n_grond_truths = len(y_true_c)
        n_predictions = len(pred_class_boxes)
        
        if n_grond_truths == 0:
            continue
        
        if n_predictions == 0:
            ap_list[c] = 0
            continue
        pred_class_boxes = pred_class_boxes[np.argsort(pred_class_boxes[:, 0])[::-1]]
            
        tp = fp = np.zeros(n_predictions)
            
        for i in range(n_predictions):
            box_pred = pred_class_boxes[i, 1:5]
            iou_max = -1
            ground_truth_match = -1
            
        for j in range(n_grond_truths):
            if y_true_c[j, 0] != c:
                continue
            
            box_true = y_true_c[j, 1:5]
            iou = calculate_iou(box_pred, box_true)
                
            if iou > iou_max:
                iou_max = iou
                ground_truth_match = j
                
            if iou_max > iou_threshold:
                if y_true_c[ground_truth_match, 2] == 0:
                    tp[i] = 1
                    y_true_c[ground_truth_match, 2] = i
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
                    
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / n_grond_truths
        precision = tp / (tp + fp + 1e-8)

        aps = calculate_ap(precision, recall)
    
    # Compute mAP over all classes
    mAP = np.mean(aps)
    return mAP

def test():
    
  gt_class_boxes = np.array([
    [0, 10, 10, 20, 20, 1, 0],
    [0, 30, 30, 40, 40, 1, 0],
  ])
  
  pred_class_boxes = np.array([
    [0.9, 10, 10, 20, 20],
    [0.8, 35, 35, 45, 45],
  ])
  
  map_list = calculate_map(gt_class_boxes, pred_class_boxes)
  print(map_list)
  
if __name__ == '__main__':
  test()
