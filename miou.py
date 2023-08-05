import numpy as np

def compute_iou(pred_mask, true_mask, num_classes):
    # 初始化每个类别的IoU和像素数
    ious = np.zeros(num_classes)
    pixel_counts = np.zeros(num_classes)

    for class_idx in range(num_classes):
        pred_class_mask = (pred_mask == class_idx)
        true_class_mask = (true_mask == class_idx)

        intersection = np.logical_and(pred_class_mask, true_class_mask).sum()
        union = np.logical_or(pred_class_mask, true_class_mask).sum()

        # 计算IoU
        iou = intersection / (union + 1e-6)  # 避免除以零的情况
        ious[class_idx] = iou
        pixel_counts[class_idx] = union

    # 计算miou
    miou = np.mean(ious)

    return miou






