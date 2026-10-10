from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = -1) -> np.ndarray:
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    pred = pred[valid].astype(np.int64)
    target = target[valid].astype(np.int64)
    keep = (pred >= 0) & (pred < num_classes)
    return np.bincount(
        target[keep] * num_classes + pred[keep], minlength=num_classes**2
    ).reshape(num_classes, num_classes)


def summarize_confusion(confusion: np.ndarray) -> dict[str, object]:
    tp = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = gt + predicted - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    accuracy = np.divide(tp, gt, out=np.full_like(tp, np.nan), where=gt > 0)
    def finite_or_none(values: np.ndarray) -> list[float | None]:
        return [float(value) if np.isfinite(value) else None for value in values]

    return {
        "mIoU": float(np.nanmean(iou)),
        "mAcc": float(np.nanmean(accuracy)),
        "allAcc": float(tp.sum() / max(gt.sum(), 1.0)),
        "class_iou": finite_or_none(iou),
        "class_accuracy": finite_or_none(accuracy),
    }
